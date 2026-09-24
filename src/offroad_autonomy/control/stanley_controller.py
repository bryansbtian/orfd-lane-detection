"""Stanley lateral controller with curvature feedforward and speed planning.

Positive steering is to the right, as in BeamNG.

Everything happens on the ground, not in the image
--------------------------------------------------
Image rows are extremely non-linear in distance on the bumper camera (6-40 m
of road sits in ~28 rows), so gains tuned in pixels would mean something
different at every range. Every path point is projected onto the ground
first and ``right(f) = a + b*f + c*f^2`` is fitted in metres, which gives
offset, heading ``atan(right'(f))`` and curvature
``kappa(f) = right'' / (1 + right'^2)^1.5`` at any distance ahead.

Lateral law
-----------
The loop runs at ~4 FPS with ~0.35 s from capture to actuation, so the
car is ``f0 = v * latency`` further along by the time a command acts. All
terms are evaluated from there::

    L     = clip(base + time*v, min, max) / (1 + g*|kappa|*...)  adaptive lookahead
    kff   = mean curvature over [f0, f0 + L]                        feedforward
    psi   = atan(ref'(f0)) - arc heading at f0                      heading error
    e     = ref(f0) - arc offset at f0, soft deadzone               cross-track
    R     = L_speed, shortened once |e| is past a recovery band     rejoin distance

    wheel = atan(wheelbase * kff)        steer for the arc before reaching it
          + k_psi * psi                  align with the path
          + atan(k * e / R)              close the offset over R metres

``ref`` is the path pulled toward the centre of the traversable mask
wherever it runs within ``edge_margin_m`` of a trail edge, and "arc" is
where the car will be after ``f0`` metres on its current steering - so the
error is read at the pose the command will actually act on.

The lookahead shrinks in tight curves, so the feedforward only averages
over the bend the car is in - not the reverse bend of an S behind it,
which is what makes a long lookahead cut corners.

Longitudinal law
----------------
The target is the minimum of: the cruise speed; a curvature profile
``v = sqrt(a_lat / |kappa(f)|)`` over the whole visible path, reached by
braking at ``curve_decel`` *before* each bend; a stopping-distance limit
at the end of the path; a saturation cut that slows the car whenever
the applied - or the required - steering approaches lock; and three
trail-keeping cuts: near a trail edge, while the cross-track error is
growing, and when the measured lateral acceleration exceeds its budget. The target then
rises by at most ``max_target_speed_increase_mps`` per frame, so the car
does not sprint through the nearly-straight middle of an S. A steering cap
that tightens with speed keeps full lock for low-speed manoeuvring only.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import (
    ControlCommand,
    PathPlan,
    PipelineConfig,
    SteeringDebug,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.control")

_MPS_PER_MPH = 0.44704
#: Farther ground points are too compressed in the image to trust.
_MAX_GROUND_RANGE_M = 30.0
_PROFILE_SAMPLES = 24
#: Beyond this much over target a descent is running away, so braking is no
#: longer capped at comfort level.
_OVERSPEED_HARD_BRAKE_MPS = 0.75
_BOUNDARY_SAMPLES_M = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0)
#: 4 m is the first distance at which a ~6 m trail's edges are inside the
#: 120 deg view, so nearer edges cannot be measured at all.
_NEAR_FIELD_M = 4.0
#: Masks stop a pixel or two short of the border, so a run ending here is the
#: field of view, not the trail edge.
_BORDER_PX = 3
#: Depth vetoes and specks punch small holes into real masks; they must not
#: split the trail (same rule as the planner).
_EDGE_MERGE_GAP_M = 0.6
_EDGE_EMA = 0.5


def path_to_ground(
    camera: CameraModel, centerline: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(forward_m, right_m, keep)``, measured from the camera, which on this
    rig is the front of the vehicle."""
    return camera.image_to_ground(centerline, max_range_m=_MAX_GROUND_RANGE_M)


class _GroundPath:
    def __init__(self, forward: np.ndarray, right: np.ndarray) -> None:
        # A cubic was tried (scripts/sbend_sim.py) and tracked S-bends worse:
        # it reacts to the far end of the path, the least precise part of it.
        degree = 1
        if len(forward) >= 4:
            degree = 2
        self.poly = np.poly1d(np.polyfit(forward, right, deg=degree))
        self.d1 = self.poly.deriv(1)
        self.d2 = self.poly.deriv(2)
        self.start = float(forward.min())
        self.end = float(forward.max())

    def right(self, f):
        return self.poly(f)

    def slope(self, f):
        return self.d1(f)

    def raw_curvature(self, f):
        return self.d2(f) / np.power(1.0 + np.square(self.slope(f)), 1.5)

    def curvature(self, f, max_curvature: float = np.inf, trust: float = 1.0):
        """A quadratic fitted to a few metres of path can report a 1 m radius
        on a straight road. Nothing the Hopper can follow is tighter than its
        own turning circle, and a short path's second derivative is mostly
        noise, so the result is clamped and scaled by trust.
        """
        return np.clip(self.raw_curvature(f), -max_curvature, max_curvature) * trust


@dataclass
class TrailBoundaries:
    """Trail edges along the vehicle's predicted arc; lateral is + right."""

    forward: np.ndarray
    left: np.ndarray
    right: np.ndarray
    vehicle: np.ndarray
    edge_px: list = field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.left + self.right)

    def near(self) -> np.ndarray:
        sel = self.forward <= _NEAR_FIELD_M
        if sel.any():
            return sel
        return np.arange(len(self.forward)) < 1

    def distances(self) -> tuple[float, float]:
        """Median, not minimum, over the near samples: one row where the mask
        is ragged must not read as the trail closing in."""
        sel = self.near()

        def robust(values: np.ndarray) -> float:
            # With a 120 deg lens a wide trail's edges are outside the image
            # for the first metres, so only in-view samples count.
            seen = values[np.isfinite(values)]
            if len(seen) == 0:
                return float("inf")
            return float(np.median(seen))

        return (
            robust(self.vehicle[sel] - self.left[sel]),
            robust(self.right[sel] - self.vehicle[sel]),
        )


def measure_trail(
    camera: CameraModel,
    mask: np.ndarray,
    arc_curvature: float,
    samples_m=_BOUNDARY_SAMPLES_M,
) -> TrailBoundaries | None:
    """Edges along the arc the vehicle is driving, not along the planned path,
    because the arc is where the body will actually be. A vehicle already off
    the mask gets a negative distance to the near edge.
    """
    if mask is None or not mask.any():
        return None
    h, w = mask.shape[:2]
    fs, lefts, rights, arcs, edge_px = [], [], [], [], []
    for f in samples_m:
        r_arc = 0.5 * arc_curvature * f * f
        u, v = camera.ground_to_image(np.array([f]), np.array([r_arc]))[0]
        row = int(round(v))
        if not 0 <= row < h:
            continue
        cols = np.flatnonzero(mask[row])
        if len(cols) == 0:
            continue
        merge_px = max(1.0, _EDGE_MERGE_GAP_M * camera.focal_px / max(f, 0.1))
        splits = np.flatnonzero(np.diff(cols) > merge_px) + 1
        runs = [(int(g[0]), int(g[-1])) for g in np.split(cols, splits)]
        u_c = float(np.clip(u, 0, w - 1))

        def gap(run):
            if run[0] <= u_c <= run[1]:
                return 0.0
            return min(abs(run[0] - u_c), abs(run[1] - u_c))

        lo, hi = min(runs, key=gap)
        ground_f, ground_r, ok = camera.image_to_ground(
            np.array([[lo - 0.5, row], [hi + 0.5, row]], dtype=np.float64)
        )
        if not ok.all():
            continue
        left_edge = -np.inf
        if lo > _BORDER_PX:
            left_edge = float(ground_r[0])
        right_edge = np.inf
        if hi < w - 1 - _BORDER_PX:
            right_edge = float(ground_r[1])
        fs.append(f)
        lefts.append(left_edge)
        rights.append(right_edge)
        arcs.append(r_arc)
        edge_px.append(((float(lo), float(row)), (float(hi), float(row))))
    if not fs:
        return None
    return TrailBoundaries(np.array(fs), np.array(lefts), np.array(rights), np.array(arcs), edge_px)


class StanleyController:
    def __init__(self, config: PipelineConfig, camera: CameraModel | None = None) -> None:
        self._k = config.stanley_gain_k
        self._k_soft = config.stanley_softening
        self._heading_gain = config.stanley_heading_gain
        self._ff_gain = float(max(config.curvature_feedforward_gain, 0.0))
        self._steering_alpha = float(np.clip(config.steering_ema_alpha, 0.0, 1.0))
        self._max_steering_delta = float(max(config.max_steering_delta, 0.0))
        self._la_base = float(config.lookahead_base_m)
        self._la_time = float(max(config.lookahead_time_s, 0.0))
        self._la_min = float(max(config.lookahead_min_m, 0.5))
        self._la_max = float(max(config.lookahead_max_m, self._la_min))
        self._la_curvature_gain = float(max(config.lookahead_curvature_gain, 0.0))
        self._latency = float(max(config.control_latency_s, 0.0))
        self._cte_tolerance = float(max(config.cross_track_tolerance_m, 0.0))
        self._full_authority_speed = float(max(config.steer_full_authority_speed_mps, 0.0))
        self._speed_falloff = float(max(config.steer_speed_falloff, 0.0))
        self._cap_full_speed = float(max(config.steer_cap_full_speed_mps, 0.0))
        self._cap_high_speed = float(
            max(config.steer_cap_high_speed_mps, self._cap_full_speed + 1e-3)
        )
        self._cap_at_high = float(np.clip(config.steer_cap_at_high_speed, 0.05, 1.0))
        self._wheelbase = float(max(config.wheelbase_m, 0.5))
        self._max_wheel_angle = math.radians(max(config.max_wheel_angle_deg, 1.0))
        self._path_end_margin = float(max(config.path_end_margin_m, 0.0))
        self._path_end_decel = float(max(config.path_end_decel_mps2, 0.1))
        self._lat_accel = float(max(config.max_lateral_accel_mps2, 0.1))
        self._curve_decel = float(max(config.curve_decel_mps2, 0.1))
        self._sat_start = float(np.clip(config.saturation_start, 0.0, 0.99))
        self._sat_min_scale = float(np.clip(config.saturation_min_speed_scale, 0.0, 1.0))
        self._target_speed = min(config.target_speed_mph, config.speed_limit_mph) * _MPS_PER_MPH
        self._speed_limit = config.speed_limit_mph * _MPS_PER_MPH
        self._min_turn_speed = min(config.min_turn_speed_mph, config.speed_limit_mph) * _MPS_PER_MPH
        self._max_throttle = config.max_throttle
        self._max_brake = config.max_brake
        self._speed_kp = config.speed_kp
        self._clearance_stop_m = float(max(config.clearance_stop_m, 0.0))
        self._clearance_slow_m = float(max(config.clearance_slow_m, self._clearance_stop_m))
        # The path is in the segmentation view's pixels, and stitched mode
        # has its own camera, so the default is only right for single views.
        self._camera = camera or CameraModel(
            config.segmentation_camera, config.preprocess_width, config.preprocess_height
        )
        self._prev_steering = 0.0
        self._prev_target: float | None = None
        self._comfort_brake = float(np.clip(config.comfort_brake, 0.0, config.max_brake))
        self._min_curvature_span = float(max(config.min_curvature_span_m, 0.5))
        self._max_curvature = math.tan(self._max_wheel_angle) / self._wheelbase
        self._target_ramp = float(max(config.max_target_speed_increase_mps, 0.0))
        self._half_width = float(max(config.vehicle_half_width_m, 0.0))
        self._xte_la_gain = float(max(config.cross_track_lookahead_gain, 0.0))
        self._rejoin_min = float(max(config.cross_track_min_lookahead_m, 0.5))
        self._xte_recovery = float(max(config.cross_track_recovery_m, 0.0))
        self._edge_margin = float(max(config.edge_margin_m, 1e-3))
        self._centering_gain = float(np.clip(config.edge_centering_gain, 0.0, 1.0))
        self._edge_speed_cut = float(np.clip(config.edge_speed_reduction, 0.0, 1.0))
        self._drift_speed_gain = float(max(config.drift_speed_gain, 0.0))
        self._lat_accel_limit = float(max(config.max_measured_lateral_accel_mps2, 0.05))
        self._soft_deadzone = bool(config.cross_track_soft_deadzone)
        self._predict_pose = bool(config.latency_pose_prediction)
        self._prev_abs_cte: float | None = None
        self._prev_time: float | None = None
        self._edge_state: list[float | None] = [None, None]
        self._cte_rate = 0.0

        if config.target_speed_mph > config.speed_limit_mph:
            logger.warning(
                "Target speed %.1f mph exceeds limit %.1f mph - clamping to the speed limit",
                config.target_speed_mph,
                config.speed_limit_mph,
            )

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        debug, path = self._lateral_control(plan, state)
        steering = self._smooth_steering(debug.desired_steering)
        debug.final_steering = steering
        throttle, brake = self._longitudinal_control(plan, state, path, debug)

        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
            debug=debug,
        )

    def lookahead_distance(self, speed_mps: float, curvature: float = 0.0) -> float:
        """Shortened in tight bends so the feedforward averages over the bend
        the car is in, not the reverse half of an S behind it."""
        base = float(
            np.clip(self._la_base + self._la_time * max(speed_mps, 0.0), self._la_min, self._la_max)
        )
        shrunk = base / (1.0 + self._la_curvature_gain * abs(curvature) * base)
        return float(np.clip(shrunk, self._la_min, self._la_max))

    def steering_authority(self, speed_mps: float) -> float:
        excess = max(0.0, speed_mps - self._full_authority_speed)
        return 1.0 / (1.0 + self._speed_falloff * excess)

    def steering_cap(self, speed_mps: float) -> float:
        """Full lock is for crawling; at speed it rolls the vehicle."""
        span = self._cap_high_speed - self._cap_full_speed
        ratio = float(np.clip((speed_mps - self._cap_full_speed) / span, 0.0, 1.0))
        return 1.0 - ratio * (1.0 - self._cap_at_high)

    def wheel_to_command(self, wheel_angle: float) -> float:
        return wheel_angle / self._max_wheel_angle

    def _lateral_control(
        self, plan: PathPlan, state: VehicleState
    ) -> tuple[SteeringDebug, _GroundPath | None]:
        speed = max(state.speed_mps, 0.0)
        debug = SteeringDebug(
            lookahead_m=self.lookahead_distance(speed),
            authority=self.steering_authority(speed),
        )
        if len(plan.centerline) < 2:
            return debug, None

        forward, right, keep = path_to_ground(self._camera, plan.centerline)
        if int(keep.sum()) < 2:
            return debug, None
        order = np.argsort(forward[keep])
        fwd = forward[keep][order]
        px = np.asarray(plan.centerline, dtype=np.float64)[keep][order]
        path = _GroundPath(fwd, right[keep][order])

        # Every term is read where the car will be when the command lands.
        f0 = min(speed * self._latency, path.end)
        trust = self._curvature_trust(path, f0)
        kappa = lambda f: path.curvature(f, self._max_curvature, trust)  # noqa: E731
        lookahead = self.lookahead_distance(speed, float(kappa(f0)))
        f_end = min(f0 + lookahead, path.end)
        window = np.linspace(f0, max(f_end, f0 + 1e-3), 8)
        kappa_ff = float(np.mean(kappa(window)))

        # The planner's path is already the per-row centre of the road; the
        # centred reference only bites where the fit or the mask shape leaves
        # it near one side.
        arc = math.tan(self._prev_steering * self._max_wheel_angle) / self._wheelbase
        trail = measure_trail(self._camera, plan.planner_mask, arc)
        ref = self._centred_reference(path, trail, debug)

        # Reading the offset at the bumper "now" left ~0.8 s of loop delay
        # uncompensated and the car weaved +-1 m with a ~5 s period
        # (scripts/sbend_sim.py), so both errors are read at the predicted pose.
        arc_offset = 0.0
        arc_heading = 0.0
        cte_at = 0.0
        if self._predict_pose:
            arc_offset = 0.5 * arc * f0 * f0
            arc_heading = arc * f0
            cte_at = f0
        debug.centering_shift_m = float(ref.right(f0) - path.right(f0))
        heading_error = math.atan(float(ref.slope(f0))) - arc_heading
        cte = float(ref.right(cte_at)) - arc_offset
        cte_eff = self._deadzone(cte)

        # The rejoin distance shortens only past ``cross_track_recovery_m``:
        # shortening it continuously raises the loop gain everywhere, and at
        # 3 FPS that turned tracking unstable (scripts/sbend_sim.py).
        excess = max(0.0, abs(cte_eff) - self._xte_recovery)
        rejoin = float(
            np.clip(
                self.lookahead_distance(speed) / (1.0 + self._xte_la_gain * excess),
                self._rejoin_min,
                self._la_max,
            )
        )

        feedforward = self._ff_gain * math.atan(self._wheelbase * kappa_ff)
        wheel = (
            feedforward + self._heading_gain * heading_error + math.atan2(self._k * cte_eff, rejoin)
        )
        command = debug.authority * self.wheel_to_command(wheel)
        cap = self.steering_cap(speed)

        debug.cross_track_m = cte
        debug.rejoin_m = rejoin
        debug.heading_error_rad = heading_error
        self._record_trail(trail, speed, debug)
        debug.curvature = kappa_ff
        debug.feedforward_steering = self.wheel_to_command(feedforward)
        debug.lookahead_m = lookahead
        debug.desired_steering = float(np.clip(command, -cap, cap))
        debug.path_end_m = path.end
        debug.lookahead_px = (
            float(np.interp(f_end, fwd, px[:, 0])),
            float(np.interp(f_end, fwd, px[:, 1])),
        )
        return debug, path

    def _curvature_trust(self, path: _GroundPath, f0: float) -> float:
        """A short path's second derivative is mostly noise, so trust grows
        with how far the path reaches past the preview point."""
        span = max(0.0, path.end - f0)
        return float(min(1.0, span / self._min_curvature_span) ** 2)

    def _deadzone(self, cte: float) -> float:
        """A hard band let the car drift to its edge and then kicked it back
        (a 3-4 s, 0.9 m weave in scripts/sbend_sim.py); the soft form
        ``e - tol*tanh(e/tol)`` has no switch to kick against."""
        tol = self._cte_tolerance
        if tol <= 0.0:
            return cte
        if self._soft_deadzone:
            return cte - tol * math.tanh(cte / tol)
        if abs(cte) <= tol:
            return 0.0
        return cte - math.copysign(tol, cte)

    def _centred_reference(
        self, path: _GroundPath, trail: TrailBoundaries | None, debug: SteeringDebug
    ) -> _GroundPath:
        """Shift proportional to how deep inside ``edge_margin_m`` the path
        runs, so a centred path is left exactly as planned."""
        if trail is None or self._centering_gain <= 0.0:
            return path
        f = trail.forward[(trail.forward >= path.start - 0.5) & (trail.forward <= path.end)]
        if len(f) < 3:
            return path
        sel = np.isin(trail.forward, f)
        left, right = trail.left[sel], trail.right[sel]
        bounded = np.isfinite(left) & np.isfinite(right)
        if bounded.sum() < 2:
            return path
        r_path = path.right(f)
        clearance = np.minimum(r_path - left, right - r_path) - self._half_width
        weight = self._centering_gain * np.clip(
            (self._edge_margin - clearance) / self._edge_margin, 0.0, 1.0
        )
        weight[~bounded] = 0.0
        centre = r_path.copy()
        centre[bounded] = 0.5 * (left[bounded] + right[bounded])
        shift = weight * (centre - r_path)
        # With a 120 deg lens a trail wider than ~3.5 m runs off the sides of
        # the image for the first couple of metres: no edge is visible there.
        # Carry the nearest visible shift back toward the car rather than
        # dropping the correction exactly where the car is.
        first = int(np.argmax(bounded))
        shift[:first] = shift[first]
        r_ref = r_path + shift
        if not np.any(shift):
            return path
        extra = np.array([])
        if path.end > f[-1] + 0.5:
            extra = np.linspace(f[-1], path.end, 6)[1:]
        return _GroundPath(np.concatenate([f, extra]), np.concatenate([r_ref, path.right(extra)]))

    def _record_trail(
        self, trail: TrailBoundaries | None, speed: float, debug: SteeringDebug
    ) -> None:
        applied_curvature = math.tan(self._prev_steering * self._max_wheel_angle) / self._wheelbase
        debug.lateral_accel_mps2 = float(speed * speed * abs(applied_curvature))
        if trail is None:
            return
        to_left, to_right = self._smooth_edges(*trail.distances())
        debug.left_boundary_m = to_left
        debug.right_boundary_m = to_right
        clearance = min(to_left, to_right) - self._half_width
        debug.edge_clearance_m = float(clearance)
        debug.edge_risk = float(
            np.clip((self._edge_margin - clearance) / self._edge_margin, 0.0, 1.0)
        )
        debug.boundary_px = trail.edge_px

    def _smooth_edges(self, to_left: float, to_right: float) -> tuple[float, float]:
        """An edge leaving the view resets its filter rather than averaging
        a real distance with infinity."""
        out = []
        for i, value in enumerate((to_left, to_right)):
            prev = self._edge_state[i]
            if not math.isfinite(value) or prev is None or not math.isfinite(prev):
                smoothed = value
            else:
                smoothed = _EDGE_EMA * value + (1.0 - _EDGE_EMA) * prev
            self._edge_state[i] = smoothed
            out.append(smoothed)
        return out[0], out[1]

    def _smooth_steering(self, steering_cmd: float) -> float:
        blended = self._prev_steering + self._steering_alpha * (steering_cmd - self._prev_steering)
        delta = float(
            np.clip(
                blended - self._prev_steering,
                -self._max_steering_delta,
                self._max_steering_delta,
            )
        )
        self._prev_steering = float(np.clip(self._prev_steering + delta, -1.0, 1.0))
        return self._prev_steering

    def _longitudinal_control(
        self,
        plan: PathPlan,
        state: VehicleState,
        path: _GroundPath | None,
        debug: SteeringDebug,
    ) -> tuple[float, float]:
        if plan.speed_scale <= 0.0:
            # The perception gate has nothing left to drive: stop now rather
            # than coasting along a path nobody perceived.
            debug.target_speed_mps, debug.speed_reason = 0.0, "gate stop"
            return 0.0, self._max_brake

        if state.speed_mps > self._speed_limit:
            overspeed = state.speed_mps - self._speed_limit
            debug.target_speed_mps, debug.speed_reason = self._speed_limit, "limit"
            if overspeed > _OVERSPEED_HARD_BRAKE_MPS:
                return 0.0, self._max_brake
            return 0.0, min(self._speed_kp * overspeed, self._max_brake)

        limits = {"cruise": self._target_speed * min(plan.speed_scale, 1.0)}
        if plan.speed_scale < 1.0:
            limits["gate hold"] = limits.pop("cruise")

        required = 0.0
        if path is not None:
            curve_speed, required, debug.max_curvature_ahead = self._curve_speed(
                path, state.speed_mps
            )
            limits["curve"] = curve_speed
            limits["path end"] = self._path_end_speed(path.end)

        # The bend ahead counts as much as the steering applied now, so the
        # car slows before it runs out of lock, not after.
        saturation = max(abs(debug.final_steering), required)
        debug.saturation = float(min(saturation, 1.0))
        sat_scale = self._saturation_scale(saturation)
        if sat_scale < 1.0:
            limits["saturation"] = self._target_speed * sat_scale

        limits["clearance"] = self._clearance_limited_speed(
            self._target_speed, plan.min_clearance_m
        )

        # Staying on the trail beats speed: slow near an edge, while the
        # offset is growing, and when the car is cornering harder than the
        # budget allows.
        if debug.edge_risk > 0.0:
            limits["edge"] = self._target_speed * (1.0 - self._edge_speed_cut * debug.edge_risk)
        self._update_cte_rate(abs(debug.cross_track_m))
        debug.cross_track_rate_mps = self._cte_rate
        if self._cte_rate > 0.0 and self._drift_speed_gain > 0.0:
            limits["drift"] = self._target_speed / (1.0 + self._drift_speed_gain * self._cte_rate)
        applied_curvature = (
            abs(math.tan(debug.final_steering * self._max_wheel_angle)) / self._wheelbase
        )
        if applied_curvature > 1e-3:
            limits["lat accel"] = math.sqrt(self._lat_accel_limit / applied_curvature)

        reason = min(limits, key=limits.get)
        target_speed = limits[reason]
        # Soft limits never go below crawl speed: a stationary vehicle cannot
        # steer out of the situation that slowed it. Hard limits (path end,
        # gate) may.
        if reason in ("curve", "saturation", "edge", "drift", "lat accel"):
            target_speed = max(target_speed, min(self._min_turn_speed, self._target_speed))
        # Between the two halves of an S the fitted path looks nearly straight
        # for a moment; snapping back to cruise there meets the reverse bend
        # fast and brakes inside it. Ramping from the vehicle's own speed also
        # stops a standing start from launching straight at cruise.
        previous = state.speed_mps
        if self._prev_target is not None:
            previous = self._prev_target
        if target_speed > previous + self._target_ramp:
            reason = "ramp up"
            target_speed = previous + self._target_ramp
        self._prev_target = target_speed
        debug.target_speed_mps = float(target_speed)
        debug.speed_reason = reason

        speed_err = target_speed - state.speed_mps
        if speed_err > 0:
            throttle = min(self._speed_kp * speed_err, self._max_throttle * sat_scale)
            brake = 0.0
        else:
            throttle = 0.0
            # Slowing for a bend is routine: brake gently, because hard braking
            # pitches the nose - and the camera - and shakes the path. Only a
            # hard limit (path end, clearance) gets the full brake.
            # Well over target (a descent, or a limit that dropped suddenly)
            # is not routine either: comfort braking could not hold the Hopper
            # on a slope in BeamNG, where it ran at 5.7 m/s against 1.3.
            hard = reason in ("path end", "clearance") or -speed_err > _OVERSPEED_HARD_BRAKE_MPS
            ceiling = self._comfort_brake
            if hard:
                ceiling = self._max_brake
            brake = min(self._speed_kp * abs(speed_err), ceiling)
        return throttle, brake

    def _update_cte_rate(self, abs_cte: float) -> None:
        now = time.perf_counter()
        if self._prev_abs_cte is not None and self._prev_time is not None:
            dt = float(np.clip(now - self._prev_time, 0.1, 0.5))
            rate = (abs_cte - self._prev_abs_cte) / dt
            self._cte_rate = 0.5 * self._cte_rate + 0.5 * rate
        self._prev_abs_cte, self._prev_time = abs_cte, now

    def _curve_speed(self, path: _GroundPath, speed: float) -> tuple[float, float, float]:
        """``(speed limit, steering the sharpest bend needs, its |curvature|)``.

        Each point ahead allows ``sqrt(v_curve^2 + 2*decel*distance)`` so the
        car is already slow on entry rather than braking inside the bend.
        """
        f0 = min(speed * self._latency, path.end)
        f = np.linspace(f0, path.end, _PROFILE_SAMPLES)
        kappa = np.abs(path.curvature(f, self._max_curvature, self._curvature_trust(path, f0)))
        v_curve = np.sqrt(self._lat_accel / np.maximum(kappa, 1e-4))
        allowed = np.sqrt(np.square(v_curve) + 2.0 * self._curve_decel * (f - f0))
        k_max = float(kappa.max())
        required = self.wheel_to_command(math.atan(self._wheelbase * k_max))
        return float(allowed.min()), float(min(required, 1.0)), k_max

    def _saturation_scale(self, saturation: float) -> float:
        if saturation <= self._sat_start:
            return 1.0
        ratio = min(1.0, (saturation - self._sat_start) / (1.0 - self._sat_start))
        return 1.0 - ratio * (1.0 - self._sat_min_scale)

    def _path_end_speed(self, path_end_m: float) -> float:
        """A path that ends short (a junction, a bush across the track) means
        there is no perceived road beyond it, so the car must never arrive
        there moving."""
        if not math.isfinite(path_end_m):
            return float("inf")
        room = max(0.0, path_end_m - self._path_end_margin)
        return math.sqrt(2.0 * self._path_end_decel * room)

    def _clearance_limited_speed(self, target_speed: float, clearance_m: float) -> float:
        """Slows to crawl rather than stopping, so the planner can still steer
        around the obstacle; a full halt is the safe stop's job."""
        if not math.isfinite(clearance_m) or clearance_m >= self._clearance_slow_m:
            return target_speed

        span = self._clearance_slow_m - self._clearance_stop_m
        if span <= 0.0 or clearance_m <= self._clearance_stop_m:
            return min(target_speed, self._min_turn_speed)

        ratio = (clearance_m - self._clearance_stop_m) / span
        limited = self._min_turn_speed + ratio * (target_speed - self._min_turn_speed)
        return min(target_speed, limited)

    def reset(self) -> None:
        self._prev_steering = 0.0
        self._prev_target = None
        self._prev_abs_cte = None
        self._prev_time = None
        self._cte_rate = 0.0
        self._edge_state = [None, None]
