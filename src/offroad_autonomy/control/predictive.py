"""Shared core of the predictive steering controllers (``control.mode: mppi``
and ``mpc``).

Both predict the vehicle with a kinematic bicycle about the rear axle, in the
camera's ground frame that the grid planner's metric path is in (forward,
right; the camera ground point at the origin, heading straight ahead), and
choose steering commands over a horizon. Two properties of BeamNG are built
into the prediction rather than corrected for afterwards:

* a command takes effect ``control.control_latency_s`` after it is sent, so
  the first steps of every prediction replay the commands already sent;
* the steering input slews toward the command at ``steer_slew_per_s``.

Only steering is predictive. Speed, the 6 mph cap and every stop come from
the Stanley controller's longitudinal law, unchanged, so a comparison between
modes compares steering and nothing else.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from offroad_autonomy.control.predictive_config import PredictiveConfig
from offroad_autonomy.control.stanley_controller import StanleyController, _SampledPath
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import ControlCommand, PathPlan, PipelineConfig, VehicleState

#: Step used to integrate the steering input between control calls.
_ACTUATOR_STEP_S = 0.01


class SteeringActuator:
    """Estimate of BeamNG's steering input, from the commands this controller
    sent: each takes effect ``latency_s`` after it is sent, and the input
    slews toward it at ``slew_per_s``. BeamNG does not report the input, and
    the estimate is what every prediction starts from."""

    def __init__(self, latency_s: float, slew_per_s: float) -> None:
        self._latency = max(float(latency_s), 0.0)
        self._slew = float(slew_per_s)
        self.reset()

    def reset(self) -> None:
        self._sent: deque[tuple[float, float]] = deque()
        self.input = 0.0
        self._time: float | None = None

    def target_at(self, t: float) -> float:
        """The command in effect at time ``t``: the last one sent at or before
        ``t - latency``, or straight ahead if none was."""
        effective = t - self._latency
        target = 0.0
        for sent_at, command in self._sent:
            if sent_at > effective:
                break
            target = command
        return target

    def advance(self, now: float) -> None:
        if self._time is None:
            self._time = now
            return
        t = self._time
        while t < now:
            step = min(_ACTUATOR_STEP_S, now - t)
            target = self.target_at(t)
            self.input += float(np.clip(target - self.input, -self._slew * step, self._slew * step))
            t += step
        self._time = now

    def record(self, now: float, command: float) -> None:
        self._sent.append((now, float(command)))
        # Keep one command older than the latency window: it is still in effect.
        while len(self._sent) > 1 and self._sent[1][0] <= now - self._latency - 1.0:
            self._sent.popleft()

    def committed(self, now: float, dt: float, steps: int) -> np.ndarray:
        """Commands already in flight for the first model steps: the step
        starting at ``now + k*dt`` uses what was sent ``latency`` before it.
        Steps from ``latency`` on belong to the commands still to be chosen."""
        count = min(int(math.ceil(self._latency / dt - 1e-9)), steps)
        return np.array([self.target_at(now + k * dt) for k in range(count)], dtype=np.float64)


class Geometry:
    """Vehicle constants for prediction."""

    def __init__(self, config: PipelineConfig) -> None:
        self.wheelbase = float(config.wheelbase_m)
        self.max_wheel = math.radians(config.max_wheel_angle_deg)
        self.camera_ahead = float(config.camera_ahead_of_rear_axle_m)
        self.half_width = float(config.vehicle_half_width_m)


def rollout(
    targets: np.ndarray,
    input0: float,
    speed: float,
    dt: float,
    slew_per_s: float,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kinematic bicycle for every row of ``targets`` (``(samples, steps)``
    steering commands, one per step). Returns the camera point's forward and
    right and the heading after each step, each ``(samples, steps)``; heading
    is positive turning right, like the commands."""
    samples, steps = targets.shape
    step = slew_per_s * dt
    x = np.full(samples, -geometry.camera_ahead)
    y = np.zeros(samples)
    yaw = np.zeros(samples)
    steer = np.full(samples, float(input0))
    fc = np.empty((samples, steps))
    rc = np.empty((samples, steps))
    heading = np.empty((samples, steps))
    turn = speed * dt / geometry.wheelbase
    for k in range(steps):
        steer = steer + np.clip(targets[:, k] - steer, -step, step)
        yaw = yaw + turn * np.tan(steer * geometry.max_wheel)
        x = x + speed * dt * np.cos(yaw)
        y = y + speed * dt * np.sin(yaw)
        fc[:, k] = x + geometry.camera_ahead * np.cos(yaw)
        rc[:, k] = y + geometry.camera_ahead * np.sin(yaw)
        heading[:, k] = yaw
    return fc, rc, heading


def tracking_residuals(
    path: _SampledPath,
    fc: np.ndarray,
    rc: np.ndarray,
    heading: np.ndarray,
    cfg: PredictiveConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted lateral and heading errors of the camera point against the
    planned path, so their squares are the tracking cost. Steps past the
    path's end count less: nothing was perceived there to track."""
    lateral = rc - path.right(fc)
    path_heading = np.arctan(path.slope(fc))
    head = np.angle(np.exp(1j * (heading - path_heading)))
    weight = np.where(fc <= path.end, 1.0, cfg.beyond_path_weight)
    root = np.sqrt(weight)
    return (
        root * math.sqrt(cfg.weight_lateral) * lateral,
        root * math.sqrt(cfg.weight_heading) * head,
    )


def plan_path(plan: PathPlan) -> _SampledPath | None:
    """The grid planner's metric path, or ``None`` when there is none to
    track (other planner modes, or a gate stop)."""
    if plan.ground_path is None or plan.ground_curvature is None or plan.speed_scale <= 0.0:
        return None
    if len(plan.ground_path) < 2:
        return None
    path = _SampledPath(plan.ground_path[:, 0], plan.ground_path[:, 1], plan.ground_curvature)
    if len(path.forward) < 2:
        return None
    return path


class PredictiveSteering:
    """Steering from a prediction over a horizon, speed from Stanley.

    Subclasses implement ``_solve``: given the path, the commands already in
    flight, the speed and the steering input now, return the command to send.
    Without a metric path (another planner mode, or a stop) Stanley steers.
    """

    def __init__(
        self,
        config: PipelineConfig,
        camera: CameraModel | None = None,
        grid_source: Callable[[], object] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._cfg = config.predictive
        self._stanley = StanleyController(config, camera=camera)
        self._geometry = Geometry(config)
        self._actuator = SteeringActuator(config.control_latency_s, self._cfg.steer_slew_per_s)
        self._grid_source = grid_source
        self._clock = clock
        self._last_command = 0.0

    def reset(self) -> None:
        self._stanley.reset()
        self._actuator.reset()
        self._last_command = 0.0
        self._reset_solution()

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        base = self._stanley.compute(plan, state)
        now = self._clock()
        self._actuator.advance(now)
        steering = base.steering
        path = plan_path(plan)
        if path is not None:
            cfg = self._cfg
            committed = self._actuator.committed(now, cfg.dt_s, cfg.horizon_steps)
            speed = max(float(state.speed_mps), cfg.min_predict_speed_mps)
            steering = self._solve(path, committed, speed, self._actuator.input)
        else:
            self._reset_solution()
        step = self._cfg.max_command_step
        steering = float(np.clip(steering, self._last_command - step, self._last_command + step))
        steering = float(np.clip(steering, -1.0, 1.0))
        self._actuator.record(now, steering)
        self._last_command = steering
        # Stanley's speed law and its latency prediction read its own last
        # steering; give it the steering actually sent.
        self._stanley._prev_steering = steering
        debug = base.debug
        if debug is not None:
            debug.desired_steering = steering
            debug.final_steering = steering
        return ControlCommand(
            steering=steering,
            throttle=base.throttle,
            brake=base.brake,
            parkingbrake=base.parkingbrake,
            debug=debug,
        )

    def _solve(
        self, path: _SampledPath, committed: np.ndarray, speed: float, input0: float
    ) -> float:
        raise NotImplementedError

    def _reset_solution(self) -> None:
        raise NotImplementedError

    def _grid(self):
        if self._grid_source is None:
            return None
        return self._grid_source()
