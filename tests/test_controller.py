"""Metric controller: curvature feedforward, adaptive lookahead, speed planning.

Paths are built on the ground (forward/right metres from the camera) and
projected into the segmentation image with the same camera model the
controller uses, so the tests exercise the real pixel -> ground round trip.
"""

import math

import numpy as np
import pytest

from offroad_autonomy.control.stanley_controller import StanleyController, path_to_ground
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(model_weights="dummy.pt", **overrides)


def _camera(config: PipelineConfig) -> CameraModel:
    return CameraModel(
        config.segmentation_camera, config.preprocess_width, config.preprocess_height
    )


def _path(config, right_of_forward, forward=np.linspace(1.5, 14.0, 20)) -> PathPlan:
    """Image path for ground points ``right = f(forward)``, far -> near."""
    cam = _camera(config)
    forward = np.asarray(forward, dtype=np.float64)
    right = np.array([right_of_forward(f) for f in forward])
    origin = cam.position
    vehicle = np.stack([origin[0] - right, origin[1] - forward, np.zeros_like(forward)], axis=1)
    u, v, _ = cam.project(cam.from_vehicle(vehicle))
    pts = np.stack([u, v], axis=1)[::-1].astype(np.float32)
    return PathPlan(centerline=pts)


def _settle(controller, plan, speed, frames=40):
    command = None
    for _ in range(frames):
        command = controller.compute(plan, VehicleState(speed_mps=speed))
    return command


def test_ground_projection_round_trip():
    config = _config()
    plan = _path(config, lambda f: 0.1 * f)
    forward, right, keep = path_to_ground(_camera(config), plan.centerline)
    assert keep.all()
    np.testing.assert_allclose(right, 0.1 * forward, atol=1e-3)


def test_small_offset_is_tolerated():
    config = _config(cross_track_tolerance_m=0.3)
    command = _settle(StanleyController(config), _path(config, lambda f: 0.15), speed=4.0)
    assert command.debug.cross_track_m == pytest.approx(0.15, abs=0.01)
    # Inside the tolerance band: no cross-track correction at all.
    assert abs(command.steering) < 0.05


def test_offset_recovers_gradually_toward_the_path():
    config = _config()
    command = _settle(StanleyController(config), _path(config, lambda f: 1.0), speed=4.0)
    # Right, but gently: under ~11 deg of wheel for a full metre of offset.
    assert 0.0 < command.steering < 0.35


def test_curve_is_followed_in_the_right_direction():
    config = _config()
    radius = 25.0
    left_curve = _path(config, lambda f: -(radius - math.sqrt(radius**2 - f**2)))
    command = _settle(StanleyController(config), left_curve, speed=4.0)
    assert command.steering < -0.05
    # Heading error is read against the car's predicted arc, which is already
    # turning left here - the curve shows up in the feedforward instead.
    assert command.debug.feedforward_steering < 0.0


def test_steering_rate_is_limited_per_frame():
    config = _config(max_steering_delta=0.05, steering_ema_alpha=1.0)
    controller = StanleyController(config)
    plan = _path(config, lambda f: 3.0 + 0.5 * f)  # large demand
    previous = 0.0
    for _ in range(5):
        command = controller.compute(plan, VehicleState(speed_mps=2.0))
        assert abs(command.steering - previous) <= 0.05 + 1e-9
        previous = command.steering
    assert command.debug.desired_steering > command.steering


def test_lookahead_grows_with_speed_and_is_bounded():
    controller = StanleyController(_config())
    assert controller.lookahead_distance(0.0) == pytest.approx(3.0)
    assert controller.lookahead_distance(5.0) > controller.lookahead_distance(2.0)
    assert controller.lookahead_distance(50.0) == pytest.approx(8.0)


def test_lookahead_shrinks_in_tight_curves():
    controller = StanleyController(_config())
    straight = controller.lookahead_distance(4.0, curvature=0.0)
    gentle = controller.lookahead_distance(4.0, curvature=0.03)
    tight = controller.lookahead_distance(4.0, curvature=0.15)
    assert straight > gentle > tight >= 2.0


def _arc(radius: float, sign: float = 1.0):
    """right(f) for a circle through the camera, tangent to the vehicle."""
    return lambda f: sign * (radius - math.sqrt(max(radius**2 - f**2, 0.0)))


def test_feedforward_steers_for_the_arc_before_any_error():
    """On the path and aligned, the command is the arc's own steering."""
    config = _config(steering_ema_alpha=1.0, max_steering_delta=1.0)
    radius = 12.0
    command = StanleyController(config).compute(
        _path(config, _arc(radius), forward=np.linspace(0.5, 6.0, 20)), VehicleState(speed_mps=1.0)
    )
    expected = math.atan(config.wheelbase_m / radius) / math.radians(config.max_wheel_angle_deg)
    assert command.debug.curvature == pytest.approx(1.0 / radius, rel=0.15)
    assert command.debug.feedforward_steering == pytest.approx(expected, rel=0.15)
    assert command.steering > 0.0


def test_slows_before_a_bend_and_keeps_speed_on_straights():
    config = _config()
    straight = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 24))

    def bend_after_8m(f):
        if f < 8.0:
            return 0.0
        return (f - 8.0) ** 2 / (2 * 6.0)

    bend_ahead = _path(config, bend_after_8m, forward=np.linspace(1.0, 14.0, 24))
    state = VehicleState(speed_mps=4.0)
    cruise = StanleyController(config).compute(straight, state)
    braking = StanleyController(config).compute(bend_ahead, state)
    # Straight: accelerating toward cruise (ramped), never slowing.
    assert cruise.debug.speed_reason in ("cruise", "ramp up") and cruise.throttle > 0.0
    # Still on the straight, but already slowing for the bend.
    assert braking.debug.target_speed_mps < 4.0 and braking.brake > 0.0
    assert braking.debug.speed_reason in ("curve", "saturation")


def test_saturation_cuts_speed_and_throttle():
    config = _config()
    tight = _path(config, _arc(5.0), forward=np.linspace(0.5, 6.0, 20))
    command = StanleyController(config).compute(tight, VehicleState(speed_mps=1.0))
    assert command.debug.saturation > 0.8
    assert command.debug.target_speed_mps < 0.6 * StanleyController(config)._target_speed


def test_overspeed_gets_full_brake_not_comfort_brake():
    """A descent must not out-run the comfort-brake ceiling."""
    config = _config()
    straight = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 24))
    controller = StanleyController(config)
    controller._prev_target = 1.0  # e.g. just slowed for a bend
    command = controller.compute(straight, VehicleState(speed_mps=5.0))
    assert command.brake > config.comfort_brake


def test_no_full_lock_at_speed():
    config = _config(steering_ema_alpha=1.0, max_steering_delta=1.0)
    controller = StanleyController(config)
    tight = _path(config, _arc(4.0), forward=np.linspace(0.5, 5.0, 20))
    fast = controller.compute(tight, VehicleState(speed_mps=6.0))
    assert abs(fast.steering) <= config.steer_cap_at_high_speed + 1e-6
    slow = StanleyController(config).compute(tight, VehicleState(speed_mps=1.0))
    assert abs(slow.steering) > abs(fast.steering)


def test_target_speed_ramps_up_after_a_bend():
    config = _config(max_target_speed_increase_mps=0.1)
    controller = StanleyController(config)
    bend = _path(config, _arc(6.0), forward=np.linspace(0.5, 8.0, 20))
    straight = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 24))
    slow_target = controller.compute(bend, VehicleState(speed_mps=2.0)).debug.target_speed_mps
    after = controller.compute(straight, VehicleState(speed_mps=2.0)).debug
    assert after.target_speed_mps == pytest.approx(slow_target + 0.1, abs=1e-6)
    assert after.speed_reason == "ramp up"


@pytest.mark.parametrize("radius", [10.0, 7.0, 5.0])
def test_s_bend_is_completed_without_lock(radius):
    """Closed loop at 4 FPS with actuation delay (scripts/sbend_sim.py)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from sbend_sim import simulate

    from offroad_autonomy.utils.config import load_config

    # The shipped tuning, not dataclass defaults: this guards configs/default.yaml.
    shipped = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    result = simulate(shipped, radius=radius, fps=4.0)
    assert result.completed
    assert result.frames_at_lock == 0
    # Standing start 0.3 rad off the road, per-frame path jitter: stays within
    # the ~2.2 m a 1.9 m-wide Hopper has either side on a ~6 m track.
    assert result.max_lateral_error < 2.0


def test_authority_falls_with_speed_but_not_at_low_speed():
    controller = StanleyController(_config(steer_full_authority_speed_mps=3.0))
    assert controller.steering_authority(1.0) == 1.0
    assert controller.steering_authority(3.0) == 1.0
    assert controller.steering_authority(8.0) < controller.steering_authority(5.0) < 1.0


def test_short_path_brakes_before_its_end():
    """A path ending at a bush 3 m ahead must not be driven into at speed."""
    config = _config()
    long_path = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 20))
    short_path = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 3.0, 20))
    state = VehicleState(speed_mps=4.0)
    assert StanleyController(config).compute(long_path, state).throttle > 0.0
    command = StanleyController(config).compute(short_path, state)
    assert command.throttle == 0.0 and command.brake > 0.0
    assert command.debug.path_end_m == pytest.approx(3.0, abs=0.1)


def test_lookahead_point_is_reported_on_the_image():
    config = _config()
    command = StanleyController(config).compute(
        _path(config, lambda f: 0.0), VehicleState(speed_mps=0.0)
    )
    x, y = command.debug.lookahead_px
    assert x == pytest.approx(config.preprocess_width / 2.0, abs=1.0)
    forward, _, _ = path_to_ground(_camera(config), np.array([[x, y]]))
    assert forward[0] == pytest.approx(3.0, abs=0.1)


from offroad_autonomy.control.stanley_controller import measure_trail  # noqa: E402


def _trail(config, width=6.0, offset=0.0, roi_top=None):
    """Straight trail mask ``width`` m wide, centred ``offset`` m to the right."""
    import cv2

    cam = _camera(config)
    f = np.linspace(0.5, 18.0, 40)
    left = cam.ground_to_image(f, np.full_like(f, offset - width / 2))
    right = cam.ground_to_image(f, np.full_like(f, offset + width / 2))
    mask = np.zeros((cam.height, cam.width), np.uint8)
    cv2.fillPoly(mask, [np.round(np.vstack([left, right[::-1]])).astype(np.int32)], 1)
    if roi_top is not None:
        mask[:roi_top] = 0
    return mask.astype(bool)


def test_soft_deadzone_is_continuous_and_tolerant():
    controller = StanleyController(_config(cross_track_tolerance_m=0.3))
    values = [controller._deadzone(e) for e in np.linspace(-1.5, 1.5, 301)]
    assert np.all(np.abs(np.diff(values)) < 0.02)  # no jump anywhere
    assert abs(controller._deadzone(0.1)) < 0.005  # small offsets ~ignored
    assert controller._deadzone(1.2) == pytest.approx(1.2 - 0.3, abs=0.01)


def test_trail_edges_are_measured_in_metres():
    config = _config()
    trail = measure_trail(_camera(config), _trail(config, width=6.0), arc_curvature=0.0)
    to_left, to_right = trail.distances()
    assert to_left == pytest.approx(3.0, abs=0.15)
    assert to_right == pytest.approx(3.0, abs=0.15)
    shifted = measure_trail(_camera(config), _trail(config, width=6.0, offset=1.5), 0.0)
    left2, right2 = shifted.distances()
    assert left2 == pytest.approx(1.5, abs=0.15) and right2 == pytest.approx(4.5, abs=0.2)


def _cruising(config):
    """A controller already at cruise, so the speed ramp does not mask limits."""
    controller = StanleyController(config)
    controller._prev_target = 10.0
    return controller


def test_edge_risk_slows_the_car():
    config = _config(min_turn_speed_mph=3.0)  # the shipped crawl floor
    straight = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 24))
    state = VehicleState(speed_mps=3.0)
    centred = replace_mask(straight, _trail(config, width=6.0))
    near_edge = replace_mask(straight, _trail(config, width=6.0, offset=2.0))  # left edge 1 m away
    a = _cruising(config).compute(centred, state).debug
    b = _cruising(config).compute(near_edge, state).debug
    assert a.edge_risk == 0.0
    assert b.edge_risk > 0.5
    assert b.target_speed_mps < a.target_speed_mps


def replace_mask(plan, mask):
    from dataclasses import replace as dc_replace

    return dc_replace(plan, planner_mask=mask)


def test_path_near_an_edge_is_pulled_toward_the_centre():
    config = _config()
    # Path hugging the left edge of a 5 m trail centred on the car.
    hugging = replace_mask(
        _path(config, lambda f: -1.5, forward=np.linspace(1.0, 10.0, 24)), _trail(config, width=5.0)
    )
    debug = StanleyController(config).compute(hugging, VehicleState(speed_mps=2.0)).debug
    assert debug.centering_shift_m > 0.2  # moved right, toward the centre


def test_growing_cross_track_error_slows_the_car():
    import time

    config = _config(min_turn_speed_mph=3.0)  # the shipped crawl floor
    controller = StanleyController(config)
    state = VehicleState(speed_mps=3.0)
    reasons = []
    for offset in (0.0, 0.4, 0.8, 1.2):
        plan = _path(config, lambda f, o=offset: o, forward=np.linspace(1.0, 14.0, 24))
        controller._prev_target = 10.0
        debug = controller.compute(plan, state).debug
        reasons.append(debug.speed_reason)
        time.sleep(0.12)
    assert debug.cross_track_rate_mps > 0.0
    assert "drift" in reasons


def test_high_lateral_acceleration_limits_speed():
    config = _config()
    controller = StanleyController(config)
    controller._prev_steering = 0.6  # already turning hard
    controller._prev_target = 5.0
    straight = _path(config, lambda f: 0.0, forward=np.linspace(1.0, 14.0, 24))
    debug = controller.compute(straight, VehicleState(speed_mps=4.5)).debug
    assert debug.lateral_accel_mps2 > config.max_measured_lateral_accel_mps2
    assert debug.target_speed_mps < 4.5


def test_rejoin_distance_shortens_only_in_recovery():
    config = _config()
    small = (
        StanleyController(config)
        .compute(
            _path(config, lambda f: 0.4, forward=np.linspace(1.0, 14.0, 24)),
            VehicleState(speed_mps=3.0),
        )
        .debug
    )
    large = (
        StanleyController(config)
        .compute(
            _path(config, lambda f: 2.0, forward=np.linspace(1.0, 14.0, 24)),
            VehicleState(speed_mps=3.0),
        )
        .debug
    )
    normal = StanleyController(config).lookahead_distance(3.0)
    assert small.rejoin_m == pytest.approx(normal, rel=1e-6)
    assert large.rejoin_m < normal
