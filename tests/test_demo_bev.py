from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.types import BEVSnapshot, PathPlan, SteeringDebug, VehicleState
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.visualization.bev import PATH, render_bev


def test_demo_caps_at_6_mph_and_coasting_band_is_off_by_default():
    cfg = load_config("configs/grid.yaml")
    assert cfg.speed_limit_mph == 6
    assert cfg.target_speed_mph < 6
    # Measured at the cap, the band let descents run past 6 mph.
    assert cfg.comfort_brake_deadband_mps == 0.0


def test_coasting_band_coasts_small_routine_error_but_brakes_above_cap():
    cfg = replace(load_config("configs/grid.yaml"), comfort_brake_deadband_mps=0.15)
    controller = StanleyController(cfg)
    plan = PathPlan(centerline=np.empty((0, 2)))
    target = cfg.target_speed_mph * 0.44704
    controller._prev_target = target
    throttle, brake = controller._longitudinal_control(
        plan, VehicleState(speed_mps=target + 0.1), None, SteeringDebug()
    )
    assert throttle == brake == 0
    throttle, brake = controller._longitudinal_control(
        plan, VehicleState(speed_mps=6 * 0.44704 + 0.05), None, SteeringDebug()
    )
    assert throttle == 0 and brake > 0


def test_demo_preserves_gate_and_clearance_braking():
    cfg = load_config("configs/grid.yaml")
    controller = StanleyController(cfg)
    plan = PathPlan(centerline=np.empty((0, 2)), speed_scale=0)
    state = VehicleState(speed_mps=2)
    assert controller._longitudinal_control(plan, state, None, SteeringDebug()) == (
        0,
        cfg.max_brake,
    )
    # Clearance braking must ignore both the comfort ceiling and the coasting
    # band; a low ceiling and a band make that visible under the 6 mph cap.
    soft = replace(cfg, comfort_brake=0.1, comfort_brake_deadband_mps=0.15)
    controller = StanleyController(soft)
    plan.speed_scale = 1
    plan.min_clearance_m = 0
    state.speed_mps = 2.6
    debug = SteeringDebug()
    _, brake = controller._longitudinal_control(plan, state, None, debug)
    assert debug.speed_reason == "clearance"
    assert brake > soft.comfort_brake
    # The full error, with no coasting band taken off it.
    assert brake == pytest.approx(soft.speed_kp * (2.6 - debug.target_speed_mps))


def test_bev_snapshot_render_keeps_evidence_and_hides_path_under_stop():
    road = np.zeros((200, 200), dtype=bool)
    road[:, 85:115] = True
    snapshot = BEVSnapshot(
        road=road,
        blocked=~road,
        visible=road.copy(),
        cell_m=0.1,
        ahead_m=16,
        half_width_m=10,
        path=np.array([[2.0, 0.0], [12.0, 0.0]]),
    )
    before = snapshot.road.copy()
    active = render_bev(snapshot, 540, 654)
    stopped = render_bev(snapshot, 540, 654, show_path=False)
    assert active.shape == (654, 540, 3)
    assert np.any(np.all(active == PATH, axis=2))
    assert not np.any(np.all(stopped == PATH, axis=2))
    np.testing.assert_array_equal(snapshot.road, before)


def test_bev_handles_non_grid_run():
    assert render_bev(None, 540, 654).shape == (654, 540, 3)


def _path_pixels(heading_rad):
    road = np.zeros((200, 200), dtype=bool)
    road[:, 85:115] = True
    snapshot = BEVSnapshot(
        road=road,
        blocked=~road,
        visible=road.copy(),
        cell_m=0.1,
        ahead_m=16,
        half_width_m=10,
        path=np.array([[2.0, 0.0], [12.0, 0.0]]),
        heading_rad=heading_rad,
    )
    image = render_bev(snapshot, 540, 654)
    ys, xs = np.nonzero(np.all(image == PATH, axis=2))
    return xs.mean(), ys.mean(), image.shape


def test_bev_up_stays_the_starting_direction_as_the_vehicle_turns():
    x0, y0, (h, w, _) = _path_pixels(0.0)
    x_left, _, _ = _path_pixels(np.pi / 2)
    x_right, _, _ = _path_pixels(-np.pi / 2)
    # Facing the start direction, a straight-ahead path points up the screen.
    assert y0 < h / 2
    assert abs(x0 - w / 2) < 5
    # Turned left, the same path points left; turned right, right.
    assert x_left < w / 2 - 50
    assert x_right > w / 2 + 50
