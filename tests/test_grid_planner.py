"""Bird's-eye grid planner: projection, memory, motion and arc choice."""

import math
from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.control.stanley_controller import StanleyController, path_to_ground
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.planning.arc_planner import ArcPlanner
from offroad_autonomy.planning.bev_grid import GroundPose, TraversabilityGrid, ground_pose
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult, VehicleState

CONFIG = replace(PipelineConfig(), planner_mode="grid")
VIEW = PerceptionView(CONFIG)
CAMERA = VIEW.camera
VALID = VIEW.valid_roi
ROI_TOP = int(round(CAMERA.height * (1.0 - CONFIG.planner_roi_height)))


def _road_mask(is_road) -> np.ndarray:
    """What a perfect segmenter would return for a road defined on the ground."""
    u, v = CAMERA.pixel_grid()
    forward, right, keep = CAMERA.image_to_ground(np.stack([u.ravel(), v.ravel()], axis=1))
    road = keep & is_road(forward, right)
    return road.reshape(CAMERA.height, CAMERA.width)


def _straight(half_width_m: float):
    return lambda f, r: np.abs(r) < half_width_m


def _bend(curvature: float, half_width_m: float = 2.0):
    return lambda f, r: np.abs(r - 0.5 * curvature * f * f) < half_width_m


def _grid(config: GridPlannerConfig | None = None) -> TraversabilityGrid:
    return TraversabilityGrid(config or CONFIG.grid, CAMERA, VALID, ROI_TOP)


def _pose(x=0.0, y=0.0, heading_rad=0.0) -> GroundPose:
    return GroundPose(x=x, y=y, forward_x=math.cos(heading_rad), forward_y=math.sin(heading_rad))


def _cells(grid, forward_m, right_m):
    row, col = grid.to_cell(np.asarray(forward_m), np.asarray(right_m))
    return grid.logodds[row, col]


def _arcs() -> ArcPlanner:
    return ArcPlanner(
        CONFIG.grid,
        wheelbase_m=CONFIG.wheelbase_m,
        max_wheel_angle_deg=CONFIG.max_wheel_angle_deg,
        vehicle_half_width_m=CONFIG.vehicle_half_width_m,
        rear_axle_behind_camera_m=CONFIG.camera_ahead_of_rear_axle_m,
    )


def test_the_mask_lands_on_the_ground_where_the_road_is():
    grid = _grid()
    grid.update(_road_mask(_straight(2.0)), _pose(), dt_s=0.0)

    assert (_cells(grid, [6.0, 10.0], [0.0, 0.0]) > 0).all()
    assert (_cells(grid, [6.0, 10.0], [4.0, -4.0]) < 0).all()


def test_ground_the_camera_cannot_see_stays_unknown():
    """Under the hood and behind the camera there is no evidence either way,
    so nothing there may be marked as off-road."""
    grid = _grid()
    grid.update(_road_mask(_straight(2.0)), _pose(), dt_s=0.0)

    assert _cells(grid, [1.5, -1.0], [0.0, 0.0]).tolist() == [0.0, 0.0]


def test_memory_carries_road_into_the_ground_the_hood_now_hides():
    grid = _grid()
    grid.update(_road_mask(_straight(2.0)), _pose(), dt_s=0.0)

    # 3 m forward with no new frame: the road seen at 4.5 m is now at 1.5 m.
    grid.update(None, _pose(x=3.0), dt_s=0.0)

    assert _cells(grid, 1.5, 0.0) > CONFIG.grid.road_threshold
    assert _cells(grid, 1.5, 4.0) < CONFIG.grid.blocked_threshold


def test_moving_left_shifts_the_road_to_the_right():
    """Catches a sign error in the motion transform, which would steer the
    vehicle toward a mirrored memory of the road."""
    grid = _grid()
    grid.update(_road_mask(lambda f, r: np.abs(r) < 1.0), _pose(), dt_s=0.0)

    # Facing +x, left is +y.
    grid.update(None, _pose(y=1.5), dt_s=0.0)

    assert _cells(grid, 8.0, 1.5) > 0
    assert _cells(grid, 8.0, -1.5) < 0


def test_turning_left_swings_the_road_to_the_right():
    grid = _grid()
    grid.update(_road_mask(lambda f, r: np.abs(r) < 1.0), _pose(), dt_s=0.0)

    grid.update(None, _pose(heading_rad=math.radians(20.0)), dt_s=0.0)

    expected_right = 8.0 * math.sin(math.radians(20.0))
    assert _cells(grid, 8.0 * math.cos(math.radians(20.0)), expected_right) > 0
    assert _cells(grid, 8.0, 0.0) < 0


def test_without_a_pose_the_grid_forgets_instead_of_smearing():
    grid = _grid()
    grid.update(_road_mask(_straight(2.0)), _pose(), dt_s=0.0)

    grid.update(None, None, dt_s=0.0)

    assert not grid.logodds.any()


def test_evidence_fades_with_the_memory_time_constant():
    grid = _grid()
    grid.update(_road_mask(_straight(2.0)), _pose(), dt_s=0.0)
    before = float(_cells(grid, 8.0, 0.0))

    grid.update(None, _pose(), dt_s=CONFIG.grid.memory_s)

    assert float(_cells(grid, 8.0, 0.0)) == pytest.approx(before / math.e, rel=1e-4)


def test_ground_pose_uses_the_simulator_direction_vector():
    state = VehicleState(position=(10.0, 5.0, 0.0), direction=(0.0, 1.0, 0.0))

    pose = ground_pose(state, CAMERA)

    # The camera sits 0.30 m ahead of the vehicle origin, along +y here.
    assert pose.x == pytest.approx(10.0)
    assert pose.y == pytest.approx(5.0 + 0.30)
    assert ground_pose(VehicleState(), CAMERA) is None


def _choose(is_road, arcs=None):
    grid = _grid()
    grid.update(_road_mask(is_road), _pose(), dt_s=0.0)
    return (arcs or _arcs()).choose(grid)


def test_a_straight_road_gets_a_near_straight_arc():
    arcs = _arcs()
    choice = _choose(_straight(2.0), arcs)

    step = arcs.curvatures[1] - arcs.curvatures[0]
    assert abs(choice.curvature) <= step + 1e-9
    assert choice.length_m >= 8.0


@pytest.mark.parametrize("curvature", [0.04, -0.04])
def test_the_arc_bends_with_the_road(curvature):
    choice = _choose(_bend(curvature))

    assert math.copysign(1.0, choice.curvature) == math.copysign(1.0, curvature)


def test_a_road_wider_than_the_view_is_still_driven_straight():
    """The failure that stopped the row planner: a road running off both sides
    of the image has no visible edges, which must not bias the path."""
    arcs = _arcs()
    choice = _choose(_straight(12.0), arcs)

    step = arcs.curvatures[1] - arcs.curvatures[0]
    assert choice is not None
    assert abs(choice.curvature) <= 2 * step + 1e-9


def test_the_path_ends_where_the_road_does():
    choice = _choose(lambda f, r: (np.abs(r) < 2.0) & (f < 7.0))

    assert choice is not None
    assert choice.forward_m.max() < 7.0


def test_no_road_means_no_arc():
    assert _choose(lambda f, r: np.zeros_like(f, dtype=bool)) is None


def test_a_road_narrower_than_the_vehicle_is_not_driven():
    assert _choose(_straight(0.5)) is None


def _stabilized(mask):
    return StabilizedResult(
        mask=mask,
        valid_roi=VALID,
        raw_result=PerceptionResult(mask=mask, confidences=[0.6]),
    )


def test_grid_mode_hands_the_controller_a_usable_path():
    planner = CenterlinePlanner(CONFIG, camera=CAMERA)
    state = VehicleState(direction=(1.0, 0.0, 0.0), speed_mps=3.0, velocity=(3.0, 0.0, 0.0))

    plan = planner.plan(_stabilized(_road_mask(_straight(2.0))), vehicle_state=state)

    assert not plan.fallback_active
    forward, right, keep = path_to_ground(CAMERA, plan.centerline)
    assert forward[keep].min() < 2.0
    assert np.abs(right[keep]).max() < 1.0
    command = StanleyController(CONFIG, camera=CAMERA).compute(plan, state)
    assert command.throttle > 0.0


def test_grid_mode_stops_when_no_arc_fits_the_vehicle():
    """A 2.0 m trail passes the gate but is narrower than the 2.1 m the
    vehicle needs, so the planner, not the gate, refuses it."""
    planner = CenterlinePlanner(CONFIG, camera=CAMERA)
    narrow = _stabilized(_road_mask(_straight(1.0)))
    state = VehicleState(direction=(1.0, 0.0, 0.0))

    plan = planner.plan(narrow, vehicle_state=state)

    assert plan.speed_scale == 0.0
    assert "no drivable arc" in plan.fallback_reason


def test_memory_keeps_the_path_through_one_bad_frame():
    planner = CenterlinePlanner(CONFIG, camera=CAMERA)
    state = VehicleState(direction=(1.0, 0.0, 0.0))
    planner.plan(_stabilized(_road_mask(_straight(2.0))), vehicle_state=state)

    plan = planner.plan(_stabilized(_road_mask(_straight(1.0))), vehicle_state=state)

    assert not plan.fallback_active


def test_a_frame_the_gate_rejects_adds_no_evidence():
    planner = CenterlinePlanner(CONFIG, camera=CAMERA)
    state = VehicleState(direction=(1.0, 0.0, 0.0))
    rejected = StabilizedResult(
        mask=_road_mask(_straight(2.0)),
        valid_roi=VALID,
        raw_result=PerceptionResult(mask=None, confidences=[0.01]),
    )

    planner.plan(rejected, vehicle_state=state)

    assert not planner.grid.logodds.any()


def test_simulator_pose_alone_cannot_create_road_evidence():
    grid, arcs = _grid(), _arcs()
    for step in range(10):
        pose = _pose(x=step, y=0.2 * step, heading_rad=0.01 * step)
        grid.update(None, pose, dt_s=0.1)
        assert not grid.road().any()
        assert not grid.logodds.any()
        assert arcs.choose(grid, pose) is None


def test_absolute_world_location_cannot_supply_hidden_map_labels():
    first, translated = _grid(), _grid()
    mask = _road_mask(_straight(2.0))
    for step in range(4):
        first.update(mask, _pose(x=step * 0.5), dt_s=0.1)
        translated.update(mask, _pose(x=1000 + step * 0.5, y=-2500), dt_s=0.1)
        np.testing.assert_allclose(first.logodds, translated.logodds, atol=1e-6)


def test_unknown_clearance_is_not_positive_road_evidence():
    grid = _grid()
    # Collision clearance deliberately permits unknown ground under the hood,
    # but this must not invent an observed road or a supported trajectory.
    assert grid.clearance_m().max() > CONFIG.grid.lookahead_m
    assert not grid.road().any()
    assert _arcs().choose(grid, _pose()) is None


def test_model_false_positive_can_persist_but_corrected_evidence_clears_it():
    grid = _grid()
    wrong_mask = _road_mask(_straight(6.0))
    correct_mask = _road_mask(_straight(2.0))
    for _ in range(10):
        grid.update(wrong_mask, _pose(), dt_s=0.0)
    assert _cells(grid, 8.0, 4.0) > CONFIG.grid.road_threshold
    grid.update(correct_mask, _pose(), dt_s=0.0)
    # The BEV has no oracle that fixes the network's mistaken labels. Its
    # accumulated evidence can outlive a single corrected observation.
    assert _cells(grid, 8.0, 4.0) > CONFIG.grid.road_threshold
    for _ in range(10):
        grid.update(correct_mask, _pose(), dt_s=0.0)
    assert _cells(grid, 8.0, 4.0) < CONFIG.grid.blocked_threshold
