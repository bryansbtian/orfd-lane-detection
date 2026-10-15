"""Predictive steering controllers (control.mode: mppi / mpc)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from offroad_autonomy.control.factory import build_controller
from offroad_autonomy.control.mpc_controller import MpcController
from offroad_autonomy.control.mppi_controller import MppiController
from offroad_autonomy.control.predictive import SteeringActuator
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState
from offroad_autonomy.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = PipelineConfig()
CAMERA = CameraModel(CONFIG.camera, CONFIG.preprocess_width, CONFIG.preprocess_height)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _plan(right_of_forward) -> PathPlan:
    """A planner-made metric path, as grid mode hands it over."""
    f = np.linspace(1.0, 14.0, 53)
    r = right_of_forward(f)
    curvature = np.gradient(np.gradient(r, f), f)
    return PathPlan(
        centerline=CAMERA.ground_to_image(f, r)[::-1],
        ground_path=np.column_stack((f, r)),
        ground_curvature=curvature,
    )


@pytest.mark.parametrize(
    "mode, kind", [("stanley", StanleyController), ("mppi", MppiController), ("mpc", MpcController)]
)
def test_control_mode_selects_the_controller(mode, kind):
    assert isinstance(build_controller(replace(CONFIG, control_mode=mode), camera=CAMERA), kind)


def test_the_new_configs_select_their_controllers_and_keep_the_planner():
    base = load_config(REPO / "configs/grid.yaml")
    for name, mode in (("grid_mpc.yaml", "mpc"), ("grid_mppi.yaml", "mppi")):
        cfg = load_config(REPO / "configs" / name)
        assert cfg.control_mode == mode
        assert cfg.planner_mode == base.planner_mode
        assert cfg.grid == base.grid
    assert base.control_mode == "stanley"


def test_a_bad_control_mode_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("control:\n  mode: pid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control.mode"):
        load_config(path)


def test_actuator_applies_commands_after_the_latency_and_slews():
    actuator = SteeringActuator(latency_s=0.3, slew_per_s=1.0)
    actuator.advance(0.0)
    actuator.record(0.0, 1.0)
    actuator.advance(0.29)
    assert actuator.input == pytest.approx(0.0)
    actuator.advance(0.8)
    # Effective from 0.3 s, slewing at 1/s: 0.5 by 0.8 s.
    assert actuator.input == pytest.approx(0.5, abs=0.02)
    # The first steps of a prediction from 0.1 s replay the command in flight.
    committed = actuator.committed(0.1, dt=0.1, steps=30)
    assert len(committed) == 3
    assert committed[0] == 0.0 and committed[-1] == 1.0


@pytest.mark.parametrize("kind", [MppiController, MpcController])
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_predictive_controllers_steer_toward_an_offset_path(kind, side):
    clock = Clock()
    controller = kind(CONFIG, camera=CAMERA, clock=clock)
    plan = _plan(lambda f: side * 1.5 + 0.0 * f)
    steering = 0.0
    for _ in range(10):
        steering = controller.compute(plan, VehicleState(speed_mps=2.0)).steering
        clock.t += 1 / 15
    assert np.sign(steering) == side


@pytest.mark.parametrize("kind", [MppiController, MpcController])
def test_predictive_controllers_hold_straight_on_a_straight_path(kind):
    clock = Clock()
    controller = kind(CONFIG, camera=CAMERA, clock=clock)
    plan = _plan(lambda f: 0.0 * f)
    for _ in range(10):
        command = controller.compute(plan, VehicleState(speed_mps=2.0))
        clock.t += 1 / 15
    assert abs(command.steering) < 0.05


@pytest.mark.parametrize("kind", [MppiController, MpcController])
def test_speed_comes_from_the_stanley_law_unchanged(kind):
    plan = _plan(lambda f: 0.02 * f * f)
    stanley = StanleyController(CONFIG, camera=CAMERA).compute(plan, VehicleState(speed_mps=1.0))
    predictive = kind(CONFIG, camera=CAMERA, clock=Clock()).compute(
        plan, VehicleState(speed_mps=1.0)
    )
    assert predictive.throttle == pytest.approx(stanley.throttle)
    assert predictive.brake == pytest.approx(stanley.brake)


@pytest.mark.parametrize("kind", [MppiController, MpcController])
def test_without_a_metric_path_stanley_steers(kind):
    plan = _plan(lambda f: 0.5 + 0.0 * f)
    plan.ground_path = None
    plan.ground_curvature = None
    stanley = StanleyController(CONFIG, camera=CAMERA).compute(plan, VehicleState(speed_mps=2.0))
    predictive = kind(CONFIG, camera=CAMERA, clock=Clock()).compute(
        plan, VehicleState(speed_mps=2.0)
    )
    assert predictive.steering == pytest.approx(stanley.steering)


class _WallGrid:
    """A grid whose right half (right > 0.5 m) is non-road."""

    rows, cols, cell_m = 260, 200, 0.1

    def clearance_m(self):
        right = (np.arange(self.cols) + 0.5) * self.cell_m - 10.0
        return np.tile(np.clip(0.5 - right, 0.0, None), (self.rows, 1))

    def to_cell(self, forward_m, right_m):
        row = np.floor((26.0 - np.asarray(forward_m)) / self.cell_m).astype(int)
        col = np.floor((np.asarray(right_m) + 10.0) / self.cell_m).astype(int)
        return row, col


def test_mppi_leaves_a_path_that_runs_into_non_road():
    """The planned path drifts right into non-road; the grid cost makes MPPI
    steer less right than tracking alone would."""
    plan = _plan(lambda f: 0.12 * f)
    tracking = MppiController(
        replace(CONFIG, predictive=replace(CONFIG.predictive, weight_offroad=0.0)),
        camera=CAMERA,
        clock=Clock(),
    )
    guarded = MppiController(CONFIG, camera=CAMERA, grid_source=_WallGrid, clock=Clock())
    a = tracking.compute(plan, VehicleState(speed_mps=2.0)).steering
    b = guarded.compute(plan, VehicleState(speed_mps=2.0)).steering
    assert b < a
