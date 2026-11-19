"""Experiment selection, config loading and the hooks into the default pipeline."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from experiment.advanced.planner import AdvancedPlanner
from experiment.config import ExperimentConfig, load_experiment_config
from experiment.grid.planner import GridPlanner
from experiment.pipeline import ExperimentPipeline, build_planner
from experiment.run import run_label
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.types import (
    CameraFrame,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    StabilizedResult,
    VehicleState,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
H, W = 465, 720


@pytest.mark.parametrize(
    ("name", "planner", "controller"),
    [
        ("grid.yaml", "grid", "stanley"),
        ("grid_mpc.yaml", "grid", "mpc"),
        ("grid_mppi.yaml", "grid", "mppi"),
        ("advanced.yaml", "advanced", "stanley"),
        ("rl.yaml", "baseline", "rl"),
        ("imitation.yaml", "baseline", "imitation"),
    ],
)
def test_shipped_experiment_configs_select_their_planner_and_controller(name, planner, controller):
    config = load_experiment_config(CONFIGS / name)
    assert config.planner_mode == planner
    assert config.control_mode == controller


@pytest.mark.parametrize(
    ("block", "match"),
    [("    arc_size: 3", "arc_size"), ("    arc_count: 40", "odd"), ("    cell_m: 0", "cell_m")],
)
def test_invalid_grid_settings_are_refused(tmp_path, block, match):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("planning:\n  grid:\n" + block + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_experiment_config(config_path)


@pytest.mark.parametrize(
    ("mode", "kind"),
    [("baseline", CenterlinePlanner), ("advanced", AdvancedPlanner), ("grid", GridPlanner)],
)
def test_planner_mode_selects_the_planner(mode, kind):
    assert type(build_planner(ExperimentConfig(planner_mode=mode))) is kind


def test_advanced_mode_is_gated_too():
    road = np.zeros((H, W), dtype=bool)
    road[230:, W // 2 - 150 : W // 2 + 150] = True
    raw = PerceptionResult(mask=road, confidences=[0.05])
    stabilized = StabilizedResult(mask=road, raw_result=raw)
    plan = AdvancedPlanner(ExperimentConfig(model_weights="dummy.pt")).plan(stabilized)
    assert plan.fallback_active
    assert not plan.kalman_active


def test_imitation_consumes_the_existing_perception_snapshot():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    packet = FramePacket(raw=frame, preprocessed=frame, timestamp=0.0, height=H, width=W)
    mask = np.zeros((H, W), dtype=bool)
    plan = PathPlan(centerline=np.array([[320.0, 300.0], [320.0, 100.0]]))
    command = ControlCommand(steering=0.0, throttle=0.3, brake=0.0)
    config = ExperimentConfig(
        model_weights="dummy.pt", preprocess_width=W, preprocess_height=H, control_mode="imitation"
    )
    with ExitStack() as stack:
        for name in ("ImagePreprocessor", "RoadSegmenter", "TemporalStabilizer"):
            stack.enter_context(patch(f"offroad_autonomy.pipeline.{name}"))
        planner = stack.enter_context(patch("experiment.pipeline.build_planner"))
        controller = stack.enter_context(patch("experiment.pipeline.build_controller"))
        pipeline = ExperimentPipeline(config)
        pipeline.preprocessor.process.return_value = packet
        pipeline.segmenter.predict.return_value = PerceptionResult(mask=mask, confidences=[0.9])
        pipeline.stabilizer.stabilize.return_value = StabilizedResult(mask=mask)
        planner.return_value.plan.return_value = plan
        controller.return_value.compute_observation.return_value = command
        state = VehicleState(speed_mps=2)
        capture = CameraFrame(image=np.zeros((620, 960, 3), dtype=np.uint8), timestamp=0.0)
        result = pipeline.step_result(capture, state)

    args = controller.return_value.compute_observation.call_args.args
    assert args[0] is plan
    assert args[1] is state
    assert args[2] is packet.raw
    assert result.command is command
    controller.return_value.compute.assert_not_called()


def test_presentation_label_names_the_experiment():
    assert run_label(ExperimentConfig(planner_mode="grid", control_mode="mpc")) == (
        "Experiment | grid + mpc"
    )
    assert run_label(ExperimentConfig(control_mode="imitation")) == "Imitation Learning"
