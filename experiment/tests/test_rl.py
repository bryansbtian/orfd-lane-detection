"""The experimental policy must preserve controller limits and reproducible comparisons."""

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiment.config import ExperimentConfig, load_experiment_config
from experiment.pipeline import build_controller
from experiment.rl.config import RLConfig
from experiment.rl.controller import FEATURES, ResidualPolicy, RLController
from experiment.run import build_parser, main
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, VehicleState

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def training():
    return importlib.import_module("experiment.rl.train")


def path(config):
    camera = CameraModel(config.camera, config.preprocess_width, config.preprocess_height)
    forward = np.linspace(2.0, 14.0, 24)
    return PathPlan(centerline=camera.ground_to_image(forward, np.ones_like(forward))[::-1])


def test_zero_policy_reproduces_stanley_and_reset():
    config = ExperimentConfig()
    rl = RLController(
        config, policy=ResidualPolicy(np.zeros(len(FEATURES)), config.rl), clock=lambda: 0
    )
    baseline = StanleyController(config, clock=lambda: 0)
    plan = path(config)
    for _ in range(2):
        for speed in (0.0, 1.0, 5.0, 2.0):
            actual = rl.compute(plan, VehicleState(speed_mps=speed))
            expected = baseline.compute(plan, VehicleState(speed_mps=speed))
            assert actual == expected
        rl.reset()
        baseline.reset()


@pytest.mark.parametrize("sign", [-1, 1])
def test_residual_obeys_caps_slew_and_speed_limit(sign):
    config = ExperimentConfig()
    rl = RLController(config, policy=ResidualPolicy(sign * np.ones(len(FEATURES)) * 100, config.rl))
    plan = path(config)
    previous = 0.0
    for _ in range(20):
        command = rl.compute(plan, VehicleState(speed_mps=8.0))
        assert abs(rl.last_residual) <= config.rl.max_residual
        assert abs(command.debug.desired_steering) <= rl.steering_cap(8.0)
        assert abs(command.steering - previous) <= config.max_steering_delta
        assert command.throttle == 0.0
        assert command.brake > 0.0
        previous = command.steering


@pytest.mark.parametrize("mode", ["empty", "stop", "hold", "invalid"])
def test_fallback_paths_bypass_the_learned_correction(mode):
    config = ExperimentConfig()
    rl = RLController(
        config, policy=ResidualPolicy(np.ones(len(FEATURES)), config.rl), clock=lambda: 0
    )
    baseline = StanleyController(config, clock=lambda: 0)
    plan = path(config)
    if mode == "empty":
        plan.centerline = np.empty((0, 2))
    if mode == "stop":
        plan.speed_scale = 0.0
    if mode == "hold":
        plan.fallback_active = True
        plan.speed_scale = 0.4
    if mode == "invalid":
        plan.valid = False
    actual = rl.compute(plan, VehicleState(speed_mps=2.0))
    assert actual == baseline.compute(plan, VehicleState(speed_mps=2.0))
    if mode == "stop":
        assert actual.throttle == 0.0
        assert actual.brake == config.max_brake


def test_checkpoint_roundtrip_factory_and_normalization_validation(tmp_path):
    config = ExperimentConfig(control_mode="rl")
    checkpoint = tmp_path / "policy.npz"
    config.rl = replace(config.rl, policy_path=str(checkpoint))
    ResidualPolicy(np.arange(len(FEATURES)), config.rl).save(checkpoint)
    controller = build_controller(config)
    assert isinstance(controller, RLController)
    np.testing.assert_array_equal(controller.policy.weights, np.arange(len(FEATURES)))
    with pytest.raises(ValueError, match="normalization"):
        ResidualPolicy.load(checkpoint, replace(config.rl, max_residual=0.1))


@pytest.mark.parametrize("bad", [np.zeros(2), np.full(len(FEATURES), np.nan)])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError, match="finite weights"):
        ResidualPolicy(bad, RLConfig())


def test_missing_and_corrupt_checkpoints_are_rejected(tmp_path):
    checkpoint = tmp_path / "missing.npz"
    with pytest.raises(ValueError, match="Train with"):
        ResidualPolicy.load(checkpoint, RLConfig())
    checkpoint.write_bytes(b"broken archive")
    with pytest.raises(ValueError, match="Cannot load"):
        ResidualPolicy.load(checkpoint, RLConfig())
    with checkpoint.open("wb") as stream:
        np.save(stream, np.zeros(len(FEATURES)))
    with pytest.raises(ValueError, match="Cannot load"):
        ResidualPolicy.load(checkpoint, RLConfig())
    np.savez(checkpoint, version=100)
    with pytest.raises(ValueError, match="version"):
        ResidualPolicy.load(checkpoint, RLConfig())


def test_rl_cli_and_missing_policy_exit_before_pipeline_creation(monkeypatch, tmp_path):
    assert build_parser().parse_args(["--rl-policy", "policy.npz"]).rl_policy == "policy.npz"

    def unexpected(*args, **kwargs):
        pytest.fail("Missing policy must fail before creating the simulator or pipeline")

    monkeypatch.setattr("offroad_autonomy.main.BeamNGClient", unexpected)
    monkeypatch.setattr("experiment.run.ExperimentPipeline", unexpected)
    with pytest.raises(SystemExit, match="RL policy not found"):
        main(["--rl-policy", str(tmp_path / "missing")])


@pytest.mark.parametrize(
    "setting,value",
    [
        ("iterations", 0),
        ("seed", -1),
        ("directions", 1.5),
        ("max_residual", 2),
        ("heading_scale_rad", float("nan")),
        ("weight_error", True),
        ("policy_path", ""),
        ("radius_max_m", 1),
        ("unknown", 1),
    ],
)
def test_invalid_rl_configuration_is_rejected(tmp_path, setting, value):
    import yaml

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        yaml.safe_dump({"experiment": {"controller": "rl"}, "control": {"rl": {setting: value}}})
    )
    with pytest.raises(ValueError, match="control.rl"):
        load_experiment_config(config_path)


def test_ars_learns_from_reward_and_handles_flat_returns(training):
    settings = RLConfig(directions=12, top_directions=6)
    weights = np.zeros(len(FEATURES))

    def objective(candidate):
        return -float(np.square(candidate - 0.5).sum())

    updated = training.ars_update(weights, objective, settings, np.random.default_rng(42))
    assert objective(updated) > objective(weights)
    unchanged = training.ars_update(
        weights, lambda candidate: 1.0, settings, np.random.default_rng(42)
    )
    np.testing.assert_array_equal(unchanged, weights)
    with pytest.raises(ValueError, match="finite"):
        training.ars_update(weights, lambda candidate: np.nan, settings, np.random.default_rng(42))


def test_offline_rollouts_reproduce_baseline_with_zero_policy(training):
    config = load_experiment_config(ROOT / "configs/default.yaml")
    episode = {"seed": 5, "mirror": True}
    first = training.rollout(config, episode, None)
    second = training.rollout(config, episode, np.zeros(len(FEATURES)))
    assert first.completed == second.completed
    np.testing.assert_array_equal(
        [row["err"] for row in first.log], [row["err"] for row in second.log]
    )
    np.testing.assert_array_equal(
        [row["steer"] for row in first.log], [row["steer"] for row in second.log]
    )
    assert training.reward(first, config.rl) == training.reward(second, config.rl)
    incomplete = replace(first, completed=False)
    assert training.reward(incomplete, config.rl) < training.reward(first, config.rl)
