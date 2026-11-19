"""Measured adapters reject bad sweeps and require explicit experimental activation."""

import copy
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from experiment.config import ExperimentConfig
from experiment.imitation.calibration import summarize_sweep
from experiment.imitation.config import ImitationConfig
from experiment.imitation.controller import ImitationController, SteeringCalibration


@pytest.fixture
def sweep():
    settings = replace(ImitationConfig(), calibration_points=5, calibration_samples=2)
    rows = []
    for command in np.linspace(-1, 1, 5).tolist():
        for _ in range(4):
            wheels = []
            for name, position in (
                ("FL", [0.8, 2.4, 0]),
                ("FR", [-0.8, 2.4, 0]),
                ("RL", [0.8, 0, 0]),
                ("RR", [-0.8, 0, 0]),
            ):
                angle = 0.0
                if name.startswith("F"):
                    angle = -command * 0.5
                wheels.append({"name": name, "position": position, "angle_left_rad": angle})
            rows.append({"command": command, "speed_mps": 0.0, "wheels": wheels})
    return rows, settings


def experimental_data(config):
    return {
        "version": 2,
        "simulation_only": True,
        "measured": True,
        "validation_mode": "simulation_experiment",
        "measurement_source": "Synthetic Calibration Fixture",
        "simulator_vehicle": config.beamng_vehicle,
        "simulator_camera": json.loads(json.dumps(asdict(config.camera))),
        "recorded_rwa_radians_positive_left_verified": False,
        "source_camera_extrinsics_verified": False,
        "recorded_units_status": "dynamics_supported_radians_hypothesis",
        "recorded_direction_status": "positive_yaw_hypothesis_not_publisher_verified",
        "real_wheelbase_m": config.imitation.real_wheelbase_m,
        "simulator_wheelbase_m": config.wheelbase_m,
        "wheel_angle_left_rad": [-0.5, 0, 0.5],
        "beamng_commands": [1, 0, -1],
    }


def test_sweep_measures_wheelbase_and_positive_left_inverse(sweep):
    rows, settings = sweep
    result = summarize_sweep(rows, settings)
    assert result["simulator_wheelbase_m"] == pytest.approx(2.4)
    assert result["wheel_angle_left_rad"] == pytest.approx([-0.5, -0.25, 0, 0.25, 0.5])
    assert result["beamng_commands"] == [1, 0.5, 0, -0.5, -1]


@pytest.mark.parametrize(
    "failure", ["moving", "nonfinite", "missing_wheel", "direction", "repeats"]
)
def test_bad_measurements_do_not_produce_an_adapter(sweep, failure):
    rows, settings = sweep
    rows = copy.deepcopy(rows)
    if failure == "moving":
        rows[0]["speed_mps"] = 1
    if failure == "nonfinite":
        rows[0]["wheels"][0]["angle_left_rad"] = float("nan")
    if failure == "missing_wheel":
        rows[0]["wheels"].pop()
    if failure == "direction":
        for row in rows:
            for wheel in row["wheels"]:
                wheel["angle_left_rad"] *= -1
    if failure == "repeats":
        rows.pop()
    with pytest.raises(ValueError):
        summarize_sweep(rows, settings)


def test_experiment_requires_opt_in_and_preserves_unknown_source_geometry():
    config = ExperimentConfig()
    data = experimental_data(config)
    with pytest.raises(ValueError, match="imitation-experiment"):
        SteeringCalibration(data)
    adapter = SteeringCalibration(data, allow_simulation_experiment=True)
    adapter.validate_simulator(config)
    assert adapter.command(0.25) == pytest.approx(-0.5)
    data["source_camera_extrinsics_verified"] = True
    with pytest.raises(ValueError, match="unverified recording"):
        SteeringCalibration(data, allow_simulation_experiment=True)


@pytest.mark.parametrize("change", ["vehicle", "camera", "wheelbase"])
def test_changed_simulator_or_recording_geometry_requires_recalibration(change):
    config = ExperimentConfig()
    adapter = SteeringCalibration(experimental_data(config), allow_simulation_experiment=True)
    if change == "vehicle":
        config.beamng_vehicle = "different_vehicle"
    if change == "camera":
        config.camera = replace(config.camera, pos=(0, 0, 3))
    if change == "wheelbase":
        config.imitation = replace(config.imitation, real_wheelbase_m=3)
    with pytest.raises(ValueError, match="differs"):
        adapter.validate_simulator(config)


def test_experimental_controller_caps_speed_without_loading_a_model():
    config = ExperimentConfig()
    adapter = SteeringCalibration(experimental_data(config), allow_simulation_experiment=True)

    class Predictor:
        def __init__(self):
            self.metadata = {"camera_calibration": {"width": 2880, "k": [1714.20536]}}

    controller = ImitationController(config, calibration=adapter, predictor=Predictor())
    assert controller._target_speed <= config.imitation.experiment_max_speed_mps
    assert controller._speed_limit <= config.imitation.experiment_max_speed_mps


@pytest.mark.parametrize("args", [["--controller", "stanley"], ["--rl-policy", "p.npz"]])
def test_experiment_flag_conflicts_with_other_controllers(args):
    from experiment.run import main

    with pytest.raises(SystemExit, match="conflict"):
        main(["--imitation-experiment", *args])


def test_failed_connection_still_disconnects_when_measurement_storage_fails(monkeypatch, tmp_path):
    from pathlib import Path
    from unittest.mock import Mock

    from experiment.imitation.calibration import measure_calibration

    client = Mock()
    client.connect.side_effect = RuntimeError("Synthetic Connection Failure")
    monkeypatch.setattr("experiment.imitation.calibration.BeamNGClient", lambda config: client)

    def unavailable_storage(*args, **kwargs):
        raise OSError("Synthetic Storage Failure")

    monkeypatch.setattr(Path, "write_text", unavailable_storage)
    with pytest.raises(OSError, match="Storage Failure"):
        measure_calibration(ExperimentConfig(), tmp_path)
    client.disconnect.assert_called_once()
