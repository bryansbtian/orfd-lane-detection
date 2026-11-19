"""Simulation-only learned steering, with an explicit measured actuator adapter."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import ControlCommand, PathPlan, PipelineConfig, VehicleState

logger = logging.getLogger("offroad_autonomy.imitation")


class SteeringCalibration:
    """An auditable physical wheel-angle to BeamNG-command lookup table."""

    def __init__(self, data: dict, *, allow_simulation_experiment: bool = False):
        version = data.get("version")
        if version not in {1, 2} or data.get("simulation_only") is not True:
            raise ValueError(
                "Imitation calibration must declare version 1 or 2 and simulation_only=true"
            )
        self.experimental = version == 2
        if self.experimental:
            if not allow_simulation_experiment:
                raise ValueError("Version 2 calibration requires --imitation-experiment")
            if (
                data.get("validation_mode") != "simulation_experiment"
                or data.get("source_camera_extrinsics_verified") is not False
                or data.get("recorded_rwa_radians_positive_left_verified") is not False
                or data.get("recorded_units_status") != "dynamics_supported_radians_hypothesis"
                or data.get("recorded_direction_status")
                != "positive_yaw_hypothesis_not_publisher_verified"
                or not isinstance(data.get("simulator_camera"), dict)
                or not data.get("simulator_vehicle")
            ):
                raise ValueError(
                    "Experimental calibration must retain unverified recording metadata"
                )
        elif data.get("camera_mount_verified") is not True:
            raise ValueError("Measured BeamNG steering and verified camera mounting are required")
        if data.get("measured") is not True:
            raise ValueError("Measured BeamNG steering is required")
        if (
            not self.experimental
            and data.get("recorded_rwa_radians_positive_left_verified") is not True
        ):
            raise ValueError(
                "Recorded steering units and positive-left direction must be verified before simulation actuation"
            )
        if not str(data.get("measurement_source", "")).strip():
            raise ValueError("Calibration must document its measurement source")
        try:
            self.real_wheelbase = float(data.get("real_wheelbase_m", 0))
            self.simulator_wheelbase = float(data.get("simulator_wheelbase_m", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Calibration wheelbases must be numeric metres") from exc
        if not all(
            math.isfinite(value) and value > 0
            for value in (self.real_wheelbase, self.simulator_wheelbase)
        ):
            raise ValueError("Calibration needs verified real and simulator wheelbases in metres")
        self.angles = np.asarray(data.get("wheel_angle_left_rad", []), dtype=float)
        self.commands = np.asarray(data.get("beamng_commands", []), dtype=float)
        if (
            self.angles.ndim != 1
            or self.angles.shape != self.commands.shape
            or len(self.angles) < 3
        ):
            raise ValueError("Calibration needs at least three paired angle/command samples")
        if not np.isfinite(self.angles).all() or not np.isfinite(self.commands).all():
            raise ValueError("Calibration samples must be finite")
        if not (np.diff(self.angles) > 0).all() or not (np.diff(self.commands) < 0).all():
            raise ValueError("Angles must increase positive-left while BeamNG commands decrease")
        if abs(self.commands).max() > 1 or not self.angles[0] < 0 < self.angles[-1]:
            raise ValueError("Calibration must span neutral and normalized command limits")
        if not math.isclose(float(np.interp(0, self.angles, self.commands)), 0.0, abs_tol=1e-6):
            raise ValueError("Calibration must map neutral angle to neutral command")
        self.metadata = data

    @classmethod
    def load(cls, path: str, *, allow_simulation_experiment: bool = False):
        if not path or not Path(path).is_file():
            raise ValueError(
                "Imitation requires imitation.calibration_path pointing to measured simulation calibration; no direct angle-to-command assumption is enabled"
            )
        return cls(
            json.loads(Path(path).read_text(encoding="utf-8")),
            allow_simulation_experiment=allow_simulation_experiment,
        )

    def validate_simulator(self, config: PipelineConfig) -> None:
        if self.experimental:
            camera = json.loads(json.dumps(asdict(config.camera)))
            if (
                config.beamng_vehicle != self.metadata["simulator_vehicle"]
                or camera != self.metadata["simulator_camera"]
            ):
                raise ValueError("Simulator vehicle or camera differs from measured calibration")
            if not math.isclose(config.imitation.real_wheelbase_m, self.real_wheelbase):
                raise ValueError("Recording wheelbase hypothesis differs from calibration")

    def command(self, angle: float) -> float:
        if not math.isfinite(angle):
            raise ValueError("Predicted steering must be finite")
        return float(np.interp(angle, self.angles, self.commands))


def match_camera_fov(image: np.ndarray, source_fov_deg: float, target_fov_deg: float):
    if not 0 < target_fov_deg <= source_fov_deg < 180:
        raise ValueError("The simulator camera must cover the training camera's field of view")
    ratio = math.tan(math.radians(target_fov_deg) / 2) / math.tan(math.radians(source_fov_deg) / 2)
    height, width = image.shape[:2]
    crop_width = max(1, int(round(width * ratio)))
    crop_height = max(1, int(round(height * ratio)))
    x = (width - crop_width) // 2
    y = (height - crop_height) // 2
    return image[y : y + crop_height, x : x + crop_width]


class ImitationController(StanleyController):
    def __init__(
        self,
        config: PipelineConfig,
        camera: CameraModel | None = None,
        *,
        predictor=None,
        calibration=None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        super().__init__(config, camera=camera, clock=clock)
        self.settings = config.imitation
        self.calibration = calibration
        if self.calibration is None:
            self.calibration = SteeringCalibration.load(
                self.settings.calibration_path,
                allow_simulation_experiment=self.settings.allow_simulation_experiment,
            )
        self.calibration.validate_simulator(config)
        if not math.isclose(config.wheelbase_m, self.calibration.simulator_wheelbase):
            raise ValueError("Controller geometry differs from calibrated simulator wheelbase")
        if self.calibration.experimental:
            self._target_speed = min(self._target_speed, self.settings.experiment_max_speed_mps)
            self._speed_limit = min(self._speed_limit, self.settings.experiment_max_speed_mps)
        self.predictor = predictor
        if self.predictor is None:
            from experiment.imitation.learning.model import SteeringPredictor

            self.predictor = SteeringPredictor(self.settings.checkpoint, self.settings.device)
        recorded = self.predictor.metadata["camera_calibration"]
        self.target_fov = math.degrees(2 * math.atan(recorded["width"] / (2 * recorded["k"][0])))
        self.source_fov = config.camera.fov_x_deg
        if self.target_fov > self.source_fov:
            raise ValueError("BeamNG camera FOV is narrower than the training camera")
        self._observation = None
        self._command_time = None
        # Kept here, not on SteeringDebug, so the shared debug type stays controller-neutral.
        self.last_angle_rad = 0.0
        self.last_inference_ms = 0.0

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        # A path-only caller cannot accidentally reuse an old camera observation.
        self.reset()
        return ControlCommand(parkingbrake=1.0, brake=self._max_brake)

    def compute_observation(
        self, plan: PathPlan, state: VehicleState, image: np.ndarray
    ) -> ControlCommand:
        if image is None or not state.valid or not math.isfinite(state.speed_mps):
            self.reset()
            return ControlCommand(parkingbrake=1.0, brake=self._max_brake)
        self._observation = image
        try:
            return super().compute(plan, state)
        except (ValueError, RuntimeError, OSError) as exc:
            logger.error("Imitation Safe Stop: %s", exc)
            self.reset()
            return ControlCommand(parkingbrake=1.0, brake=self._max_brake)
        finally:
            self._observation = None

    def _lateral_control(self, plan: PathPlan, state: VehicleState):
        debug, path = super()._lateral_control(plan, state)
        if path is None or not plan.valid or plan.fallback_active or plan.speed_scale <= 0:
            return debug, path
        image = match_camera_fov(self._observation, self.source_fov, self.target_fov)
        started = time.perf_counter()
        angle = self.predictor.predict(image, state.speed_mps)
        self.last_inference_ms = (time.perf_counter() - started) * 1000
        if self.last_inference_ms > self.settings.max_inference_ms:
            raise ValueError("Inference exceeded imitation.max_inference_ms")
        self.last_angle_rad = angle
        simulator_angle = math.atan(
            self.calibration.simulator_wheelbase * math.tan(angle) / self.calibration.real_wheelbase
        )
        command = self.calibration.command(simulator_angle)
        cap = self.steering_cap(state.speed_mps)
        now = self._clock()
        maximum_step = 0.0
        if self._command_time is not None:
            maximum_step = self.settings.steering_slew_per_s * max(0.0, now - self._command_time)
        self._command_time = now
        command = float(np.clip(command, -cap, cap))
        debug.desired_steering = float(
            np.clip(command, self._prev_steering - maximum_step, self._prev_steering + maximum_step)
        )
        return debug, path

    def reset(self):
        super().reset()
        self._observation = None
        self._command_time = None
