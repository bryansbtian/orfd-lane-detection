"""A learned steering residual, applied before Stanley's smoothing and speed law.

Only planner and vehicle observations enter the policy. Training rewards can use
offline ground truth, but inference never reads simulator geometry or pose.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from experiment.rl.config import RLConfig
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, PipelineConfig, SteeringDebug, VehicleState

FEATURES = (
    "cross_track",
    "heading_error",
    "curvature",
    "speed",
    "previous_steering",
    "stanley_steering",
)
POLICY_VERSION = 1


class ResidualPolicy:
    def __init__(self, weights: np.ndarray, settings: RLConfig) -> None:
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        if self.weights.shape != (len(FEATURES),) or not np.isfinite(self.weights).all():
            raise ValueError(f"RL policy must contain {len(FEATURES)} finite weights")
        self.settings = settings

    def correction(self, debug: SteeringDebug, speed: float, previous: float) -> float:
        cfg = self.settings
        observation = np.array(
            [
                debug.cross_track_m / cfg.cross_track_scale_m,
                debug.heading_error_rad / cfg.heading_scale_rad,
                debug.curvature / cfg.curvature_scale_inv_m,
                speed / cfg.speed_scale_mps,
                previous,
                debug.desired_steering,
            ]
        )
        if not np.isfinite(observation).all():
            return 0.0
        return float(cfg.max_residual * np.tanh(self.weights @ np.clip(observation, -1.0, 1.0)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A file handle prevents numpy from silently appending another suffix.
        with path.open("wb") as stream:
            np.savez(
                stream,
                version=POLICY_VERSION,
                features=np.array(FEATURES),
                weights=self.weights,
                scales=self._scales(self.settings),
            )

    @staticmethod
    def _scales(settings: RLConfig) -> np.ndarray:
        return np.array(
            [
                settings.max_residual,
                settings.cross_track_scale_m,
                settings.heading_scale_rad,
                settings.curvature_scale_inv_m,
                settings.speed_scale_mps,
            ]
        )

    @classmethod
    def load(cls, path: str | Path, settings: RLConfig) -> ResidualPolicy:
        path = Path(path)
        if not path.is_file():
            raise ValueError(
                f"RL policy not found: {path}. Train with python -m experiment.rl.train first, "
                "or select a checkpoint with --rl-policy."
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                if data["version"].shape != () or data["version"].item() != POLICY_VERSION:
                    raise ValueError("unsupported policy version")
                if not np.array_equal(data["features"], np.array(FEATURES)):
                    raise ValueError("policy feature layout does not match")
                if not np.array_equal(data["scales"], cls._scales(settings)):
                    raise ValueError(
                        "policy normalization or residual limit differs from control.rl"
                    )
                return cls(data["weights"], settings)
        except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as exc:
            raise ValueError(f"Cannot load RL policy {path}: {exc}") from exc


class RLController(StanleyController):
    def __init__(
        self,
        config: PipelineConfig,
        camera: CameraModel | None = None,
        *,
        policy: ResidualPolicy | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__(config, camera=camera, clock=clock)
        self.policy = policy
        # Kept here, not on SteeringDebug, so the shared debug type stays controller-neutral.
        self.last_residual = 0.0
        if self.policy is None:
            self.policy = ResidualPolicy.load(config.rl.policy_path, config.rl)

    def _lateral_control(self, plan: PathPlan, state: VehicleState):
        debug, path = super()._lateral_control(plan, state)
        # Held or stopped paths must keep the existing fallback behavior.
        if path is None or not plan.valid or plan.fallback_active or plan.speed_scale <= 0.0:
            return debug, path
        residual = self.policy.correction(debug, state.speed_mps, self._prev_steering)
        cap = self.steering_cap(state.speed_mps)
        self.last_residual = residual
        debug.desired_steering = float(np.clip(debug.desired_steering + residual, -cap, cap))
        return debug, path
