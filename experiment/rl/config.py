"""Settings for the optional residual reinforcement learning experiment."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RLConfig:
    policy_path: str = "models/rl_policy.npz"
    max_residual: float = 0.2
    cross_track_scale_m: float = 2.0
    heading_scale_rad: float = 0.5
    curvature_scale_inv_m: float = 0.15
    speed_scale_mps: float = 5.0
    iterations: int = 12
    directions: int = 4
    top_directions: int = 2
    learning_rate: float = 0.03
    exploration_std: float = 0.1
    train_episodes: int = 2
    eval_episodes: int = 6
    seed: int = 42
    radius_min_m: float = 8.0
    radius_max_m: float = 16.0
    fps_min: float = 3.0
    fps_max: float = 6.0
    start_yaw_max_rad: float = 0.3
    start_lateral_max_m: float = 0.5
    delay_max_frames: int = 3
    weight_error: float = 1.0
    weight_edge: float = 5.0
    weight_steer_change: float = 0.1
    failure_penalty: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.policy_path, str) or not self.policy_path.strip():
            raise ValueError("control.rl.policy_path must be a nonempty path")
        integers = {
            "iterations",
            "directions",
            "top_directions",
            "train_episodes",
            "eval_episodes",
            "seed",
            "delay_max_frames",
        }
        nonnegative = {"seed", "start_yaw_max_rad", "start_lateral_max_m"}
        for item in fields(self):
            if item.name == "policy_path":
                continue
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"control.rl.{item.name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"control.rl.{item.name} must be finite")
            if item.name in integers and not isinstance(value, int):
                raise ValueError(f"control.rl.{item.name} must be an integer")
            if item.name in nonnegative:
                if value < 0:
                    raise ValueError(f"control.rl.{item.name} must be >= 0")
            elif value <= 0:
                raise ValueError(f"control.rl.{item.name} must be > 0")
        if self.max_residual > 1.0:
            raise ValueError("control.rl.max_residual must be <= 1")
        if self.top_directions > self.directions:
            raise ValueError("control.rl.top_directions must be <= directions")
        if self.radius_max_m < self.radius_min_m or self.fps_max < self.fps_min:
            raise ValueError("control.rl sampling maxima must be >= their minima")
