"""Reproducible settings shared by extraction, training, and simulation inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ImitationConfig:
    dataset_dir: str = "datasets/imitation"
    run_dir: str = "output/imitation/baseline"
    checkpoint: str = "models/imitation_best.pt"
    calibration_path: str = ""
    allow_simulation_experiment: bool = False
    image_topic: str = "/image_processing/image_compressed"
    camera_info_topic: str = "/image_processing/camera_info"
    can_topic: str = "/CAN"
    sample_hz: float = 10.0
    image_width: int = 256
    image_height: int = 144
    crop_top: float = 0.0
    crop_bottom: float = 0.28
    jpeg_quality: int = 92
    max_alignment_ms: float = 20.0
    max_state_age_ms: float = 40.0
    max_gap_ms: float = 100.0
    split_guard_s: float = 5.0
    temporal_train_fraction: float = 0.6
    temporal_validation_fraction: float = 0.2
    label_scale: float = 1.0
    real_wheelbase_m: float = 2.815
    physics_min_correlation: float = 0.9
    physics_min_scale: float = 0.7
    physics_max_scale: float = 1.3
    min_turn_rad: float = 0.03
    max_angle_rad: float = 0.5585053606381855
    speed_scale_mps: float = 10.0
    batch_size: int = 64
    epochs: int = 8
    frozen_epochs: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    dropout: float = 0.1
    turn_sampling_weight: float = 4.0
    flip_probability: float = 0.5
    color_jitter: float = 0.15
    seed: int = 42
    workers: int = 0
    cpu_threads: int = 4
    pretrained: bool = True
    device: str = "auto"
    latency_warmup: int = 10
    latency_samples: int = 100
    max_inference_ms: float = 100.0
    steering_slew_per_s: float = 0.9
    route_cell_m: float = 30.0
    evaluation_seconds: float = 30.0
    connection_timeout_s: float = 2.0
    comparison_physics_hz: int = 60
    comparison_steps: int = 15
    calibration_points: int = 11
    calibration_settle_s: float = 1.0
    calibration_samples: int = 5
    calibration_sample_interval_s: float = 0.05
    calibration_neutral_tolerance_rad: float = 0.005
    calibration_repeatability_rad: float = 0.035
    calibration_stationary_speed_mps: float = 0.1
    experiment_max_speed_mps: float = 2.0

    def __post_init__(self) -> None:
        zero_allowed = {
            "crop_top",
            "crop_bottom",
            "split_guard_s",
            "frozen_epochs",
            "workers",
            "seed",
            "dropout",
            "flip_probability",
            "color_jitter",
            "weight_decay",
        }
        integers = {
            "comparison_physics_hz",
            "comparison_steps",
            "calibration_points",
            "calibration_samples",
            "image_width",
            "image_height",
            "jpeg_quality",
            "batch_size",
            "epochs",
            "frozen_epochs",
            "workers",
            "cpu_threads",
            "seed",
            "latency_warmup",
            "latency_samples",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {
                "dataset_dir",
                "run_dir",
                "checkpoint",
                "calibration_path",
                "image_topic",
                "camera_info_topic",
                "can_topic",
                "device",
            }:
                if not isinstance(value, str) or not value and item.name != "calibration_path":
                    raise ValueError(f"imitation.{item.name} must be a valid string")
                continue
            if item.name in {"pretrained", "allow_simulation_experiment"}:
                if not isinstance(value, bool):
                    raise TypeError(f"imitation.{item.name} must be boolean")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"imitation.{item.name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"imitation.{item.name} must be finite")
            if item.name in integers and not isinstance(value, int):
                raise ValueError(f"imitation.{item.name} must be an integer")
            if item.name in zero_allowed:
                if value < 0:
                    raise ValueError(f"imitation.{item.name} must be >= 0")
            elif value <= 0:
                raise ValueError(f"imitation.{item.name} must be > 0")
        if self.crop_top + self.crop_bottom >= 1:
            raise ValueError("imitation crop must preserve some image rows")
        if (
            not 0 < self.temporal_train_fraction < 1
            or not 0 < self.temporal_validation_fraction < 1
        ):
            raise ValueError("imitation split fractions must be between 0 and 1")
        if self.temporal_train_fraction + self.temporal_validation_fraction >= 1:
            raise ValueError("imitation split fractions must leave a test partition")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("imitation.device must be auto, cpu, or cuda")
        if self.calibration_points < 5 or self.calibration_points % 2 != 1:
            raise ValueError("imitation.calibration_points must be odd and at least five")
        if self.label_scale != 1.0:
            raise ValueError("Only the dynamics-checked raw-radians proxy is currently supported")
        if self.flip_probability > 1 or self.dropout >= 1 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("Invalid imitation probability or JPEG quality")
