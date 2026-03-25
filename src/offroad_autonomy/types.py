"""Shared data types for the offroad_autonomy pipeline.

Every pipeline stage communicates through these typed containers. Downstream
modules depend only on these types, never on upstream implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_PERCEPTION_PROMPTS = [
    "traversable road",
    "dirt road",
    "off-road trail",
    "drivable terrain",
    "gravel path",
]

DEFAULT_DASHBOARD_COLORS = {
    "BG": (18, 22, 26),
    "PANEL_BG": (28, 33, 38),
    "CARD_BG": (34, 40, 46),
    "CARD_BORDER": (54, 62, 70),
    "TEXT_PRIMARY": (238, 240, 242),
    "TEXT_SECONDARY": (155, 163, 172),
    "MUTED_LINE": (72, 80, 88),
    "MASK_FILL": (100, 172, 116),
    "MASK_EDGE": (182, 224, 193),
    "PATH_GLOW": (92, 184, 255),
    "PATH_CORE": (72, 174, 255),
    "PATH_HIGHLIGHT": (246, 249, 252),
    "GOOD": (102, 187, 106),
    "WARN": (82, 181, 233),
    "BAD": (90, 92, 225),
}


@dataclass
class FramePacket:
    """Raw and preprocessed camera frame with metadata."""

    raw: np.ndarray
    preprocessed: np.ndarray
    timestamp: float
    height: int
    width: int


@dataclass
class PerceptionResult:
    """Output of the road segmentation model."""

    mask: np.ndarray
    confidences: list[float] = field(default_factory=list)
    num_detections: int = 0
    inference_time_ms: float = 0.0


@dataclass
class StabilizedResult:
    """Temporally smoothed perception output."""

    mask: np.ndarray
    stability_score: float = 1.0
    raw_result: PerceptionResult | None = None


@dataclass
class PathPlan:
    """Planned centerline path through the traversable region."""

    centerline: np.ndarray
    heading_rad: float = 0.0
    curvature: float = 0.0
    road_width_px: float = 0.0
    kalman_active: bool = False


@dataclass
class VehicleState:
    """Current vehicle state from the simulator."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_mps: float = 0.0
    heading_rad: float = 0.0


@dataclass
class ControlCommand:
    """Actuator command sent to the simulator."""

    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass
class PipelineStepResult:
    """Outputs from one full pipeline iteration."""

    frame: FramePacket
    perception: PerceptionResult
    stabilized: StabilizedResult
    plan: PathPlan
    command: ControlCommand


@dataclass
class PipelineConfig:
    """Structured representation of the full configuration."""

    beamng_home: str = ""
    beamng_host: str = "localhost"
    beamng_port: int = 64256
    beamng_map: str = "automation_test_track"
    beamng_vehicle: str = "pickup"
    beamng_spawn_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fov: float = 120.0
    camera_pos: list[float] = field(default_factory=lambda: [0, -2.5, 0.8])
    camera_dir: list[float] = field(default_factory=lambda: [0, -1, -0.1])
    map_spawns: dict = field(default_factory=dict)

    model_weights: str = "models/yoloe-26x-seg.pt"
    confidence_threshold: float = 0.25
    perception_input_size: int = 640
    perception_prompts: list[str] = field(
        default_factory=lambda: DEFAULT_PERCEPTION_PROMPTS.copy()
    )

    preprocess_width: int = 640
    preprocess_height: int = 360
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    ema_alpha: float = 0.7
    min_mask_area_fraction: float = 0.001
    morphology_kernel_size: int = 5

    centerline_samples: int = 20
    kalman_process_noise: float = 1e-3
    kalman_measurement_noise: float = 1e-1
    fallback_after_n_misses: int = 3
    min_road_pixels: int = 500

    stanley_gain_k: float = 2.5
    stanley_softening: float = 1.0
    target_speed_mps: float = 5.0
    max_throttle: float = 0.6
    max_brake: float = 0.8
    speed_kp: float = 0.3
    dashboard_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: DEFAULT_DASHBOARD_COLORS.copy()
    )
