"""Typed containers shared by every pipeline stage.

Stages depend only on these types, never on each other's implementations,
so any stage can be replaced or tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from offroad_autonomy.planning.grid_config import GridPlannerConfig

DEFAULT_PERCEPTION_PROMPTS = [
    "traversable road",
    "dirt road",
    "off-road trail",
    "drivable terrain",
    "gravel path",
]

DEFAULT_DASHBOARD_COLORS = {
    "BG": (17, 15, 14),
    "PANEL_BG": (27, 24, 22),
    "CARD_BG": (36, 32, 29),
    "CARD_BORDER": (58, 52, 48),
    "TEXT_PRIMARY": (255, 255, 255),
    "TEXT_SECONDARY": (180, 172, 166),
    "MUTED_LINE": (66, 59, 54),
    "MASK_FILL": (191, 179, 79),
    "MASK_EDGE": (240, 235, 189),
    "PATH_GLOW": (77, 154, 255),
    "PATH_CORE": (26, 122, 255),
    "PATH_HIGHLIGHT": (234, 244, 255),
    "GOOD": (138, 214, 91),
    "WARN": (59, 185, 245),
    "BAD": (91, 91, 255),
    # Distinct from MASK_FILL and BAD so the excluded hood cannot be mistaken
    # for traversable trail or for a fault.
    "EGO_EXCLUDED": (255, 91, 212),
}


@dataclass(frozen=True)
class DashboardThresholds:
    """What the dashboard colours its health readouts against.

    ``confidence_floor`` and ``road_floor`` are copied from the gate and the
    safe stop at load time, so the ticks on the bars cannot drift from the
    values that actually trigger them.
    """

    target_fps: float = 7.0
    fps_warn_fraction: float = 0.8
    latency_budget_ms: float = 143.0
    confidence_floor: float = 0.18
    confidence_good: float = 0.5
    road_floor: float = 0.015
    fps_bar_scale: float = 2.0
    latency_bar_scale: float = 1.4
    road_bar_full_scale: float = 0.2


DEBUG_VIEWS = (
    "default",
    "raw",
    "mask",
    "pipeline",
)

#: Keep the existing keys when depth-specific views are removed.
DEBUG_VIEW_KEYS = {0: "default", 1: "raw", 6: "mask", 9: "pipeline"}

#: ``shared_memory`` only works when BeamNG runs on the same machine; a
#: remote client (e.g. a Jetson) has to pull frames over the socket.
CAMERA_TRANSPORTS = ("shared_memory", "socket")


@dataclass(frozen=True)
class CameraSensor:
    """One physical camera model, shared by every mount that uses it.

    Field of view is horizontal because that is how datasheets state it;
    BeamNG wants the vertical angle, which is derived rather than configured
    so the two can never disagree.
    """

    model: str = "GMSL2"
    width: int = 2880
    height: int = 1860
    fov_x_deg: float = 120.0
    target_fps: float = 28.0

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.fov_x_deg) / 2.0)

    @property
    def fov_y_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.focal_px))

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / max(self.target_fps, 1e-6)

    def describe(self) -> str:
        return (
            f"{self.model} {self.width}x{self.height} "
            f"hfov={self.fov_x_deg:.0f} deg vfov={self.fov_y_deg:.1f} deg "
            f"f={self.focal_px:.1f} px @{self.target_fps:.0f} fps"
        )


GMSL2_SENSOR = CameraSensor()

#: Rendered at a third of the imager per axis to limit capture cost; the stack works at
#: 720x465 (exactly 3/4 of this) so intrinsics still scale uniformly.
GMSL2_CAPTURE_SENSOR = CameraSensor(width=960, height=620, target_fps=30.0)


@dataclass(frozen=True)
class EgoMaskSpec:
    """Image region occupied by the ego vehicle.

    Normalised coordinates let one polygon serve every working resolution.
    Pixels inside are *invalid*, not non-drivable: calling them non-drivable
    would convince the planner it is permanently boxed in by its own hood.
    """

    enabled: bool = False
    polygon: tuple[tuple[float, float], ...] = ()
    #: Dilation at working resolution so edge fringing cannot leak into the
    #: valid region.
    margin_px: int = 2

    def __post_init__(self) -> None:
        if self.enabled and len(self.polygon) < 3:
            raise ValueError("An enabled ego mask needs a polygon of >= 3 points")


def mount_pose(
    pitch_deg: float,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """``(dir, up)`` in BeamNG vehicle space (``+X`` left, ``-Y`` forward, ``+Z`` up).

    Every camera derives its orientation here so "pitch" means the same thing
    across the whole rig.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    forward = (
        math.sin(yaw) * math.cos(pitch),
        -math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
    )
    right = (-math.cos(yaw), -math.sin(yaw), 0.0)
    up0 = (
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    up = tuple(math.cos(roll) * u + math.sin(roll) * r for u, r in zip(up0, right))
    return forward, up


@dataclass
class CameraSpec:
    """Where one camera is mounted, in BeamNG vehicle space (metres)."""

    name: str
    pos: tuple[float, float, float]
    dir: tuple[float, float, float] = (0.0, -1.0, -0.1)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    sensor: CameraSensor = GMSL2_SENSOR
    ego_mask: EgoMaskSpec = EgoMaskSpec()

    @property
    def width(self) -> int:
        return self.sensor.width

    @property
    def height(self) -> int:
        return self.sensor.height

    @property
    def fov_x_deg(self) -> float:
        return self.sensor.fov_x_deg

    @property
    def fov_y_deg(self) -> float:
        return self.sensor.fov_y_deg

    @property
    def target_fps(self) -> float:
        return self.sensor.target_fps


# The existing dashcam mount supplies every stage, including the dashboard.
_DASHCAM_DIR, _DASHCAM_UP = mount_pose(-8.0)
DEFAULT_CAMERA = CameraSpec(
    name="dashcam",
    pos=(0.0, -0.30, 1.85),
    dir=_DASHCAM_DIR,
    up=_DASHCAM_UP,
    sensor=GMSL2_CAPTURE_SENSOR,
    ego_mask=EgoMaskSpec(
        enabled=True,
        polygon=(
            (0.1794, 1.0),
            (0.8220, 1.0),
            (0.6871, 0.7888),
            (0.6314, 0.7134),
            (0.5007, 0.6983),
            (0.3700, 0.7134),
            (0.3143, 0.7888),
        ),
        margin_px=1,
    ),
)


# A chase camera for the presentation video only. No pipeline stage reads it,
# so it needs no ego mask and its pose does not affect perception.
_ORBIT_DIR, _ORBIT_UP = mount_pose(-12.0)
DEFAULT_ORBIT_CAMERA = CameraSpec(
    name="orbit",
    pos=(0.0, 6.0, 2.6),
    dir=_ORBIT_DIR,
    up=_ORBIT_UP,
    sensor=CameraSensor(model="Orbit", width=960, height=728, fov_x_deg=95.0, target_fps=20.0),
)


@dataclass
class CameraFrame:
    """One dashcam capture shared by inference and display."""

    image: np.ndarray | None
    timestamp: float = 0.0
    frame_id: int = 0
    is_new: bool = True


@dataclass
class FramePacket:
    raw: np.ndarray
    preprocessed: np.ndarray
    timestamp: float
    height: int
    width: int


@dataclass
class PerceptionResult:
    """Segmentation output.

    ``road_fraction`` is measured over valid pixels only; it drives the safe
    stop because, unlike a mean detection score, bodywork in frame cannot
    move it.
    """

    mask: np.ndarray
    confidences: list[float] = field(default_factory=list)
    num_detections: int = 0
    inference_time_ms: float = 0.0
    valid_roi: np.ndarray | None = None
    road_fraction: float = 0.0


@dataclass
class StabilizedResult:
    mask: np.ndarray
    stability_score: float = 1.0
    raw_result: PerceptionResult | None = None
    valid_roi: np.ndarray | None = None
    road_fraction: float = 0.0


@dataclass
class PathPlan:
    centerline: np.ndarray
    heading_rad: float = 0.0
    curvature: float = 0.0
    road_width_px: float = 0.0
    kalman_active: bool = False
    min_clearance_m: float = float("inf")
    fallback_active: bool = False
    fallback_reason: str = ""
    speed_scale: float = 1.0
    planner_mask: np.ndarray | None = None
    roi_top: int = 0


@dataclass
class VehicleState:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_mps: float = 0.0
    heading_rad: float = 0.0
    valid: bool = True
    #: World-frame forward vector from the simulator. Kept separate from the
    #: quaternion so frame-to-frame motion never depends on its convention.
    direction: tuple[float, float, float] | None = None


@dataclass
class ControlCommand:
    steering: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    parkingbrake: float = 0.0
    #: ``None`` for manual input, which has no controller internals to show.
    debug: SteeringDebug | None = None


@dataclass
class SteeringDebug:
    """Controller internals for one frame, for the dashboard and logs.

    Lateral quantities are positive to the right, matching BeamNG steering.
    """

    cross_track_m: float = 0.0
    heading_error_rad: float = 0.0
    desired_steering: float = 0.0
    final_steering: float = 0.0
    lookahead_m: float = 0.0
    lookahead_px: tuple[float, float] | None = None
    authority: float = 1.0
    path_end_m: float = float("inf")
    curvature: float = 0.0
    max_curvature_ahead: float = 0.0
    feedforward_steering: float = 0.0
    target_speed_mps: float = 0.0
    speed_reason: str = ""
    saturation: float = 0.0
    rejoin_m: float = 0.0
    centering_shift_m: float = 0.0
    #: ``inf`` = edge outside the camera's view, ``nan`` = no mask.
    left_boundary_m: float = float("nan")
    right_boundary_m: float = float("nan")
    edge_clearance_m: float = float("nan")
    edge_risk: float = 0.0
    lateral_accel_mps2: float = 0.0
    cross_track_rate_mps: float = 0.0
    boundary_px: list = field(default_factory=list)


@dataclass
class PipelineStepResult:
    frame: FramePacket
    perception: PerceptionResult
    stabilized: StabilizedResult
    plan: PathPlan
    command: ControlCommand
    capture: CameraFrame | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Structured representation of the full configuration.

    Every tuning rationale lives next to its key in ``configs/default.yaml``.
    """

    beamng_home: str = ""
    beamng_host: str = "localhost"
    beamng_port: int = 64256
    #: False when attaching to an already running (usually remote) simulator.
    beamng_launch: bool = True
    beamng_camera_transport: str = "shared_memory"
    beamng_map: str = "automation_test_track"
    beamng_vehicle: str = "pickup"
    beamng_spawn_index: int = 0
    camera: CameraSpec = field(default_factory=lambda: replace(DEFAULT_CAMERA))
    #: Attached only with --presentation.
    orbit_camera: CameraSpec = field(default_factory=lambda: replace(DEFAULT_ORBIT_CAMERA))
    map_spawns: dict = field(default_factory=dict)

    model_weights: str = "models/yoloe-26x-seg.pt"
    confidence_threshold: float = 0.25
    perception_input_size: int = 640
    perception_prompts: list[str] = field(default_factory=lambda: DEFAULT_PERCEPTION_PROMPTS.copy())
    preprocess_width: int = 720
    preprocess_height: int = 465
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    safety_min_road_fraction: float = 0.015
    safety_no_road_time_s: float = 2.0

    ema_alpha: float = 0.7
    min_mask_area_fraction: float = 0.001
    morphology_kernel_size: int = 5
    enable_morphology: bool = True
    enable_ema: bool = True

    planner_mode: str = "baseline"
    planner_roi_height: float = 0.50
    baseline_temporal_blend: float = 0.0
    baseline_max_shift_m: float = 0.5

    gate_min_confidence: float = 0.18
    gate_min_mask_area: float = 0.05
    gate_hold_frames: int = 10
    gate_hold_speed_scale: float = 0.4

    centerline_samples: int = 20
    planner_backend: str = "heuristic"
    planner_horizon_fraction: float = 0.82
    planner_smoothing_window: int = 7
    planner_clearance_weight: float = 0.65
    planner_prior_std_fraction: float = 0.10
    planner_min_confidence: float = 0.18
    planner_segment_center_weight: float = 0.68
    planner_temporal_blend: float = 0.58
    planner_max_lateral_step_px: float = 32.0
    planner_straight_blend: float = 0.72
    planner_straight_residual_px: float = 4.0
    planner_straight_heading_threshold: float = 0.08
    kalman_process_noise: float = 1e-3
    kalman_measurement_noise: float = 1e-1
    fallback_after_n_misses: int = 3
    min_road_pixels: int = 500

    grid: GridPlannerConfig = field(default_factory=GridPlannerConfig)
    stanley_gain_k: float = 1.5
    stanley_softening: float = 2.4
    stanley_heading_gain: float = 0.85
    steering_ema_alpha: float = 0.45
    max_steering_delta: float = 0.3
    lookahead_base_m: float = 3.0
    lookahead_time_s: float = 0.6
    lookahead_min_m: float = 2.0
    lookahead_max_m: float = 8.0
    cross_track_tolerance_m: float = 0.3
    steer_full_authority_speed_mps: float = 3.0
    steer_speed_falloff: float = 0.08
    wheelbase_m: float = 2.6
    camera_ahead_of_rear_axle_m: float = 1.4
    max_wheel_angle_deg: float = 32.0
    path_end_margin_m: float = 1.5
    path_end_decel_mps2: float = 2.0
    max_lateral_accel_mps2: float = 0.5
    curve_decel_mps2: float = 0.8
    control_latency_s: float = 0.35
    lookahead_curvature_gain: float = 2.0
    curvature_feedforward_gain: float = 1.0
    saturation_start: float = 0.5
    saturation_min_speed_scale: float = 0.35
    steer_cap_full_speed_mps: float = 2.0
    steer_cap_high_speed_mps: float = 5.0
    steer_cap_at_high_speed: float = 0.6
    max_target_speed_increase_mps: float = 0.1
    comfort_brake: float = 0.35
    min_curvature_span_m: float = 2.0
    cross_track_soft_deadzone: bool = True
    latency_pose_prediction: bool = True
    cross_track_lookahead_gain: float = 1.0
    cross_track_min_lookahead_m: float = 1.5
    cross_track_recovery_m: float = 0.5
    vehicle_half_width_m: float = 0.95
    edge_margin_m: float = 0.75
    edge_centering_gain: float = 0.7
    edge_speed_reduction: float = 0.6
    drift_speed_gain: float = 2.0
    max_measured_lateral_accel_mps2: float = 0.8
    target_speed_mph: float = 12.0
    speed_limit_mph: float = 15.0
    min_turn_speed_mph: float = 7.0
    max_throttle: float = 0.45
    max_brake: float = 0.8
    speed_kp: float = 0.22
    clearance_slow_m: float = 6.0
    clearance_stop_m: float = 2.5
    dashboard_colors: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: DEFAULT_DASHBOARD_COLORS.copy()
    )
    dashboard_thresholds: DashboardThresholds = field(default_factory=DashboardThresholds)
    #: No window at all: for containers and remote runs without a display.
    ui_headless: bool = False
    ui_render_every_n: int = 1
    ui_debug_view: str = "default"
    ui_timing_overlay: bool = False
    ui_display_async: bool = True
    ui_display_rate_hz: float = 20.0
    runtime_log_interval_s: float = 5.0
    recording_fps: float = 20.0
    recording_crf: int = 23
    recording_preset: str = "veryfast"
    recording_queue_frames: int = 8
