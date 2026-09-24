"""Typed containers shared by every pipeline stage.

Stages depend only on these types, never on each other's implementations,
so any stage can be replaced or tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

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

#: Stereo never sees the stitched image: its rotation-only mapping has a
#: parallax seam that would read as false disparity.
SEGMENTATION_MODES = ("left", "right", "stitched")

#: Keyboard order matters: key ``N`` selects ``DEBUG_VIEWS[N]``.
DEBUG_VIEWS = (
    "default",
    "raw",
    "rectified",
    "disparity",
    "depth",
    "stitched",
    "mask",
    "fused",
    "roi",
    "pipeline",
)

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

#: Rendered at a third of the imager per axis: two full-resolution streams
#: cost BeamNG more than the whole 50 ms loop budget, and the stack works at
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


@dataclass(frozen=True)
class StereoRigSpec:
    """The front stereo pair, declared once so both mounts derive from it.

    A single midpoint, baseline and attitude makes it impossible for the two
    cameras to drift apart in height, pitch or roll through a config typo.
    ``toe_out_deg`` widens the stitched view but eats stereo overlap, which
    is why it defaults to zero.
    """

    baseline_m: float = 0.6
    #: Front bumper, 0.17 m ahead of the Hopper's frontmost node, so no
    #: bodywork can project into frame at any field of view.
    center: tuple[float, float, float] = (0.0, -1.95, 0.95)
    pitch_deg: float = -2.0
    roll_deg: float = 0.0
    toe_out_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.baseline_m <= 0.0:
            raise ValueError("stereo_rig.baseline_m must be positive")

    @property
    def lateral_offset_m(self) -> float:
        return self.center[0]

    @property
    def forward_offset_m(self) -> float:
        return self.center[1]

    @property
    def height_m(self) -> float:
        return self.center[2]

    def mount(
        self, side: str
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """``(pos, dir, up)`` in vehicle space for ``left`` or ``right``."""
        if side not in ("left", "right"):
            raise ValueError(f"Unknown stereo side: {side!r}")
        # +X is the vehicle's left, so the left camera takes the positive half.
        if side == "left":
            sign = 1.0
        else:
            sign = -1.0
        cx, cy, cz = self.center
        pos = (cx + sign * self.baseline_m / 2.0, cy, cz)
        forward, up = mount_pose(self.pitch_deg, self.roll_deg, yaw_deg=sign * self.toe_out_deg)
        return pos, forward, up


DEFAULT_STEREO_RIG = StereoRigSpec()


@dataclass(frozen=True)
class DisplayRigSpec:
    """A camera that exists only to be looked at.

    It sits high and pitched down because that reads well to a person, which
    is the opposite of what perception wants. Keeping it off the compute path
    lets each camera be posed for its own job, and the dashboard reprojects
    results onto it instead of running a second segmentation.
    """

    enabled: bool = True
    name: str = "path_view"
    center: tuple[float, float, float] = (0.0, -0.30, 1.85)
    pitch_deg: float = -8.0
    roll_deg: float = 0.0

    @property
    def height_m(self) -> float:
        return self.center[2]

    @property
    def forward_offset_m(self) -> float:
        return self.center[1]

    def mount(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        forward, up = mount_pose(self.pitch_deg, self.roll_deg)
        return tuple(float(v) for v in self.center), forward, up


DEFAULT_DISPLAY_RIG = DisplayRigSpec()

#: The bumper pair sees no bodywork, so there is nothing to exclude. Re-run
#: ``scripts/derive_ego_mask.py`` after moving the rig: a polygon derived for
#: one pose is wrong for any other.
DEFAULT_LEFT_EGO_MASK = EgoMaskSpec(enabled=False, margin_px=1)
DEFAULT_RIGHT_EGO_MASK = EgoMaskSpec(enabled=False, margin_px=1)

#: Only a drawing clip, so the reprojected road mask is not painted over the
#: bonnet; nothing is computed from the display camera.
DEFAULT_DISPLAY_OVERLAY_CLIP = EgoMaskSpec(
    enabled=True,
    polygon=(
        (0.2242, 1.0000),
        (0.7769, 1.0000),
        (0.7070, 0.8887),
        (0.5057, 0.8627),
        (0.2941, 0.8887),
    ),
    margin_px=1,
)


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


def stereo_camera_spec(
    rig: StereoRigSpec,
    side: str,
    name: str,
    sensor: CameraSensor,
    ego_mask: EgoMaskSpec,
) -> CameraSpec:
    pos, direction, up = rig.mount(side)
    return CameraSpec(name=name, pos=pos, dir=direction, up=up, sensor=sensor, ego_mask=ego_mask)


DEFAULT_LEFT_CAMERA = stereo_camera_spec(
    DEFAULT_STEREO_RIG, "left", "left_bumper", GMSL2_CAPTURE_SENSOR, DEFAULT_LEFT_EGO_MASK
)
DEFAULT_RIGHT_CAMERA = stereo_camera_spec(
    DEFAULT_STEREO_RIG, "right", "right_bumper", GMSL2_CAPTURE_SENSOR, DEFAULT_RIGHT_EGO_MASK
)

#: Small and slow on purpose: it is rendered, never matched, so resolution
#: would buy nothing but BeamNG render time.
DISPLAY_SENSOR = CameraSensor(model="VIEW", width=960, height=540, fov_x_deg=95.0, target_fps=20.0)


def display_camera_spec(
    rig: DisplayRigSpec,
    sensor: CameraSensor,
    overlay_clip: EgoMaskSpec,
) -> CameraSpec:
    """The display camera as an ordinary ``CameraSpec``.

    It is deliberately unreachable from ``PipelineConfig.segmentation_camera``
    and ``build_camera_models``, which is what keeps it out of the compute path.
    """
    pos, direction, up = rig.mount()
    return CameraSpec(
        name=rig.name, pos=pos, dir=direction, up=up, sensor=sensor, ego_mask=overlay_clip
    )


DEFAULT_DISPLAY_CAMERA = display_camera_spec(
    DEFAULT_DISPLAY_RIG, DISPLAY_SENSOR, DEFAULT_DISPLAY_OVERLAY_CLIP
)


@dataclass
class StereoFramePair:
    """One capture from the front stereo pair, exactly as delivered.

    An unsynchronised pair is still fine for single-camera segmentation, but
    stereo must skip it: a one-frame offset at speed is a systematic disparity
    error that looks exactly like real geometry.
    """

    left: np.ndarray | None
    right: np.ndarray | None
    timestamp: float = 0.0
    frame_id: int = 0
    synchronized: bool = True
    read_skew_ms: float = 0.0
    #: False when neither camera rendered since the last pair, so there is
    #: nothing new for stereo to measure.
    is_new: bool = True
    sync_note: str = ""

    @property
    def has_stereo(self) -> bool:
        return self.left is not None and self.right is not None and self.synchronized

    def frame(self, side: str) -> np.ndarray | None:
        if side == "left":
            return self.left
        return self.right


@dataclass
class FramePacket:
    raw: np.ndarray
    preprocessed: np.ndarray
    timestamp: float
    height: int
    width: int


@dataclass
class DepthResult:
    """Metric depth from the stereo pair.

    ``depth_m``, ``valid`` and ``points_vehicle`` are on the segmentation grid
    so they fuse pixel for pixel with the mask; ``disparity`` and
    ``depth_rect`` stay on the rectified grid the matcher ran on.
    """

    depth_m: np.ndarray
    valid: np.ndarray
    points_vehicle: np.ndarray
    cloud_vehicle: np.ndarray
    disparity: np.ndarray | None = None
    depth_rect: np.ndarray | None = None
    rectified_left: np.ndarray | None = None
    rectified_right: np.ndarray | None = None
    coverage: float = 0.0
    valid_disparity_fraction: float = 0.0
    median_forward_depth_m: float = float("nan")
    min_corridor_depth_m: float = float("nan")
    compute_time_ms: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class TerrainAnalysis:
    height_above_ground: np.ndarray
    slope_rad: np.ndarray
    obstacle_mask: np.ndarray
    clearance_m: np.ndarray
    traversability: np.ndarray
    valid: np.ndarray
    range_m: np.ndarray | None = None
    #: Only triangulated pixels may raise obstacles; ground-plane-inferred
    #: ones sit on the plane by construction and prove nothing.
    measured: np.ndarray | None = None
    inferred: np.ndarray | None = None
    ground_plane: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ground_slope_deg: float = 0.0
    obstacle_fraction: float = 0.0
    coverage: float = 0.0
    measured_coverage: float = 0.0
    min_forward_clearance_m: float = float("inf")
    analyze_time_ms: float = 0.0


@dataclass
class StereoGeometry:
    """One result from the stereo worker.

    It carries its capture time because the planner uses the newest result
    and discounts it by age rather than waiting for a fresh one.
    """

    depth: DepthResult
    terrain: TerrainAnalysis | None
    frame_id: int
    capture_time: float
    completed_time: float
    latency_ms: float = 0.0

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.capture_time)


@dataclass
class TraversabilityMap:
    """Segmentation fused with depth.

    Depth may carve obstacles out of the mask, never add road to it: stereo
    is sparse and noisy at range, and false road would steer off the trail.
    """

    binary_mask: np.ndarray
    depth_map: np.ndarray
    confidence_map: np.ndarray
    valid_depth_mask: np.ndarray
    obstacle_mask: np.ndarray
    boundary_distance_m: np.ndarray
    forward_clearance_m: float = float("inf")
    depth_age_s: float = 0.0


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
    traversability: np.ndarray | None = None
    rgb_mask: np.ndarray | None = None
    valid_roi: np.ndarray | None = None
    road_fraction: float = 0.0


@dataclass
class StabilizedResult:
    mask: np.ndarray
    stability_score: float = 1.0
    raw_result: PerceptionResult | None = None
    traversability: np.ndarray | None = None
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
    frames: StereoFramePair | None = None
    depth: DepthResult | None = None
    terrain: TerrainAnalysis | None = None
    traversability: TraversabilityMap | None = None
    depth_age_s: float = float("inf")
    #: "", "off", "warming up" or "stale".
    depth_fallback: str = ""
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
    stereo_rig: StereoRigSpec = DEFAULT_STEREO_RIG
    left_camera: CameraSpec = field(default_factory=lambda: replace(DEFAULT_LEFT_CAMERA))
    right_camera: CameraSpec = field(default_factory=lambda: replace(DEFAULT_RIGHT_CAMERA))
    display_rig: DisplayRigSpec = DEFAULT_DISPLAY_RIG
    display_camera: CameraSpec = field(default_factory=lambda: replace(DEFAULT_DISPLAY_CAMERA))
    map_spawns: dict = field(default_factory=dict)

    sync_max_read_skew_ms: float = 8.0
    sync_require_both_new: bool = True

    model_weights: str = "models/yoloe-26x-seg.pt"
    confidence_threshold: float = 0.25
    perception_input_size: int = 640
    perception_prompts: list[str] = field(default_factory=lambda: DEFAULT_PERCEPTION_PROMPTS.copy())
    segmentation_mode: str = "left"

    stitch_enabled: bool = False
    stitch_feather_px: int = 24

    preprocess_width: int = 720
    preprocess_height: int = 465
    enable_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    depth_enabled: bool = True
    stereo_async: bool = True
    stereo_rate_hz: float = 10.0
    stereo_max_age_s: float = 0.4
    stereo_width: int = 720
    stereo_height: int = 465
    #: 0 derives the search range from baseline, focal length and min depth.
    stereo_num_disparities: int = 0
    stereo_max_disparities: int = 128
    stereo_block_size: int = 7
    stereo_p1_factor: int = 8
    stereo_p2_factor: int = 32
    stereo_uniqueness_ratio: int = 15
    stereo_speckle_window_size: int = 96
    stereo_speckle_range: int = 2
    stereo_disp12_max_diff: int = 1
    stereo_min_depth_m: float = 2.0
    stereo_max_depth_m: float = 40.0
    stereo_median_blur: int = 5
    stereo_rectify_alpha: float = 0.0
    #: ``None`` derives from the rig: BeamNG renders an ideal pinhole, so a
    #: real calibration is only needed for real cameras.
    stereo_K_left: tuple | None = None
    stereo_K_right: tuple | None = None
    stereo_D_left: tuple | None = None
    stereo_D_right: tuple | None = None
    stereo_R: tuple | None = None
    stereo_T: tuple | None = None

    depth_roi_enabled: bool = True
    depth_roi_row_top: float = 0.35
    depth_roi_row_bottom: float = 1.0
    depth_roi_use_mask: bool = True
    depth_roi_mask_dilation_px: int = 25
    depth_roi_corridor_half_width_m: float = 4.0
    depth_roi_corridor_length_m: float = 35.0

    obstacle_height_m: float = 0.35
    drop_height_m: float = -0.45
    max_slope_deg: float = 22.0
    max_point_height_m: float = 5.0
    min_point_height_m: float = -3.0
    ground_fit_min_points: int = 400
    ground_fit_near_m: float = 2.0
    ground_fit_far_m: float = 18.0
    ground_fit_lateral_m: float = 5.0
    bev_forward_m: float = 30.0
    bev_lateral_m: float = 10.0
    bev_cell_m: float = 0.25
    bev_min_points_per_cell: int = 2
    bev_min_obstacle_ratio: float = 0.4
    vehicle_half_width_m: float = 0.95
    clearance_lookahead_m: float = 18.0
    ground_fill_enabled: bool = True
    ground_fill_max_m: float = 25.0
    depth_fusion_weight: float = 0.65
    depth_unknown_support: float = 0.6
    depth_inferred_support: float = 0.8

    safety_min_road_fraction: float = 0.015
    safety_no_road_time_s: float = 2.0

    ema_alpha: float = 0.7
    min_mask_area_fraction: float = 0.001
    morphology_kernel_size: int = 5
    enable_morphology: bool = True
    enable_ema: bool = True

    planner_mode: str = "baseline"
    planner_roi_height: float = 0.45
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
    planner_depth_clearance_weight: float = 0.35
    planner_min_clearance_m: float = 1.15
    planner_obstacle_penalty: float = 0.85

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
    #: No window at all: for containers and remote runs without a display.
    ui_headless: bool = False
    ui_render_every_n: int = 1
    ui_debug_view: str = "default"
    ui_timing_overlay: bool = False
    ui_display_async: bool = True
    ui_display_rate_hz: float = 20.0
    runtime_log_interval_s: float = 5.0

    @property
    def segmentation_camera(self) -> CameraSpec:
        if self.segmentation_mode == "right":
            return self.right_camera
        return self.left_camera
