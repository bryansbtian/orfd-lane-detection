"""YAML -> ``PipelineConfig``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from offroad_autonomy.types import (
    CAMERA_TRANSPORTS,
    DEBUG_VIEWS,
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_DISPLAY_CAMERA,
    DEFAULT_DISPLAY_RIG,
    DEFAULT_LEFT_CAMERA,
    DEFAULT_PERCEPTION_PROMPTS,
    DEFAULT_RIGHT_CAMERA,
    DEFAULT_STEREO_RIG,
    DISPLAY_SENSOR,
    GMSL2_CAPTURE_SENSOR,
    SEGMENTATION_MODES,
    CameraSensor,
    CameraSpec,
    DisplayRigSpec,
    EgoMaskSpec,
    PipelineConfig,
    StereoRigSpec,
    display_camera_spec,
)

logger = logging.getLogger("offroad_autonomy.config")

_TRUE_STRINGS = ("1", "true", "yes", "on")


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_dashboard_colors(raw: object) -> dict[str, tuple[int, int, int]]:
    colors = DEFAULT_DASHBOARD_COLORS.copy()
    if not isinstance(raw, dict):
        return colors
    for key, default in DEFAULT_DASHBOARD_COLORS.items():
        colors[key] = _parse_color(raw.get(key), default)
    return colors


def _parse_vec3(
    raw: object,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_sensor(raw: object, default: CameraSensor = GMSL2_CAPTURE_SENSOR) -> CameraSensor:
    if not isinstance(raw, dict):
        return default
    return CameraSensor(
        model=str(raw.get("model", default.model)),
        width=int(raw.get("width", default.width)),
        height=int(raw.get("height", default.height)),
        fov_x_deg=float(raw.get("fov_h", default.fov_x_deg)),
        target_fps=float(raw.get("target_fps", default.target_fps)),
    )


def _load_ego_mask(raw: object, default: EgoMaskSpec) -> EgoMaskSpec:
    """A malformed polygon fails loudly: silently excluding the wrong region
    would hide real road or expose bodywork as terrain."""
    if not isinstance(raw, dict):
        return default

    enabled = bool(raw.get("enabled", default.enabled))
    margin = int(raw.get("margin_px", default.margin_px))

    raw_polygon = raw.get("polygon")
    if raw_polygon is None:
        polygon = default.polygon
    else:
        points: list[tuple[float, float]] = []
        if isinstance(raw_polygon, list):
            for point in raw_polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        polygon = tuple(points)

    if enabled and len(polygon) < 3:
        raise ValueError("cameras.*.ego_mask is enabled but its polygon has fewer than 3 points")
    return EgoMaskSpec(enabled=enabled, polygon=polygon, margin_px=margin)


def _load_camera(
    raw: object,
    default: CameraSpec,
    sensor: CameraSensor,
    rig: StereoRigSpec,
    side: str,
) -> CameraSpec:
    """An explicit ``pos``/``dir``/``up`` still wins so a deliberately bad rig
    can be tested, and is logged because it breaks the shared-attitude
    guarantee."""
    block = raw
    if not isinstance(block, dict):
        block = {}
    pos, direction, up = rig.mount(side)
    explicit = [key for key in ("pos", "dir", "up") if key in block]
    if explicit:
        logger.warning(
            "cameras.%s overrides %s from stereo_rig - the pair may no longer "
            "share height/pitch/roll",
            side,
            ", ".join(explicit),
        )
    return CameraSpec(
        name=str(block.get("name", default.name)),
        pos=_parse_vec3(block.get("pos"), pos),
        dir=_parse_vec3(block.get("dir"), direction),
        up=_parse_vec3(block.get("up"), up),
        sensor=sensor,
        ego_mask=_load_ego_mask(block.get("ego_mask"), default.ego_mask),
    )


def _mount_center(raw: dict, default: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(raw.get("lateral_offset_m", default[0])),
        float(raw.get("forward_offset_m", default[1])),
        float(raw.get("height_m", default[2])),
    )


def _load_stereo_rig(raw: object) -> StereoRigSpec:
    if not isinstance(raw, dict):
        return DEFAULT_STEREO_RIG
    d = DEFAULT_STEREO_RIG
    return StereoRigSpec(
        baseline_m=float(raw.get("baseline_m", d.baseline_m)),
        center=_mount_center(raw, d.center),
        pitch_deg=float(raw.get("pitch_deg", d.pitch_deg)),
        roll_deg=float(raw.get("roll_deg", d.roll_deg)),
        toe_out_deg=float(raw.get("toe_out_deg", d.toe_out_deg)),
    )


def _load_display_rig(raw: dict) -> DisplayRigSpec:
    d = DEFAULT_DISPLAY_RIG
    return DisplayRigSpec(
        enabled=bool(raw.get("enabled", d.enabled)),
        name=str(raw.get("name", d.name)),
        center=_mount_center(raw, d.center),
        pitch_deg=float(raw.get("pitch_deg", d.pitch_deg)),
        roll_deg=float(raw.get("roll_deg", d.roll_deg)),
    )


def _load_display_camera(raw: object) -> tuple[DisplayRigSpec, CameraSpec]:
    """Read separately from the stereo sensor: it is rendered, never matched,
    so it is free to differ in resolution, field of view and frame rate."""
    block = raw
    if not isinstance(block, dict):
        block = {}
    rig = _load_display_rig(block)
    sensor = _load_sensor(block.get("sensor"), DISPLAY_SENSOR)
    clip = _load_ego_mask(block.get("overlay_clip"), DEFAULT_DISPLAY_CAMERA.ego_mask)
    return rig, display_camera_spec(rig, sensor, clip)


def _load_camera_rig(cameras: dict) -> tuple[StereoRigSpec, CameraSpec, CameraSpec]:
    sensor = _load_sensor(cameras.get("sensor"))
    rig = _load_stereo_rig(cameras.get("stereo_rig"))
    return (
        rig,
        _load_camera(cameras.get("left"), DEFAULT_LEFT_CAMERA, sensor, rig, "left"),
        _load_camera(cameras.get("right"), DEFAULT_RIGHT_CAMERA, sensor, rig, "right"),
    )


def _optional_matrix(raw: object) -> tuple | None:
    if raw is None:
        return None
    flat: list[float] = []

    def _walk(value: object) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                _walk(item)
        else:
            flat.append(float(value))

    _walk(raw)
    return tuple(flat)


def _deep_merge(base: dict, override: dict) -> dict:
    """Lists are replaced, not merged, so an overlay can shorten a polygon."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Resolves ``extends:`` so experiment and deployment configs only state
    what they change."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, seen + (path,)))}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_ref = raw.pop("extends", None)
    if base_ref:
        base = _read_yaml((path.parent / str(base_ref)), seen + (path,))
        raw = _deep_merge(base, raw)
    return raw


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _planner_mode(raw: object) -> str:
    mode = str(raw).lower()
    if mode not in ("baseline", "advanced"):
        raise ValueError(f"planning.mode must be 'baseline' or 'advanced', got {mode!r}")
    return mode


def _camera_transport(raw: object) -> str:
    transport = str(raw).lower()
    if transport not in CAMERA_TRANSPORTS:
        raise ValueError(
            f"beamng.camera_transport must be one of {CAMERA_TRANSPORTS}, got {transport!r}"
        )
    return transport


def _apply_env_overrides(bng: dict) -> dict:
    """The simulator's address differs per deployment, so it can be set
    without editing a YAML file baked into a container image."""
    bng = dict(bng)
    if os.environ.get("BEAMNG_HOST"):
        bng["host"] = os.environ["BEAMNG_HOST"]
    if os.environ.get("BEAMNG_PORT"):
        bng["port"] = int(os.environ["BEAMNG_PORT"])
    if os.environ.get("BEAMNG_HOME"):
        bng["home"] = os.environ["BEAMNG_HOME"]
    if os.environ.get("BEAMNG_LAUNCH"):
        bng["launch"] = os.environ["BEAMNG_LAUNCH"].lower() in _TRUE_STRINGS
    return bng


def load_config(path: str | Path) -> PipelineConfig:
    raw = _read_yaml(Path(path))

    bng = _apply_env_overrides(_section(raw, "beamng"))
    cameras = _section(bng, "cameras")
    stereo_rig, left_cam, right_cam = _load_camera_rig(cameras)
    display_rig, display_cam = _load_display_camera(cameras.get("visualization"))
    sync = _section(cameras, "sync")
    perc = _section(raw, "perception")
    stitching = _section(raw, "stitching")
    depth = _section(raw, "depth")
    calibration = _section(depth, "calibration")
    roi = _section(depth, "roi")
    ui = _section(raw, "ui")
    terrain = _section(raw, "terrain")
    safety = _section(raw, "safety")
    pre = _section(raw, "preprocessing")
    post = _section(raw, "postprocessing")
    plan = _section(raw, "planning")
    gate = _section(plan, "gate")
    ctrl = _section(raw, "control")
    dashboard = _section(_section(raw, "visualization"), "dashboard")

    mode = str(perc.get("segmentation_mode", "left")).lower()
    if mode not in SEGMENTATION_MODES:
        raise ValueError(
            f"perception.segmentation_mode must be one of {SEGMENTATION_MODES}, got {mode!r}"
        )
    debug_view = str(ui.get("debug_view", "default")).lower()
    if debug_view not in DEBUG_VIEWS:
        raise ValueError(f"ui.debug_view must be one of {DEBUG_VIEWS}, got {debug_view!r}")

    return PipelineConfig(
        beamng_home=str(bng.get("home", "")),
        beamng_host=str(bng.get("host", "localhost")),
        beamng_port=int(bng.get("port", 64256)),
        beamng_launch=bool(bng.get("launch", True)),
        beamng_camera_transport=_camera_transport(bng.get("camera_transport", "shared_memory")),
        beamng_map=bng.get("map", "automation_test_track"),
        beamng_vehicle=bng.get("vehicle", "pickup"),
        beamng_spawn_index=bng.get("spawn_index", 0),
        stereo_rig=stereo_rig,
        left_camera=left_cam,
        right_camera=right_cam,
        display_rig=display_rig,
        display_camera=display_cam,
        map_spawns=bng.get("maps", {}),
        sync_max_read_skew_ms=float(sync.get("max_read_skew_ms", 8.0)),
        sync_require_both_new=bool(sync.get("require_both_new", True)),
        segmentation_mode=mode,
        stitch_enabled=bool(stitching.get("enabled", False)),
        stitch_feather_px=int(stitching.get("feather_px", 24)),
        model_weights=perc.get("model_weights", "models/yoloe-26x-seg.pt"),
        confidence_threshold=perc.get("segmentation_threshold", 0.25),
        perception_input_size=perc.get("input_size", 640),
        perception_prompts=perc.get("prompts", DEFAULT_PERCEPTION_PROMPTS.copy()),
        preprocess_width=pre.get("target_width", 720),
        preprocess_height=pre.get("target_height", 465),
        enable_clahe=pre.get("enable_clahe", False),
        clahe_clip_limit=pre.get("clahe_clip_limit", 2.0),
        clahe_grid_size=pre.get("clahe_grid_size", 8),
        depth_enabled=bool(depth.get("enabled", True)),
        stereo_async=bool(depth.get("async", True)),
        stereo_rate_hz=float(depth.get("rate_hz", 10.0)),
        stereo_max_age_s=float(depth.get("max_age_s", 0.4)),
        stereo_width=depth.get("stereo_width", 720),
        stereo_height=depth.get("stereo_height", 465),
        stereo_num_disparities=depth.get("num_disparities", 0),
        stereo_max_disparities=depth.get("max_disparities", 128),
        stereo_block_size=depth.get("block_size", 7),
        stereo_p1_factor=int(depth.get("p1_factor", 8)),
        stereo_p2_factor=int(depth.get("p2_factor", 32)),
        stereo_uniqueness_ratio=depth.get("uniqueness_ratio", 15),
        stereo_speckle_window_size=depth.get("speckle_window_size", 96),
        stereo_speckle_range=depth.get("speckle_range", 2),
        stereo_disp12_max_diff=depth.get("disp12_max_diff", 1),
        stereo_min_depth_m=depth.get("min_depth_m", 2.0),
        stereo_max_depth_m=depth.get("max_depth_m", 40.0),
        stereo_median_blur=depth.get("median_blur", 5),
        stereo_rectify_alpha=float(depth.get("rectify_alpha", 0.0)),
        stereo_K_left=_optional_matrix(calibration.get("K_left")),
        stereo_K_right=_optional_matrix(calibration.get("K_right")),
        stereo_D_left=_optional_matrix(calibration.get("D_left")),
        stereo_D_right=_optional_matrix(calibration.get("D_right")),
        stereo_R=_optional_matrix(calibration.get("R")),
        stereo_T=_optional_matrix(calibration.get("T")),
        depth_roi_enabled=bool(roi.get("enabled", True)),
        depth_roi_row_top=float(roi.get("row_top", 0.35)),
        depth_roi_row_bottom=float(roi.get("row_bottom", 1.0)),
        depth_roi_use_mask=bool(roi.get("use_road_mask", True)),
        depth_roi_mask_dilation_px=int(roi.get("mask_dilation_px", 25)),
        depth_roi_corridor_half_width_m=float(roi.get("corridor_half_width_m", 4.0)),
        depth_roi_corridor_length_m=float(roi.get("corridor_length_m", 35.0)),
        obstacle_height_m=terrain.get("obstacle_height_m", 0.35),
        drop_height_m=terrain.get("drop_height_m", -0.45),
        max_slope_deg=terrain.get("max_slope_deg", 22.0),
        max_point_height_m=terrain.get("max_point_height_m", 5.0),
        min_point_height_m=terrain.get("min_point_height_m", -3.0),
        ground_fit_min_points=terrain.get("ground_fit_min_points", 400),
        ground_fit_near_m=terrain.get("ground_fit_near_m", 2.0),
        ground_fit_far_m=terrain.get("ground_fit_far_m", 18.0),
        ground_fit_lateral_m=terrain.get("ground_fit_lateral_m", 5.0),
        bev_forward_m=terrain.get("bev_forward_m", 30.0),
        bev_lateral_m=terrain.get("bev_lateral_m", 10.0),
        bev_cell_m=terrain.get("bev_cell_m", 0.25),
        bev_min_points_per_cell=terrain.get("bev_min_points_per_cell", 2),
        bev_min_obstacle_ratio=terrain.get("bev_min_obstacle_ratio", 0.4),
        vehicle_half_width_m=terrain.get("vehicle_half_width_m", 0.95),
        clearance_lookahead_m=terrain.get("clearance_lookahead_m", 18.0),
        ground_fill_enabled=bool(terrain.get("ground_fill_enabled", True)),
        ground_fill_max_m=terrain.get("ground_fill_max_m", 25.0),
        depth_fusion_weight=terrain.get("fusion_weight", 0.65),
        depth_unknown_support=terrain.get("unknown_support", 0.6),
        depth_inferred_support=terrain.get("inferred_support", 0.8),
        safety_min_road_fraction=safety.get("min_road_fraction", 0.015),
        safety_no_road_time_s=safety.get("no_road_time_s", 2.0),
        ema_alpha=post.get("ema_alpha", 0.7),
        min_mask_area_fraction=post.get("min_mask_area_fraction", 0.001),
        morphology_kernel_size=post.get("morphology_kernel_size", 5),
        enable_morphology=bool(post.get("enable_morphology", True)),
        enable_ema=bool(post.get("enable_ema", True)),
        planner_mode=_planner_mode(plan.get("mode", "baseline")),
        planner_roi_height=float(plan.get("roi_height", 0.45)),
        baseline_temporal_blend=float(plan.get("baseline_temporal_blend", 0.0)),
        baseline_max_shift_m=float(plan.get("baseline_max_shift_m", 0.5)),
        gate_min_confidence=float(gate.get("confidence_threshold", 0.18)),
        gate_min_mask_area=float(gate.get("min_mask_area", 0.05)),
        gate_hold_frames=int(gate.get("hold_frames", 10)),
        gate_hold_speed_scale=float(gate.get("hold_speed_scale", 0.4)),
        centerline_samples=plan.get("centerline_samples", 20),
        planner_backend=plan.get("backend", "heuristic"),
        planner_horizon_fraction=plan.get("horizon_fraction", 0.82),
        planner_smoothing_window=plan.get("smoothing_window", 7),
        planner_clearance_weight=plan.get("clearance_weight", 0.65),
        planner_prior_std_fraction=plan.get("prior_std_fraction", 0.10),
        planner_min_confidence=plan.get("min_confidence", 0.18),
        planner_segment_center_weight=plan.get("segment_center_weight", 0.68),
        planner_temporal_blend=plan.get("temporal_blend", 0.58),
        planner_max_lateral_step_px=plan.get("max_lateral_step_px", 32.0),
        planner_straight_blend=plan.get("straight_blend", 0.72),
        planner_straight_residual_px=plan.get("straight_residual_px", 4.0),
        planner_straight_heading_threshold=plan.get("straight_heading_threshold", 0.08),
        kalman_process_noise=plan.get("kalman_process_noise", 1e-3),
        kalman_measurement_noise=plan.get("kalman_measurement_noise", 1e-1),
        fallback_after_n_misses=plan.get("fallback_after_n_misses", 3),
        min_road_pixels=plan.get("min_road_pixels", 500),
        planner_depth_clearance_weight=plan.get("depth_clearance_weight", 0.35),
        planner_min_clearance_m=plan.get("min_clearance_m", 1.15),
        planner_obstacle_penalty=plan.get("obstacle_penalty", 0.85),
        stanley_gain_k=ctrl.get("stanley_gain_k", 1.5),
        stanley_softening=ctrl.get("stanley_softening", 2.4),
        stanley_heading_gain=ctrl.get("stanley_heading_gain", 0.85),
        steering_ema_alpha=ctrl.get("steering_ema_alpha", 0.45),
        max_steering_delta=ctrl.get("max_steering_delta", 0.3),
        lookahead_base_m=float(ctrl.get("lookahead_base_m", 3.0)),
        lookahead_time_s=float(ctrl.get("lookahead_time_s", 0.6)),
        lookahead_min_m=float(ctrl.get("lookahead_min_m", 2.0)),
        lookahead_max_m=float(ctrl.get("lookahead_max_m", 8.0)),
        cross_track_tolerance_m=float(ctrl.get("cross_track_tolerance_m", 0.3)),
        steer_full_authority_speed_mps=float(ctrl.get("steer_full_authority_speed_mps", 3.0)),
        steer_speed_falloff=float(ctrl.get("steer_speed_falloff", 0.08)),
        wheelbase_m=float(ctrl.get("wheelbase_m", 2.6)),
        max_wheel_angle_deg=float(ctrl.get("max_wheel_angle_deg", 32.0)),
        path_end_margin_m=float(ctrl.get("path_end_margin_m", 1.5)),
        path_end_decel_mps2=float(ctrl.get("path_end_decel_mps2", 2.0)),
        max_lateral_accel_mps2=float(ctrl.get("max_lateral_accel_mps2", 0.5)),
        curve_decel_mps2=float(ctrl.get("curve_decel_mps2", 0.8)),
        control_latency_s=float(ctrl.get("control_latency_s", 0.35)),
        lookahead_curvature_gain=float(ctrl.get("lookahead_curvature_gain", 2.0)),
        curvature_feedforward_gain=float(ctrl.get("curvature_feedforward_gain", 1.0)),
        saturation_start=float(ctrl.get("saturation_start", 0.5)),
        saturation_min_speed_scale=float(ctrl.get("saturation_min_speed_scale", 0.35)),
        steer_cap_full_speed_mps=float(ctrl.get("steer_cap_full_speed_mps", 2.0)),
        steer_cap_high_speed_mps=float(ctrl.get("steer_cap_high_speed_mps", 5.0)),
        steer_cap_at_high_speed=float(ctrl.get("steer_cap_at_high_speed", 0.6)),
        max_target_speed_increase_mps=float(ctrl.get("max_target_speed_increase_mps", 0.1)),
        comfort_brake=float(ctrl.get("comfort_brake", 0.35)),
        min_curvature_span_m=float(ctrl.get("min_curvature_span_m", 2.0)),
        cross_track_soft_deadzone=bool(ctrl.get("cross_track_soft_deadzone", True)),
        latency_pose_prediction=bool(ctrl.get("latency_pose_prediction", True)),
        cross_track_lookahead_gain=float(ctrl.get("cross_track_lookahead_gain", 1.0)),
        cross_track_min_lookahead_m=float(ctrl.get("cross_track_min_lookahead_m", 1.5)),
        cross_track_recovery_m=float(ctrl.get("cross_track_recovery_m", 0.5)),
        edge_margin_m=float(ctrl.get("edge_margin_m", 0.75)),
        edge_centering_gain=float(ctrl.get("edge_centering_gain", 0.7)),
        edge_speed_reduction=float(ctrl.get("edge_speed_reduction", 0.6)),
        drift_speed_gain=float(ctrl.get("drift_speed_gain", 2.0)),
        max_measured_lateral_accel_mps2=float(ctrl.get("max_measured_lateral_accel_mps2", 0.8)),
        target_speed_mph=float(ctrl.get("target_speed_mph", 12.0)),
        speed_limit_mph=float(ctrl.get("speed_limit_mph", 15.0)),
        min_turn_speed_mph=float(ctrl.get("min_turn_speed_mph", 7.0)),
        max_throttle=ctrl.get("max_throttle", 0.45),
        max_brake=ctrl.get("max_brake", 0.8),
        speed_kp=ctrl.get("speed_kp", 0.22),
        clearance_slow_m=ctrl.get("clearance_slow_m", 6.0),
        clearance_stop_m=ctrl.get("clearance_stop_m", 2.5),
        dashboard_colors=_load_dashboard_colors(dashboard.get("colors")),
        ui_headless=bool(ui.get("headless", False)),
        ui_render_every_n=max(1, int(ui.get("render_every_n", 1))),
        ui_debug_view=debug_view,
        ui_timing_overlay=bool(ui.get("timing_overlay", False)),
        ui_display_async=bool(ui.get("display_async", True)),
        ui_display_rate_hz=float(ui.get("display_rate_hz", 20.0)),
        runtime_log_interval_s=float(ui.get("log_interval_s", 5.0)),
    )
