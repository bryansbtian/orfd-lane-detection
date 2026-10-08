"""YAML -> ``PipelineConfig``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.runtime.video_recorder import X264_PRESETS
from offroad_autonomy.types import (
    CAMERA_TRANSPORTS,
    DEBUG_VIEWS,
    DEFAULT_CAMERA,
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_ORBIT_CAMERA,
    DEFAULT_PERCEPTION_PROMPTS,
    GMSL2_CAPTURE_SENSOR,
    CameraSensor,
    CameraSpec,
    DashboardThresholds,
    EgoMaskSpec,
    PipelineConfig,
    mount_pose,
)
from offroad_autonomy.utils import environment

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


def _load_dashboard_thresholds(dashboard: dict, gate: dict, safety: dict) -> DashboardThresholds:
    """The two floors come from the gate and the safe stop, not from the
    dashboard section, so the bar ticks always match what triggers them."""
    defaults = DashboardThresholds()
    thresholds = DashboardThresholds(
        target_fps=float(dashboard.get("target_fps", defaults.target_fps)),
        fps_warn_fraction=float(dashboard.get("fps_warn_fraction", defaults.fps_warn_fraction)),
        latency_budget_ms=float(dashboard.get("latency_budget_ms", defaults.latency_budget_ms)),
        confidence_floor=float(gate.get("confidence_threshold", defaults.confidence_floor)),
        confidence_good=float(dashboard.get("confidence_good", defaults.confidence_good)),
        road_floor=float(safety.get("min_road_fraction", defaults.road_floor)),
        fps_bar_scale=float(dashboard.get("fps_bar_scale", defaults.fps_bar_scale)),
        latency_bar_scale=float(dashboard.get("latency_bar_scale", defaults.latency_bar_scale)),
        road_bar_full_scale=float(
            dashboard.get("road_bar_full_scale", defaults.road_bar_full_scale)
        ),
    )
    if thresholds.target_fps <= 0.0 or thresholds.latency_budget_ms <= 0.0:
        raise ValueError("visualization.dashboard target_fps and latency_budget_ms must be > 0")
    if not 0.0 < thresholds.fps_warn_fraction < 1.0:
        raise ValueError("visualization.dashboard.fps_warn_fraction must be between 0 and 1")
    if thresholds.confidence_good <= thresholds.confidence_floor:
        raise ValueError(
            "visualization.dashboard.confidence_good must be above planning.gate.confidence_threshold"
        )
    # A full scale at or below the target would push the target tick off the bar.
    if min(thresholds.fps_bar_scale, thresholds.latency_bar_scale) <= 1.0:
        raise ValueError("visualization.dashboard bar scales must be above 1")
    if thresholds.road_bar_full_scale <= thresholds.road_floor:
        raise ValueError(
            "visualization.dashboard.road_bar_full_scale must exceed the safe stop floor"
        )
    return thresholds


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
        raise ValueError(
            "beamng.camera.ego_mask is enabled but its polygon has fewer than 3 points"
        )
    return EgoMaskSpec(enabled=enabled, polygon=polygon, margin_px=margin)


def _load_camera(raw: dict) -> CameraSpec:
    default = DEFAULT_CAMERA
    direction, up = mount_pose(
        float(raw.get("pitch_deg", -8.0)),
        float(raw.get("roll_deg", 0.0)),
        float(raw.get("yaw_deg", 0.0)),
    )
    pos = (
        float(raw.get("lateral_offset_m", default.pos[0])),
        float(raw.get("forward_offset_m", default.pos[1])),
        float(raw.get("height_m", default.pos[2])),
    )
    return CameraSpec(
        name=str(raw.get("name", default.name)),
        pos=_parse_vec3(raw.get("pos"), pos),
        dir=_parse_vec3(raw.get("dir"), direction),
        up=_parse_vec3(raw.get("up"), up),
        sensor=_load_sensor(raw.get("sensor")),
        ego_mask=_load_ego_mask(raw.get("ego_mask"), default.ego_mask),
    )


def _load_orbit_camera(raw: dict) -> CameraSpec:
    """A bad pose fails at load time, before a run is recorded with the
    vehicle out of frame."""
    default = DEFAULT_ORBIT_CAMERA
    raw_pos = raw.get("pos", list(default.pos))
    if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 3:
        raise ValueError("presentation.orbit_camera.pos must be a list of 3 numbers")
    pos = tuple(float(value) for value in raw_pos)
    pitch = float(raw.get("pitch_deg", -12.0))
    if not -90.0 < pitch < 90.0:
        raise ValueError("presentation.orbit_camera.pitch_deg must be between -90 and 90")
    sensor = _load_sensor(raw.get("sensor"), default.sensor)
    if sensor.width < 1 or sensor.height < 1:
        raise ValueError("presentation.orbit_camera.sensor width and height must be positive")
    if not 0.0 < sensor.fov_x_deg < 180.0:
        raise ValueError("presentation.orbit_camera.sensor.fov_h must be between 0 and 180")
    if sensor.target_fps <= 0.0:
        raise ValueError("presentation.orbit_camera.sensor.target_fps must be > 0")
    direction, up = mount_pose(pitch)
    return CameraSpec(name=default.name, pos=pos, dir=direction, up=up, sensor=sensor)


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
    if mode not in ("baseline", "advanced", "grid"):
        raise ValueError(f"planning.mode must be 'baseline', 'advanced' or 'grid', got {mode!r}")
    return mode


def _camera_transport(raw: object) -> str:
    transport = str(raw).lower()
    if transport not in CAMERA_TRANSPORTS:
        raise ValueError(
            f"beamng.camera_transport must be one of {CAMERA_TRANSPORTS}, got {transport!r}"
        )
    return transport


def _load_recording(raw: dict) -> dict:
    fps = float(raw.get("fps", 20.0))
    crf = int(raw.get("crf", 23))
    preset = str(raw.get("preset", "veryfast")).lower()
    queue_frames = int(raw.get("queue_frames", 8))
    if fps <= 0.0:
        raise ValueError("recording.fps must be > 0")
    if not 0 <= crf <= 51:
        raise ValueError("recording.crf must be between 0 and 51")
    if preset not in X264_PRESETS:
        raise ValueError(f"recording.preset must be one of {X264_PRESETS}, got {preset!r}")
    if queue_frames < 1:
        raise ValueError("recording.queue_frames must be at least 1")
    return {
        "recording_fps": fps,
        "recording_crf": crf,
        "recording_preset": preset,
        "recording_queue_frames": queue_frames,
    }


def _load_grid(raw: dict) -> GridPlannerConfig:
    try:
        return GridPlannerConfig(**raw)
    except TypeError as exc:
        # A misspelled key would otherwise surface as a bare dataclass error.
        raise ValueError(f"planning.grid: {exc}") from exc


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


def _resolve_platform_settings(bng: dict, ui: dict) -> tuple[dict, dict]:
    """A fully explicit config never inspects the machine, so deployments
    that pin every value behave the same everywhere."""
    if not environment.needs_detection(bng, ui):
        return bng, ui
    facts = environment.detect_platform()
    if (
        facts.os_name == "windows"
        and bng.get("host") in (environment.AUTO, *environment.LOCAL_HOSTS)
        and bng.get("launch") == environment.AUTO
        and not bng.get("home")
        and "BEAMNG_HOME" not in os.environ
    ):
        saved_home = environment.saved_windows_beamng_home()
        if saved_home:
            bng = dict(bng, home=saved_home)
            logger.info("Using saved Windows BEAMNG_HOME: %s", saved_home)
    bng, ui = environment.resolve_auto(bng, ui, facts)
    platform = facts.os_name
    if facts.is_wsl:
        platform = f"wsl2 ({facts.wsl_networking})"
    action = "attaching to"
    if bng.get("launch"):
        action = "launching or attaching to"
    host = bng.get("host") or "<unset, set BEAMNG_HOST>"
    logger.info(
        "Platform %s: %s BeamNG at %s:%s over %s",
        platform,
        action,
        host,
        bng.get("port", 64256),
        bng.get("camera_transport"),
    )
    return bng, ui


def load_config(path: str | Path) -> PipelineConfig:
    raw = _read_yaml(Path(path))

    bng, ui = _resolve_platform_settings(
        _apply_env_overrides(_section(raw, "beamng")), _section(raw, "ui")
    )
    if "cameras" in bng:
        raise ValueError(
            "Replace beamng.cameras with the single beamng.camera dashcam configuration"
        )
    camera = _load_camera(_section(bng, "camera"))
    perc = _section(raw, "perception")
    safety = _section(raw, "safety")
    pre = _section(raw, "preprocessing")
    post = _section(raw, "postprocessing")
    plan = _section(raw, "planning")
    gate = _section(plan, "gate")
    ctrl = _section(raw, "control")
    if "controller" in ctrl:
        raise ValueError("Remove control.controller: Stanley is the only controller")
    dashboard = _section(_section(raw, "visualization"), "dashboard")

    if "segmentation_mode" in perc or "stitching" in raw:
        raise ValueError("Remove segmentation_mode and stitching: the dashcam is the only input")
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
        camera=camera,
        orbit_camera=_load_orbit_camera(_section(_section(raw, "presentation"), "orbit_camera")),
        map_spawns=bng.get("maps", {}),
        model_weights=perc.get("model_weights", "models/yoloe-26x-seg.pt"),
        confidence_threshold=perc.get("segmentation_threshold", 0.25),
        perception_input_size=perc.get("input_size", 640),
        perception_prompts=perc.get("prompts", DEFAULT_PERCEPTION_PROMPTS.copy()),
        preprocess_width=pre.get("target_width", 720),
        preprocess_height=pre.get("target_height", 465),
        enable_clahe=pre.get("enable_clahe", False),
        clahe_clip_limit=pre.get("clahe_clip_limit", 2.0),
        clahe_grid_size=pre.get("clahe_grid_size", 8),
        safety_min_road_fraction=safety.get("min_road_fraction", 0.015),
        safety_no_road_time_s=safety.get("no_road_time_s", 2.0),
        ema_alpha=post.get("ema_alpha", 0.7),
        min_mask_area_fraction=post.get("min_mask_area_fraction", 0.001),
        morphology_kernel_size=post.get("morphology_kernel_size", 5),
        enable_morphology=bool(post.get("enable_morphology", True)),
        enable_ema=bool(post.get("enable_ema", True)),
        planner_mode=_planner_mode(plan.get("mode", "baseline")),
        planner_roi_height=float(plan.get("roi_height", 0.50)),
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
        grid=_load_grid(_section(plan, "grid")),
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
        camera_ahead_of_rear_axle_m=float(ctrl.get("camera_ahead_of_rear_axle_m", 1.4)),
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
        vehicle_half_width_m=float(ctrl.get("vehicle_half_width_m", 0.95)),
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
        dashboard_thresholds=_load_dashboard_thresholds(dashboard, gate, safety),
        ui_headless=bool(ui.get("headless", False)),
        ui_render_every_n=max(1, int(ui.get("render_every_n", 1))),
        ui_debug_view=debug_view,
        ui_timing_overlay=bool(ui.get("timing_overlay", False)),
        ui_display_async=bool(ui.get("display_async", True)),
        ui_display_rate_hz=float(ui.get("display_rate_hz", 20.0)),
        runtime_log_interval_s=float(ui.get("log_interval_s", 5.0)),
        **_load_recording(_section(raw, "recording")),
    )
