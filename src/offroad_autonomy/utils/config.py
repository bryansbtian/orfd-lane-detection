"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from offroad_autonomy.types import (
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_PERCEPTION_PROMPTS,
    PipelineConfig,
)


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


def _load_speed_setting_mph(
    raw: dict,
    mph_key: str,
    mps_key: str | None,
    default: float,
) -> float:
    if mph_key in raw:
        try:
            return float(raw[mph_key])
        except (TypeError, ValueError):
            return default

    if mps_key is not None and mps_key in raw:
        try:
            return float(raw[mps_key]) * 2.2369362920544
        except (TypeError, ValueError):
            return default

    return default


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML configuration file into a ``PipelineConfig``."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    bng = raw.get("beamng", {})
    cam = bng.get("camera", {})
    perc = raw.get("perception", {})
    pre = raw.get("preprocessing", {})
    post = raw.get("postprocessing", {})
    plan = raw.get("planning", {})
    ctrl = raw.get("control", {})
    viz = raw.get("visualization", {})
    dashboard = viz.get("dashboard", {})

    return PipelineConfig(
        beamng_home=bng.get("home", ""),
        beamng_host=bng.get("host", "localhost"),
        beamng_port=bng.get("port", 64256),
        beamng_map=bng.get("map", "automation_test_track"),
        beamng_vehicle=bng.get("vehicle", "pickup"),
        beamng_spawn_index=bng.get("spawn_index", 0),
        camera_width=cam.get("width", 1280),
        camera_height=cam.get("height", 720),
        camera_fov=cam.get("fov", 120.0),
        camera_pos=cam.get("pos", [0, -2.5, 0.8]),
        camera_dir=cam.get("dir", [0, -1, -0.1]),
        map_spawns=bng.get("maps", {}),
        model_weights=perc.get("model_weights", "models/yoloe-26x-seg.pt"),
        confidence_threshold=perc.get("confidence_threshold", 0.25),
        perception_input_size=perc.get("input_size", 640),
        perception_prompts=perc.get("prompts", DEFAULT_PERCEPTION_PROMPTS.copy()),
        preprocess_width=pre.get("target_width", 640),
        preprocess_height=pre.get("target_height", 360),
        enable_clahe=pre.get("enable_clahe", False),
        clahe_clip_limit=pre.get("clahe_clip_limit", 2.0),
        clahe_grid_size=pre.get("clahe_grid_size", 8),
        ema_alpha=post.get("ema_alpha", 0.7),
        min_mask_area_fraction=post.get("min_mask_area_fraction", 0.001),
        morphology_kernel_size=post.get("morphology_kernel_size", 5),
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
        stanley_gain_k=ctrl.get("stanley_gain_k", 1.15),
        stanley_softening=ctrl.get("stanley_softening", 2.4),
        stanley_heading_gain=ctrl.get("stanley_heading_gain", 0.75),
        stanley_lookahead_fraction=ctrl.get("stanley_lookahead_fraction", 0.35),
        stanley_near_path_weight=ctrl.get("stanley_near_path_weight", 0.35),
        steering_ema_alpha=ctrl.get("steering_ema_alpha", 0.28),
        max_steering_delta=ctrl.get("max_steering_delta", 0.10),
        steering_deadband=ctrl.get("steering_deadband", 0.03),
        target_speed_mph=_load_speed_setting_mph(
            ctrl,
            mph_key="target_speed_mph",
            mps_key="target_speed_mps",
            default=12.0,
        ),
        speed_limit_mph=_load_speed_setting_mph(
            ctrl,
            mph_key="speed_limit_mph",
            mps_key=None,
            default=15.0,
        ),
        min_turn_speed_mph=_load_speed_setting_mph(
            ctrl,
            mph_key="min_turn_speed_mph",
            mps_key=None,
            default=7.0,
        ),
        curve_speed_gain=ctrl.get("curve_speed_gain", 475.0),
        heading_speed_gain=ctrl.get("heading_speed_gain", 1.1),
        max_throttle=ctrl.get("max_throttle", 0.45),
        max_brake=ctrl.get("max_brake", 0.8),
        speed_kp=ctrl.get("speed_kp", 0.22),
        dashboard_colors=_load_dashboard_colors(dashboard.get("colors")),
    )
