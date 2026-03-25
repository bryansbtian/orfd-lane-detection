"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml

from offroad_autonomy.types import DEFAULT_PERCEPTION_PROMPTS, PipelineConfig


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
        kalman_process_noise=plan.get("kalman_process_noise", 1e-3),
        kalman_measurement_noise=plan.get("kalman_measurement_noise", 1e-1),
        fallback_after_n_misses=plan.get("fallback_after_n_misses", 3),
        min_road_pixels=plan.get("min_road_pixels", 500),
        stanley_gain_k=ctrl.get("stanley_gain_k", 2.5),
        stanley_softening=ctrl.get("stanley_softening", 1.0),
        target_speed_mps=ctrl.get("target_speed_mps", 5.0),
        max_throttle=ctrl.get("max_throttle", 0.6),
        max_brake=ctrl.get("max_brake", 0.8),
        speed_kp=ctrl.get("speed_kp", 0.3),
    )
