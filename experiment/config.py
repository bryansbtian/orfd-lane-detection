"""Experiment settings layered on the default ``PipelineConfig``.

An experiment config extends ``configs/default.yaml`` and adds an
``experiment:`` section that picks the planner and controller, plus the
sections those need. The default loader refuses such a file, so an
experiment can only run through ``python -m experiment.run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

from experiment.grid.config import GridPlannerConfig
from experiment.imitation.config import ImitationConfig
from experiment.predictive.config import PredictiveConfig
from experiment.rl.config import RLConfig
from offroad_autonomy.types import PipelineConfig
from offroad_autonomy.utils.config import config_from_dict, read_config_dict

PLANNERS = ("baseline", "advanced", "grid")
CONTROLLERS = ("stanley", "mppi", "mpc", "rl", "imitation")


@dataclass
class ExperimentConfig(PipelineConfig):
    planner_mode: str = "baseline"
    control_mode: str = "stanley"

    # Advanced planner only (experiment.planner: advanced).
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

    grid: GridPlannerConfig = field(default_factory=GridPlannerConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _choice(raw: dict, key: str, choices: tuple[str, ...], default: str) -> str:
    value = str(raw.get(key, default)).lower()
    if value not in choices:
        raise ValueError(f"experiment.{key} must be one of {choices}, got {value!r}")
    return value


def _settings(cls, raw: dict, name: str):
    try:
        return cls(**raw)
    except TypeError as exc:
        # A misspelled key would otherwise surface as a bare dataclass error.
        raise ValueError(f"{name}: {exc}") from exc


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = read_config_dict(path)
    selection = _section(raw, "experiment")
    plan = _section(raw, "planning")
    ctrl = _section(raw, "control")
    base = config_from_dict(raw)
    shared = {f.name: getattr(base, f.name) for f in fields(PipelineConfig)}
    return ExperimentConfig(
        **shared,
        planner_mode=_choice(selection, "planner", PLANNERS, "baseline"),
        control_mode=_choice(selection, "controller", CONTROLLERS, "stanley"),
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
        grid=_settings(GridPlannerConfig, _section(plan, "grid"), "planning.grid"),
        predictive=_settings(PredictiveConfig, _section(ctrl, "predictive"), "control.predictive"),
        rl=_settings(RLConfig, _section(ctrl, "rl"), "control.rl"),
        imitation=_settings(ImitationConfig, _section(raw, "imitation"), "imitation"),
    )
