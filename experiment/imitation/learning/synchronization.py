"""Timestamp joins and split rules independent of video and neural-network libraries."""

from __future__ import annotations

import math

import numpy as np

from experiment.imitation.config import ImitationConfig


def aligned_indices(times: np.ndarray, timestamp: int, settings: ImitationConfig):
    """Separate a nearest measured target from a causal vehicle-state input."""
    insertion = int(np.searchsorted(times, timestamp, side="right"))
    if insertion == 0:
        return None
    prior = insertion - 1
    nearest = prior
    if insertion < len(times) and times[insertion] - timestamp < timestamp - times[prior]:
        nearest = insertion
    if abs(int(times[nearest]) - timestamp) > settings.max_alignment_ms * 1e6:
        return None
    if timestamp - int(times[prior]) > settings.max_state_age_ms * 1e6:
        return None
    return prior, nearest


def temporal_split(elapsed: float, duration: float, settings: ImitationConfig) -> str:
    first = duration * settings.temporal_train_fraction
    second = duration * (settings.temporal_train_fraction + settings.temporal_validation_fraction)
    guard = settings.split_guard_s
    if abs(elapsed - first) <= guard or abs(elapsed - second) <= guard:
        return "discard"
    if elapsed < first:
        return "train"
    if elapsed < second:
        return "validation"
    return "test"


def steering_evidence(rows: list[dict], settings: ImitationConfig) -> dict:
    speed = np.array([row["speed_mps"] for row in rows])
    angle = np.array([row["raw_rwa"] for row in rows])
    yaw = np.array([row["yaw_rate"] for row in rows])
    keep = (speed > 1) & np.isfinite(speed) & np.isfinite(angle) & np.isfinite(yaw)
    predicted = speed[keep] * np.tan(angle[keep]) / settings.real_wheelbase_m
    result = {
        "target_kind": "measured_steering_proxy",
        "human_control_confirmed": False,
        "units_status": "dynamics_supported_radians_hypothesis",
        "direction_status": "positive_yaw_hypothesis_not_publisher_verified",
        "wheelbase_m_assumed": settings.real_wheelbase_m,
        "independent_reference_confirmed": False,
        "passed": False,
    }
    if len(predicted) < 3 or np.std(predicted) == 0 or np.std(yaw[keep]) == 0:
        return result
    correlation = float(np.corrcoef(predicted, yaw[keep])[0, 1])
    scale = float(predicted @ yaw[keep] / (predicted @ predicted))
    result.update(correlation=correlation, fitted_scale=scale)
    result["passed"] = (
        math.isfinite(correlation)
        and correlation >= settings.physics_min_correlation
        and settings.physics_min_scale <= scale <= settings.physics_max_scale
    )
    return result
