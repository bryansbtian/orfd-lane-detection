"""ViPlanner-style local planning with predictive fallback."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from scipy.signal import savgol_filter

from offroad_autonomy.types import PathPlan, PipelineConfig, StabilizedResult, VehicleState

logger = logging.getLogger("offroad_autonomy.planning")


class _KalmanTracker:
    """Linear Kalman filter over [lateral_offset, heading, curvature]."""

    DIM_X = 3
    DIM_Z = 2

    def __init__(self, q: float, r: float) -> None:
        self.x = np.zeros(self.DIM_X)
        self.P = np.eye(self.DIM_X) * 100.0

        self.F = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.Q = np.eye(self.DIM_X) * q
        self.R = np.eye(self.DIM_Z) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.DIM_X) - K @ self.H) @ self.P
        return self.x.copy()


@dataclass
class _PlannerScene:
    mask: np.ndarray
    traversability: np.ndarray
    clearance: np.ndarray
    stability_score: float
    lateral_prior_px: float
    heading_prior: float
    ego_heading_rad: float


class _PlannerBackend(Protocol):
    name: str

    def infer(self, scene: _PlannerScene) -> tuple[np.ndarray, float]:
        """Infer a local trajectory from the planner scene."""


class _HeuristicViPlannerBackend:
    """Traversability-driven local planner backend."""

    name = "heuristic"

    def __init__(self, config: PipelineConfig) -> None:
        self._n_samples = config.centerline_samples
        self._horizon_fraction = float(np.clip(config.planner_horizon_fraction, 0.30, 0.95))
        self._clearance_weight = float(np.clip(config.planner_clearance_weight, 0.0, 1.0))
        self._prior_std_fraction = max(0.02, float(config.planner_prior_std_fraction))

    def infer(self, scene: _PlannerScene) -> tuple[np.ndarray, float]:
        """Infer waypoints from dense traversability features."""
        h, w = scene.mask.shape[:2]
        top_y = max(0, int(round(h * (1.0 - self._horizon_fraction))))
        bottom_y = min(h - 1, int(round(h * 0.95)))
        row_indices = np.linspace(bottom_y, top_y, self._n_samples, dtype=int)

        feature_map = (
            (1.0 - self._clearance_weight) * scene.traversability
            + self._clearance_weight * scene.clearance
        )
        cols = np.arange(w, dtype=np.float32)
        prior_x = float(w / 2.0 + scene.lateral_prior_px)
        heading = float(np.clip(scene.heading_prior, -0.75, 0.75))
        sigma = max(12.0, w * self._prior_std_fraction)

        points: list[tuple[float, float]] = []
        row_confidences: list[float] = []

        for idx, y in enumerate(row_indices):
            valid = scene.mask[y, :]
            if int(valid.sum()) < 2:
                continue

            scores = feature_map[y, :] * valid.astype(np.float32)
            if float(scores.max()) <= 1e-6:
                continue

            prior = np.exp(-0.5 * np.square((cols - prior_x) / sigma))
            scores = scores * (0.35 + 0.65 * prior)
            score_sum = float(scores.sum())
            if score_sum <= 1e-6:
                continue

            x = float(np.dot(scores, cols) / score_sum)
            points.append((x, float(y)))
            row_confidences.append(float(scores[int(np.clip(round(x), 0, w - 1))]))

            if len(points) >= 2:
                prev_x, prev_y = points[-2]
                dy = max(prev_y - y, 1.0)
                measured_heading = math.atan2(x - prev_x, dy)
                heading = float(np.clip(0.7 * heading + 0.3 * measured_heading, -0.75, 0.75))

            if idx + 1 < len(row_indices):
                dy_next = max(y - row_indices[idx + 1], 1.0)
                prior_x = float(np.clip(x + math.tan(heading) * dy_next, 0.0, w - 1.0))

        if len(points) < 2:
            return np.empty((0, 2), dtype=np.float32), 0.0

        coverage = len(points) / len(row_indices)
        local_confidence = float(np.mean(row_confidences)) if row_confidences else 0.0
        confidence = float(
            np.clip(
                (0.55 * coverage + 0.45 * local_confidence) * max(scene.stability_score, 0.0),
                0.0,
                1.0,
            )
        )
        return np.array(points, dtype=np.float32), confidence


def _build_backend(config: PipelineConfig) -> _PlannerBackend:
    if config.planner_backend == "heuristic":
        return _HeuristicViPlannerBackend(config)
    raise ValueError(f"Unsupported planner backend: {config.planner_backend}")


class CenterlinePlanner:
    """Local planner over traversability features with predictive fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self._n_samples = config.centerline_samples
        self._min_road_px = config.min_road_pixels
        self._max_misses = config.fallback_after_n_misses
        self._min_confidence = float(np.clip(config.planner_min_confidence, 0.0, 1.0))
        self._horizon_fraction = float(np.clip(config.planner_horizon_fraction, 0.30, 0.95))
        self._smoothing_window = max(3, int(config.planner_smoothing_window))
        self._temporal_blend = float(np.clip(config.planner_temporal_blend, 0.0, 1.0))
        self._max_lateral_step_px = float(max(config.planner_max_lateral_step_px, 0.0))
        self._backend = _build_backend(config)

        self._kf = _KalmanTracker(
            q=config.kalman_process_noise,
            r=config.kalman_measurement_noise,
        )
        self._consecutive_misses = 0
        self._prev_centerline: np.ndarray | None = None
        logger.info("Planning backend: %s", self._backend.name)

    def plan(
        self,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState | None = None,
    ) -> PathPlan:
        """Compute a local trajectory from the traversability scene."""
        mask = stabilized.mask
        h, w = mask.shape[:2]
        road_px = int(mask.sum())
        prior_state = self._kf.predict()

        if road_px < self._min_road_px:
            return self._fallback(h, w, prior_state)

        scene = self._build_scene(mask, stabilized.stability_score, prior_state, vehicle_state)
        waypoints, confidence = self._backend.infer(scene)
        if len(waypoints) < 2 or confidence < self._min_confidence:
            return self._fallback(h, w, prior_state)

        centerline = self._postprocess_trajectory(waypoints, h, w)
        if len(centerline) < 2:
            return self._fallback(h, w, prior_state)
        centerline = self._stabilize_trajectory(centerline, w)

        lateral_offset = centerline[-1, 0] - w / 2.0
        heading = self._estimate_heading(centerline)
        road_width = self._estimate_road_width(mask, centerline)

        z = np.array([lateral_offset, heading])
        state = self._kf.update(z)
        self._consecutive_misses = 0
        self._prev_centerline = centerline.copy()
        smoothed_heading = float(np.clip(0.9 * heading + 0.1 * state[1], -0.55, 0.55))

        return PathPlan(
            centerline=centerline,
            heading_rad=smoothed_heading,
            curvature=self._estimate_curvature(centerline),
            road_width_px=road_width,
            kalman_active=False,
        )

    def _build_scene(
        self,
        mask: np.ndarray,
        stability_score: float,
        prior_state: np.ndarray,
        vehicle_state: VehicleState | None,
    ) -> _PlannerScene:
        mask_u8 = mask.astype(np.uint8)
        traversability = cv2.GaussianBlur(mask_u8.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.4)
        if float(traversability.max()) > 0.0:
            traversability /= float(traversability.max())

        clearance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if float(clearance.max()) > 0.0:
            clearance /= float(clearance.max())

        return _PlannerScene(
            mask=mask.astype(bool),
            traversability=traversability,
            clearance=clearance,
            stability_score=float(stability_score),
            lateral_prior_px=float(prior_state[0]),
            heading_prior=float(prior_state[1]),
            ego_heading_rad=0.0 if vehicle_state is None else float(vehicle_state.heading_rad),
        )

    def _postprocess_trajectory(self, waypoints: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resample and smooth backend waypoints into a controller path."""
        if len(waypoints) < 2:
            return np.empty((0, 2), dtype=np.float32)

        points = np.asarray(waypoints, dtype=np.float32)
        order = np.argsort(points[:, 1])
        points = points[order]

        y_vals, first_indices = np.unique(points[:, 1], return_index=True)
        x_vals = points[first_indices, 0]
        if len(y_vals) < 2:
            return np.empty((0, 2), dtype=np.float32)

        top_y = max(0.0, float(y_vals[0]))
        bottom_y = min(float(h - 1), float(y_vals[-1]))
        target_y = np.linspace(top_y, bottom_y, self._n_samples, dtype=np.float32)
        target_x = np.interp(target_y, y_vals, x_vals)
        target_x = self._smooth_lateral_positions(target_x)
        target_x = np.clip(target_x, 0.0, float(w - 1))

        return np.stack([target_x, target_y], axis=1).astype(np.float32)

    def _smooth_lateral_positions(self, x_vals: np.ndarray) -> np.ndarray:
        """Apply stable smoothing to lateral waypoint positions."""
        count = len(x_vals)
        if count < 3:
            return x_vals

        window = min(self._smoothing_window, count)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return x_vals

        polyorder = 2 if window >= 5 else 1
        return savgol_filter(x_vals, window_length=window, polyorder=polyorder, mode="interp")

    def _stabilize_trajectory(self, centerline: np.ndarray, w: int) -> np.ndarray:
        """Blend the current trajectory with the previous one to reduce zigzag."""
        if self._prev_centerline is None or len(self._prev_centerline) < 2:
            return centerline

        prev = self._prev_centerline
        prev_x = np.interp(centerline[:, 1], prev[:, 1], prev[:, 0])
        blended_x = self._temporal_blend * centerline[:, 0] + (1.0 - self._temporal_blend) * prev_x

        if self._max_lateral_step_px > 0.0:
            delta = np.clip(
                blended_x - prev_x,
                -self._max_lateral_step_px,
                self._max_lateral_step_px,
            )
            blended_x = prev_x + delta

        stabilized = centerline.copy()
        stabilized[:, 0] = np.clip(blended_x, 0.0, float(w - 1))
        return stabilized

    def _fallback(self, h: int, w: int, state: np.ndarray) -> PathPlan:
        """Use the predicted planner state when the scene is unreliable."""
        self._consecutive_misses += 1

        if self._consecutive_misses > self._max_misses:
            logger.warning(
                "No valid planner trajectory for %d frames - predictive fallback active",
                self._consecutive_misses,
            )

        top_y = max(0.0, h * (1.0 - self._horizon_fraction))
        bottom_y = min(float(h - 1), h * 0.95)
        y_vals = np.linspace(top_y, bottom_y, max(3, self._n_samples), dtype=np.float32)
        bottom_x = w / 2.0 + float(state[0])
        heading = float(np.clip(state[1], -0.75, 0.75))
        x_vals = bottom_x + np.tan(heading) * (bottom_y - y_vals)
        x_vals = np.clip(x_vals, 0.0, float(w - 1))
        fallback_pts = np.stack([x_vals, y_vals], axis=1).astype(np.float32)

        return PathPlan(
            centerline=fallback_pts,
            heading_rad=float(np.clip(state[1], -0.70, 0.70)),
            curvature=float(state[2]),
            road_width_px=0.0,
            kalman_active=True,
        )

    @staticmethod
    def _estimate_road_width(mask: np.ndarray, centerline: np.ndarray) -> float:
        """Estimate local corridor width around the trajectory."""
        if len(centerline) == 0:
            return 0.0

        h, w = mask.shape[:2]
        widths: list[float] = []
        sample_indices = np.linspace(0, len(centerline) - 1, min(len(centerline), 8), dtype=int)

        for idx in sample_indices:
            x, y = centerline[idx]
            yi = int(np.clip(round(y), 0, h - 1))
            xi = int(np.clip(round(x), 0, w - 1))
            row = mask[yi, :]
            if not row[xi]:
                continue

            left = xi
            while left > 0 and row[left - 1]:
                left -= 1

            right = xi
            while right < w - 1 and row[right + 1]:
                right += 1

            widths.append(float(right - left))

        return float(np.mean(widths)) if widths else 0.0

    @staticmethod
    def _estimate_heading(centerline: np.ndarray) -> float:
        """Compute heading from the lower section of the trajectory."""
        n = len(centerline)
        if n < 2:
            return 0.0

        lookback = max(1, n // 3)
        pt_near = centerline[-1]
        pt_far = centerline[max(0, n - 1 - lookback)]

        dx = pt_far[0] - pt_near[0]
        dy = pt_near[1] - pt_far[1]
        if dy < 1e-6:
            return 0.0

        return float(math.atan2(dx, dy))

    @staticmethod
    def _estimate_curvature(centerline: np.ndarray) -> float:
        """Estimate mean geometric curvature over the planned path."""
        if len(centerline) < 3:
            return 0.0

        dx = np.gradient(centerline[:, 0])
        dy = np.gradient(centerline[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = np.power(dx * dx + dy * dy, 1.5)
        valid = denom > 1e-6
        if not np.any(valid):
            return 0.0

        curvature = np.abs(dx[valid] * ddy[valid] - dy[valid] * ddx[valid]) / denom[valid]
        return float(np.mean(curvature))

    def reset(self) -> None:
        """Clear Kalman state (e.g. on map reload)."""
        self._kf = _KalmanTracker(
            q=self._kf.Q[0, 0],
            r=self._kf.R[0, 0],
        )
        self._consecutive_misses = 0
        self._prev_centerline = None
