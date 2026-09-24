"""Planning entry point: the perception gate, then the baseline or the
advanced (ViPlanner-style) planner."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
from scipy.signal import savgol_filter

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.baseline_planner import BaselinePlanner
from offroad_autonomy.planning.perception_gate import GateDecision, PerceptionGate
from offroad_autonomy.types import (
    PathPlan,
    PipelineConfig,
    StabilizedResult,
    TerrainAnalysis,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.planning")


class _KalmanTracker:
    """Linear Kalman filter over [lateral_offset, heading, curvature]."""

    DIM_X = 3
    DIM_Z = 2

    def __init__(self, q: float, r: float) -> None:
        self.x = np.zeros(self.DIM_X)
        self.P = np.eye(self.DIM_X) * 100.0

        self.F = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
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
    clearance_m: np.ndarray | None = None
    min_clearance_m: float = float("inf")


class _PlannerBackend(Protocol):
    name: str

    def infer(self, scene: _PlannerScene) -> tuple[np.ndarray, float]:
        """Infer a local trajectory from the planner scene."""


class _HeuristicViPlannerBackend:
    name = "heuristic"

    def __init__(self, config: PipelineConfig) -> None:
        self._n_samples = config.centerline_samples
        self._horizon_fraction = float(np.clip(config.planner_horizon_fraction, 0.30, 0.95))
        self._clearance_weight = float(np.clip(config.planner_clearance_weight, 0.0, 1.0))
        self._segment_center_weight = float(np.clip(config.planner_segment_center_weight, 0.0, 1.0))
        self._prior_std_fraction = max(0.02, float(config.planner_prior_std_fraction))

    def infer(self, scene: _PlannerScene) -> tuple[np.ndarray, float]:
        h, w = scene.mask.shape[:2]
        top_y = max(0, int(round(h * (1.0 - self._horizon_fraction))))
        bottom_y = min(h - 1, int(round(h * 0.95)))
        row_indices = np.linspace(bottom_y, top_y, self._n_samples, dtype=int)

        feature_map = (
            1.0 - self._clearance_weight
        ) * scene.traversability + self._clearance_weight * scene.clearance
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

            segment = self._select_segment(valid, scores, prior_x, sigma)
            if segment is None:
                continue

            start, end = segment
            segment_cols = cols[start : end + 1]
            segment_scores = scores[start : end + 1]
            prior = np.exp(-0.5 * np.square((segment_cols - prior_x) / sigma))
            segment_scores = segment_scores * (0.35 + 0.65 * prior)
            score_sum = float(segment_scores.sum())
            if score_sum <= 1e-6:
                continue

            weighted_x = float(np.dot(segment_scores, segment_cols) / score_sum)
            segment_center = 0.5 * (start + end)
            x = float(
                self._segment_center_weight * segment_center
                + (1.0 - self._segment_center_weight) * weighted_x
            )
            points.append((x, float(y)))
            row_confidences.append(float(segment_scores.mean()))

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
        local_confidence = 0.0
        if row_confidences:
            local_confidence = float(np.mean(row_confidences))
        confidence = float(
            np.clip(
                (0.55 * coverage + 0.45 * local_confidence) * max(scene.stability_score, 0.0),
                0.0,
                1.0,
            )
        )
        return np.array(points, dtype=np.float32), confidence

    @staticmethod
    def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return []

        splits = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, splits)
        return [(int(group[0]), int(group[-1])) for group in groups if len(group) > 0]

    def _select_segment(
        self,
        valid: np.ndarray,
        scores: np.ndarray,
        prior_x: float,
        sigma: float,
    ) -> tuple[int, int] | None:
        segments = self._segments(valid)
        if not segments:
            return None

        width_scale = max(float(len(valid)), 1.0)
        min_width_px = max(4, int(width_scale * 0.05))
        candidates = [(s, e) for s, e in segments if (e - s + 1) >= min_width_px]
        if not candidates:
            candidates = segments

        best_segment = None
        best_value = -math.inf

        for start, end in candidates:
            segment_scores = scores[start : end + 1]
            if len(segment_scores) == 0:
                continue

            center = 0.5 * (start + end)
            prior = math.exp(-0.5 * math.pow((center - prior_x) / sigma, 2.0))
            mean_score = float(segment_scores.mean())
            peak_score = float(segment_scores.max())
            width_score = (end - start + 1) / width_scale
            value = 0.36 * prior + 0.26 * mean_score + 0.13 * peak_score + 0.25 * width_score

            if value > best_value:
                best_value = value
                best_segment = (start, end)

        return best_segment


def _build_backend(config: PipelineConfig) -> _PlannerBackend:
    if config.planner_backend == "heuristic":
        return _HeuristicViPlannerBackend(config)
    raise ValueError(f"Unsupported planner backend: {config.planner_backend}")


class CenterlinePlanner:
    def __init__(self, config: PipelineConfig, camera: CameraModel | None = None) -> None:
        self._n_samples = config.centerline_samples
        self._min_road_px = config.min_road_pixels
        self._max_misses = config.fallback_after_n_misses
        self._min_confidence = float(np.clip(config.planner_min_confidence, 0.0, 1.0))
        self._horizon_fraction = float(np.clip(config.planner_horizon_fraction, 0.30, 0.95))
        self._smoothing_window = max(3, int(config.planner_smoothing_window))
        self._temporal_blend = float(np.clip(config.planner_temporal_blend, 0.0, 1.0))
        self._max_lateral_step_px = float(max(config.planner_max_lateral_step_px, 0.0))
        self._straight_blend = float(np.clip(config.planner_straight_blend, 0.0, 1.0))
        self._straight_residual_px = float(max(config.planner_straight_residual_px, 0.0))
        self._straight_heading_threshold = float(
            max(config.planner_straight_heading_threshold, 0.0)
        )
        self._depth_clearance_weight = float(
            np.clip(config.planner_depth_clearance_weight, 0.0, 1.0)
        )
        self._min_clearance_m = float(max(config.planner_min_clearance_m, 0.0))
        self._obstacle_penalty = float(np.clip(config.planner_obstacle_penalty, 0.0, 1.0))
        self._vehicle_half_width_m = float(max(config.vehicle_half_width_m, 1e-3))
        self._backend = _build_backend(config)

        self._kf = _KalmanTracker(
            q=config.kalman_process_noise,
            r=config.kalman_measurement_noise,
        )
        self._consecutive_misses = 0
        self._prev_centerline: np.ndarray | None = None

        self._mode = config.planner_mode
        self._gate = PerceptionGate(config)
        self._baseline = BaselinePlanner(config, camera=camera)
        self._hold_frames = max(0, int(config.gate_hold_frames))
        self._hold_speed_scale = float(np.clip(config.gate_hold_speed_scale, 0.0, 1.0))
        self._last_good: PathPlan | None = None
        self._held = 0
        logger.info("Planning mode: %s (advanced backend %s)", self._mode, self._backend.name)

    def plan(
        self,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState | None = None,
        terrain: TerrainAnalysis | None = None,
    ) -> PathPlan:
        """A frame that fails the gate never produces a new path: driving a
        path no perception supports is how the vehicle used to leave the
        trail. The last good path is held briefly at reduced speed instead.
        """
        decision = self._gate.evaluate(stabilized)
        if not decision.ok:
            return self._hold(decision)

        if self._mode == "baseline":
            centerline = self._baseline.centerline(decision.component, decision.roi_top)
            if len(centerline) < 2:
                decision.reason = "centerline too short"
                return self._hold(decision)
            # No clearance term: the baseline must not be slowed or steered
            # by depth until perception is verified on its own.
            plan = PathPlan(
                centerline=centerline,
                heading_rad=self._estimate_heading(centerline),
                curvature=self._estimate_curvature(centerline),
                road_width_px=self._estimate_road_width(decision.component, centerline),
            )
        else:
            plan = self._plan_advanced(stabilized, vehicle_state, terrain)
            if plan.kalman_active:
                # A predicted path is not a perceived one; never drive it at
                # full speed or remember it as the last good path.
                decision.reason = "advanced planner found no path"
                return self._hold(decision)

        plan.planner_mask = decision.component
        plan.roi_top = decision.roi_top
        self._last_good = plan
        self._held = 0
        return plan

    def _hold(self, decision: GateDecision) -> PathPlan:
        self._held += 1
        if self._held == self._hold_frames + 1:
            logger.warning("Perception gate: %s - no valid path, braking", decision.reason)

        if self._last_good is not None and self._held <= self._hold_frames:
            return replace(
                self._last_good,
                fallback_active=True,
                fallback_reason=(
                    f"{decision.reason} - holding path {self._held}/{self._hold_frames}"
                ),
                speed_scale=self._hold_speed_scale,
                planner_mask=decision.component,
                roi_top=decision.roi_top,
            )
        # Out of held path: forget it, so the next road seen is taken as-is
        # rather than blended toward a stale one.
        self._baseline.reset()
        return PathPlan(
            centerline=np.empty((0, 2), dtype=np.float32),
            fallback_active=True,
            fallback_reason=f"{decision.reason} - stopping",
            speed_scale=0.0,
            planner_mask=decision.component,
            roi_top=decision.roi_top,
        )

    def _plan_advanced(
        self,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState | None = None,
        terrain: TerrainAnalysis | None = None,
    ) -> PathPlan:
        """With ``terrain`` the scene gains a metric clearance channel, so the
        planner prefers gaps the vehicle actually fits through rather than the
        widest patch of road-coloured pixels."""
        mask = stabilized.mask
        h, w = mask.shape[:2]
        road_px = int(mask.sum())
        prior_state = self._kf.predict()

        min_clearance = float("inf")
        if terrain is not None:
            min_clearance = float(terrain.min_forward_clearance_m)

        if road_px < self._min_road_px:
            return self._fallback(h, w, prior_state, min_clearance)

        scene = self._build_scene(
            mask,
            stabilized.stability_score,
            prior_state,
            vehicle_state,
            stabilized.traversability,
            terrain,
        )
        waypoints, confidence = self._backend.infer(scene)
        if len(waypoints) < 2 or confidence < self._min_confidence:
            return self._fallback(h, w, prior_state, min_clearance)

        centerline = self._postprocess_trajectory(waypoints, h, w)
        if len(centerline) < 2:
            return self._fallback(h, w, prior_state, min_clearance)
        centerline = self._stabilize_trajectory(centerline, w)
        centerline = self._straighten_trajectory(centerline, w)

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
            min_clearance_m=self._path_clearance(centerline, scene, min_clearance),
        )

    def _build_scene(
        self,
        mask: np.ndarray,
        stability_score: float,
        prior_state: np.ndarray,
        vehicle_state: VehicleState | None,
        fused_traversability: np.ndarray | None = None,
        terrain: TerrainAnalysis | None = None,
    ) -> _PlannerScene:
        mask_u8 = mask.astype(np.uint8)

        if fused_traversability is not None and fused_traversability.shape == mask.shape:
            # The fused field already encodes geometry; blur only to soften
            # per-pixel stereo noise, not to reshape the corridor.
            traversability = cv2.GaussianBlur(
                fused_traversability.astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=1.6
            )
        else:
            traversability = cv2.GaussianBlur(
                mask_u8.astype(np.float32), (0, 0), sigmaX=2.4, sigmaY=2.4
            )
        if float(traversability.max()) > 0.0:
            traversability /= float(traversability.max())

        clearance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if float(clearance.max()) > 0.0:
            clearance /= float(clearance.max())

        clearance_m: np.ndarray | None = None
        min_clearance = float("inf")
        if terrain is not None and terrain.clearance_m.shape == mask.shape:
            clearance_m = terrain.clearance_m
            min_clearance = float(terrain.min_forward_clearance_m)
            clearance = self._blend_metric_clearance(clearance, clearance_m, terrain)

        ego_heading = 0.0
        if vehicle_state is not None:
            ego_heading = float(vehicle_state.heading_rad)

        return _PlannerScene(
            mask=mask.astype(bool),
            traversability=traversability,
            clearance=clearance,
            stability_score=float(stability_score),
            lateral_prior_px=float(prior_state[0]),
            heading_prior=float(prior_state[1]),
            ego_heading_rad=ego_heading,
            clearance_m=clearance_m,
            min_clearance_m=min_clearance,
        )

    def _blend_metric_clearance(
        self,
        pixel_clearance: np.ndarray,
        clearance_m: np.ndarray,
        terrain: TerrainAnalysis,
    ) -> np.ndarray:
        """Pixel distance shrinks with range purely because of perspective,
        biasing the planner toward the bottom of the frame; the metric channel
        makes a far gap comparable to a near one."""
        if self._depth_clearance_weight <= 0.0:
            return pixel_clearance

        # One vehicle width of room is "as clear as it needs to be".
        full_clearance = max(2.0 * self._vehicle_half_width_m, 1e-3)
        metric = np.clip(clearance_m / full_clearance, 0.0, 1.0).astype(np.float32)
        metric = np.where(terrain.valid, metric, pixel_clearance)

        blended = (
            1.0 - self._depth_clearance_weight
        ) * pixel_clearance + self._depth_clearance_weight * metric

        # Anything too narrow to drive through is pushed down hard rather
        # than merely scored lower, so the planner routes around it.
        too_narrow = terrain.valid & (clearance_m < self._min_clearance_m)
        blended[too_narrow] *= 1.0 - self._obstacle_penalty
        return blended.astype(np.float32)

    def _path_clearance(
        self,
        centerline: np.ndarray,
        scene: _PlannerScene,
        fallback: float,
    ) -> float:
        if scene.clearance_m is None or len(centerline) == 0:
            return fallback

        h, w = scene.clearance_m.shape[:2]
        xs = np.clip(np.round(centerline[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(centerline[:, 1]).astype(int), 0, h - 1)
        sampled = scene.clearance_m[ys, xs]
        on_road = scene.mask[ys, xs]
        if on_road.any():
            sampled = sampled[on_road]
        if sampled.size == 0:
            return fallback
        return float(min(float(sampled.min()), fallback))

    def _postprocess_trajectory(self, waypoints: np.ndarray, h: int, w: int) -> np.ndarray:
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
        count = len(x_vals)
        if count < 3:
            return x_vals

        window = min(self._smoothing_window, count)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return x_vals

        polyorder = 1
        if window >= 5:
            polyorder = 2
        return savgol_filter(x_vals, window_length=window, polyorder=polyorder, mode="interp")

    def _stabilize_trajectory(self, centerline: np.ndarray, w: int) -> np.ndarray:
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

    def _straighten_trajectory(self, centerline: np.ndarray, w: int) -> np.ndarray:
        """Mask noise on a straight road otherwise reads as small curves the
        controller would chase."""
        if len(centerline) < 3 or self._straight_blend <= 0.0:
            return centerline

        residual_px, heading_change = self._straightness_metrics(centerline)
        if (
            residual_px > self._straight_residual_px
            or heading_change > self._straight_heading_threshold
        ):
            return centerline

        line_coeffs = np.polyfit(centerline[:, 1], centerline[:, 0], deg=1)
        line_x = np.polyval(line_coeffs, centerline[:, 1])

        straightened = centerline.copy()
        straightened[:, 0] = np.clip(
            self._straight_blend * line_x + (1.0 - self._straight_blend) * centerline[:, 0],
            0.0,
            float(w - 1),
        )
        return straightened

    @staticmethod
    def _straightness_metrics(centerline: np.ndarray) -> tuple[float, float]:
        line_coeffs = np.polyfit(centerline[:, 1], centerline[:, 0], deg=1)
        line_x = np.polyval(line_coeffs, centerline[:, 1])
        residual_px = float(np.mean(np.abs(centerline[:, 0] - line_x)))

        dx = np.diff(centerline[:, 0])
        dy = np.diff(centerline[:, 1])
        valid = dy > 1e-6
        if not np.any(valid):
            return residual_px, 0.0

        headings = np.unwrap(np.arctan2(dx[valid], dy[valid]))
        if len(headings) == 0:
            return residual_px, 0.0

        window = min(3, len(headings))
        near_heading = float(np.mean(headings[-window:]))
        far_heading = float(np.mean(headings[:window]))
        heading_change = abs(far_heading - near_heading)
        return residual_px, heading_change

    def _fallback(
        self,
        h: int,
        w: int,
        state: np.ndarray,
        min_clearance_m: float = float("inf"),
    ) -> PathPlan:
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
            min_clearance_m=min_clearance_m,
        )

    @staticmethod
    def _estimate_road_width(mask: np.ndarray, centerline: np.ndarray) -> float:
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

        if not widths:
            return 0.0
        return float(np.mean(widths))

    @staticmethod
    def _estimate_heading(centerline: np.ndarray) -> float:
        """From the near third only: that is the part the vehicle is about to
        drive, and the far end is the least precise."""
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
        self._kf = _KalmanTracker(
            q=self._kf.Q[0, 0],
            r=self._kf.R[0, 0],
        )
        self._consecutive_misses = 0
        self._prev_centerline = None
        self._baseline.reset()
        self._last_good = None
        self._held = 0
