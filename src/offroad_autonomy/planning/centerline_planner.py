"""Planning entry point: the perception gate, then the baseline planner.

Experiments subclass ``CenterlinePlanner`` and override ``_propose`` (or
``plan``) to swap the path source while keeping the gate and its fallback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.baseline_planner import BaselinePlanner
from offroad_autonomy.planning.corridor_validation import validate_corridor
from offroad_autonomy.planning.perception_gate import GateDecision, PerceptionGate
from offroad_autonomy.types import (
    BEVSnapshot,
    PathPlan,
    PipelineConfig,
    StabilizedResult,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.planning")


class CenterlinePlanner:
    def __init__(self, config: PipelineConfig, camera: CameraModel | None = None) -> None:
        self._gate = PerceptionGate(config)
        self._baseline = BaselinePlanner(config, camera=camera)
        self._hold_frames = max(0, int(config.gate_hold_frames))
        self._hold_speed_scale = float(np.clip(config.gate_hold_speed_scale, 0.0, 1.0))
        self._last_good: PathPlan | None = None
        self._held = 0

        self._camera = self._baseline._camera
        self._validate_corridor = config.validate_path_corridor
        self._path_clearance_px = config.path_min_clearance_px
        self._path_half_width_m = config.path_half_width_m
        if self._path_half_width_m > 0:
            self._path_half_width_m += config.path_boundary_margin_m
        self._projection_options = {
            "max_range_m": config.projection_max_range_m,
            "pitch_uncertainty_deg": config.projection_pitch_uncertainty_deg,
            "max_relative_error": config.projection_max_relative_error,
        }

    def plan(
        self,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState | None = None,
    ) -> PathPlan:
        """A frame that fails the gate never produces a new path: driving a
        path no perception supports is how the vehicle used to leave the
        trail. The last good path is held briefly at reduced speed instead.
        """
        decision = self._gate_decision(stabilized)
        if not decision.ok:
            return self._hold(decision)

        plan = self._propose(stabilized, decision, vehicle_state)
        if plan is None:
            return self._hold(decision)

        if self._validate_corridor:
            footprint_evidence = None
            if self._path_half_width_m > 0:
                footprint_evidence = self._gate.connected_evidence(stabilized, decision.component)
            check = validate_corridor(
                plan.centerline,
                decision.component,
                self._path_clearance_px,
                camera=self._camera,
                half_width_m=self._path_half_width_m,
                footprint_component=footprint_evidence,
            )
            if not check.ok:
                decision.reason = check.reason
                self._forget_path()
                return self._hold(decision)
            plan.boundary_clearance_px = check.min_clearance_px
            plan.vehicle_width_validated = self._path_half_width_m > 0
            if not plan.vehicle_width_validated:
                plan.status = "centerline_only"
            _, _, reliable = self._camera.reliable_ground(
                plan.centerline, **self._projection_options
            )
            if not reliable.all():
                decision.reason = "camera-ground projection exceeds uncertainty bound"
                self._baseline.reset()
                return self._hold(decision)
        else:
            plan.status = "unchecked"
        plan.planner_mask = decision.component
        plan.roi_top = decision.roi_top
        self._last_good = plan
        self._held = 0
        return plan

    def _gate_decision(self, stabilized: StabilizedResult) -> GateDecision:
        decision = self._gate.evaluate(stabilized)
        valid = stabilized.valid_roi
        if valid is not None and valid.shape == decision.component.shape:
            # A hood-cut row exposes only fragments of the road width. Using
            # their centre would steer around the hood as if it were an obstacle.
            decision.component[~valid.all(axis=1)] = False
        return decision

    def _propose(
        self,
        stabilized: StabilizedResult,
        decision: GateDecision,
        vehicle_state: VehicleState | None,
    ) -> PathPlan | None:
        """A path from a frame the gate accepted, or None with
        ``decision.reason`` set to hold the last good one instead."""
        del stabilized, vehicle_state
        centerline = self._baseline.centerline(decision.component, decision.roi_top)
        if len(centerline) < 2:
            decision.reason = "centerline too short"
            return None
        return PathPlan(
            centerline=centerline,
            heading_rad=self._estimate_heading(centerline),
            curvature=self._estimate_curvature(centerline),
            road_width_px=self._estimate_road_width(decision.component, centerline),
        )

    def _forget_path(self) -> None:
        """A path the corridor check rejected must not be blended into the next one."""
        self._baseline.reset()

    def bev_snapshot(self) -> BEVSnapshot | None:
        """The baseline plans in the image, so it has no bird's-eye evidence to show."""
        return None

    def _hold(self, decision: GateDecision) -> PathPlan:
        self._held += 1
        if self._held == 1:
            logger.warning("Perception gate: %s - holding previous path", decision.reason)
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
                valid=False,
                status="held",
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
            valid=False,
            status="unavailable",
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
        self._baseline.reset()
        self._last_good = None
        self._held = 0
