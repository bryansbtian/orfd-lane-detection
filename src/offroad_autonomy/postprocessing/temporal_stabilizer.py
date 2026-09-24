"""Temporal mask stabilisation.

A raw per-frame mask flickers at its edges, and the planner would turn that
flicker into steering jitter.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.perception.ego_mask import road_fraction
from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult

logger = logging.getLogger("offroad_autonomy.postprocessing")


class TemporalStabilizer:
    def __init__(self, config: PipelineConfig) -> None:
        self._alpha = config.ema_alpha
        self._min_area_frac = config.min_mask_area_fraction
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size),
        )
        # Separate switches so each stage can be verified on its own against
        # the raw mask.
        self._use_morphology = bool(config.enable_morphology)
        self._use_ema = bool(config.enable_ema)
        self._accum: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None

    def stabilize(self, result: PerceptionResult) -> StabilizedResult:
        mask_f = result.mask.astype(np.float32)

        if not self._use_ema or self._accum is None or self._accum.shape != mask_f.shape:
            self._accum = mask_f.copy()
        else:
            # With alpha > 0.5 a single empty frame drops every pixel below
            # 0.5, so EMA does not bridge a one-frame dropout; the perception
            # gate's path hold does that instead.
            self._accum = self._alpha * mask_f + (1.0 - self._alpha) * self._accum

        smoothed = (self._accum > 0.5).astype(np.uint8)

        if self._use_morphology:
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, self._kernel)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, self._kernel)

        h, w = smoothed.shape
        min_area = int(h * w * self._min_area_frac)
        if int(smoothed.sum()) < min_area:
            smoothed = np.zeros_like(smoothed)

        stable_mask = smoothed.astype(bool)

        # Morphology can push the mask a few pixels outward, so re-apply the
        # ego exclusion afterwards: closing a gap must never grow the road
        # onto the vehicle's own bodywork.
        roi = result.valid_roi
        if roi is not None and roi.shape == stable_mask.shape:
            stable_mask &= roi

        stability = self._temporal_iou(stable_mask)
        self._prev_mask = stable_mask

        return StabilizedResult(
            mask=stable_mask,
            stability_score=stability,
            raw_result=result,
            traversability=self._carry_traversability(result, stable_mask),
            valid_roi=roi,
            road_fraction=road_fraction(stable_mask, roi),
        )

    @staticmethod
    def _carry_traversability(
        result: PerceptionResult,
        stable_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Clipped to the new mask rather than smoothed: blurring a confidence
        map across an obstacle edge would soften exactly the boundary the
        planner needs to respect."""
        field = result.traversability
        if field is None or field.shape != stable_mask.shape:
            return None
        carried = field.astype(np.float32, copy=True)
        carried[~stable_mask] = 0.0
        return carried

    def _temporal_iou(self, mask: np.ndarray) -> float:
        if self._prev_mask is None:
            return 1.0
        inter = int((mask & self._prev_mask).sum())
        union = int((mask | self._prev_mask).sum())
        if union <= 0:
            return 1.0
        return inter / union

    def reset(self) -> None:
        self._accum = None
        self._prev_mask = None
