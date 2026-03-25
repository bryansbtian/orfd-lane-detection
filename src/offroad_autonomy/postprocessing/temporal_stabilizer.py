"""Temporal mask stabilisation for frame-to-frame consistency.

Applies exponential moving average (EMA) blending and morphological
cleanup to reduce jitter in the binary road mask across consecutive
frames.  Also tracks a temporal IoU stability score.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult

logger = logging.getLogger("offroad_autonomy.postprocessing")


class TemporalStabilizer:
    """EMA-based mask smoother with morphological cleanup."""

    def __init__(self, config: PipelineConfig) -> None:
        self._alpha = config.ema_alpha
        self._min_area_frac = config.min_mask_area_fraction
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size),
        )
        self._accum: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None

    def stabilize(self, result: PerceptionResult) -> StabilizedResult:
        """Smooth the perception mask and compute a stability score."""
        mask_f = result.mask.astype(np.float32)

        if self._accum is None:
            self._accum = mask_f.copy()
        else:
            if self._accum.shape != mask_f.shape:
                self._accum = mask_f.copy()
            else:
                self._accum = self._alpha * mask_f + (1.0 - self._alpha) * self._accum

        smoothed = (self._accum > 0.5).astype(np.uint8)

        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, self._kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, self._kernel)

        h, w = smoothed.shape
        min_area = int(h * w * self._min_area_frac)
        if int(smoothed.sum()) < min_area:
            smoothed = np.zeros_like(smoothed)

        stable_mask = smoothed.astype(bool)
        stability = self._temporal_iou(stable_mask)
        self._prev_mask = stable_mask

        return StabilizedResult(
            mask=stable_mask,
            stability_score=stability,
            raw_result=result,
        )

    def _temporal_iou(self, mask: np.ndarray) -> float:
        if self._prev_mask is None:
            return 1.0
        inter = int((mask & self._prev_mask).sum())
        union = int((mask | self._prev_mask).sum())
        return inter / union if union > 0 else 1.0

    def reset(self) -> None:
        """Clear internal state (e.g. on scene change)."""
        self._accum = None
        self._prev_mask = None
