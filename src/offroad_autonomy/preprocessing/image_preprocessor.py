"""Frame preprocessing for the perception pipeline."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from offroad_autonomy.types import FramePacket, PipelineConfig

logger = logging.getLogger("offroad_autonomy.preprocessing")


class ImagePreprocessor:
    """Prepare raw camera frames for downstream perception.

    Operations (all optional and config-driven):
    1. Resize to the perception model's expected input dimensions.
    2. CLAHE contrast enhancement for challenging lighting.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._target_w = config.preprocess_width
        self._target_h = config.preprocess_height
        self._enable_clahe = config.enable_clahe
        self._clahe: cv2.CLAHE | None = None

        if self._enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
            )

    def process(self, raw_bgr: np.ndarray) -> FramePacket:
        """Convert a raw BGR frame into a ``FramePacket``."""
        timestamp = time.perf_counter()
        h, w = raw_bgr.shape[:2]

        frame = raw_bgr
        if (w, h) != (self._target_w, self._target_h):
            frame = cv2.resize(
                frame,
                (self._target_w, self._target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        if self._clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return FramePacket(
            raw=raw_bgr,
            preprocessed=frame,
            timestamp=timestamp,
            height=self._target_h,
            width=self._target_w,
        )
