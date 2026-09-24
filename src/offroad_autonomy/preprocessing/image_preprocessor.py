"""Frame preprocessing for the perception pipeline."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from offroad_autonomy.types import FramePacket, PipelineConfig

logger = logging.getLogger("offroad_autonomy.preprocessing")


class ImagePreprocessor:
    def __init__(
        self,
        config: PipelineConfig,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        # The stitched view is wider than one camera, so the segmentation
        # view may override the configured working size.
        width, height = target_size or (config.preprocess_width, config.preprocess_height)
        self._target_w = int(width)
        self._target_h = int(height)
        self._enable_clahe = config.enable_clahe
        self._clahe: cv2.CLAHE | None = None

        if self._enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
            )

    def process(self, raw_bgr: np.ndarray) -> FramePacket:
        timestamp = time.perf_counter()
        h, w = raw_bgr.shape[:2]

        frame = raw_bgr
        if (w, h) != (self._target_w, self._target_h):
            # INTER_AREA avoids the aliasing a bilinear shrink leaves behind.
            frame = cv2.resize(
                frame,
                (self._target_w, self._target_h),
                interpolation=cv2.INTER_AREA,
            )

        if self._clahe is not None:
            # Luminance only: equalising colour channels shifts hue, and the
            # segmenter relies on the colour of dirt versus vegetation.
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
