"""Road segmentation using YOLOE-26 family models.

Wraps the Ultralytics YOLO API into a clean adapter that accepts a
``FramePacket`` and returns a ``PerceptionResult``. The concrete model
(weights, size, prompts) is driven entirely by ``PipelineConfig``.

The class is designed so swapping to a different segmentation backend
(SAM3, SAM 2.1, a custom TensorRT engine, etc.) only requires
implementing the same ``predict`` interface.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from ultralytics import YOLO

from offroad_autonomy.types import FramePacket, PerceptionResult, PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception")


class RoadSegmenter:
    """YOLOE-26 based traversable-road segmenter.

    Supports both open-vocabulary models (via text prompts) and fine-tuned
    checkpoints (automatic class loading).
    """

    def __init__(self, config: PipelineConfig) -> None:
        weights = config.model_weights
        self._conf = config.confidence_threshold

        logger.info("Loading model weights: %s", weights)
        self._model = YOLO(weights)

        prompts = config.perception_prompts
        if prompts:
            try:
                self._model.set_classes(prompts)
                logger.info("Open-vocab classes: %s", prompts)
            except TypeError:
                logger.info("Fine-tuned model detected - skipping open-vocab prompts")

    def predict(self, frame: FramePacket) -> PerceptionResult:
        """Run segmentation on the preprocessed frame."""
        img = frame.preprocessed
        h, w = frame.height, frame.width

        t0 = time.perf_counter()
        results = self._model.predict(img, conf=self._conf, verbose=False)
        t_ms = (time.perf_counter() - t0) * 1000.0

        result = results[0] if results else None
        mask, confs = self._extract_masks(result, h, w)

        return PerceptionResult(
            mask=mask,
            confidences=confs,
            num_detections=len(confs),
            inference_time_ms=t_ms,
        )

    @staticmethod
    def _extract_masks(result, h: int, w: int) -> tuple[np.ndarray, list[float]]:
        """Combine all instance masks into a single binary road mask."""
        combined = np.zeros((h, w), dtype=bool)
        confs: list[float] = []

        if result is None or result.masks is None:
            return combined, confs

        for i, m in enumerate(result.masks.data):
            m_np = m.cpu().numpy().astype(np.uint8)
            if m_np.shape != (h, w):
                m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
            combined |= m_np.astype(bool)

            conf = 1.0
            if result.boxes is not None and i < len(result.boxes.conf):
                conf = float(result.boxes.conf[i].cpu())
            confs.append(conf)

        return combined, confs
