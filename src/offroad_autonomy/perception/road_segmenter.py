"""Traversable-road segmentation with a YOLOE-26 model.

Accepts PyTorch weights (open-vocabulary, prompted at load time) or an
exported TensorRT engine, which is what makes the model fast enough on a
Jetson; see ``scripts/export_engine.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from offroad_autonomy.perception.ego_mask import (
    apply_roi,
    road_fraction,
    weighted_confidence,
)
from offroad_autonomy.types import FramePacket, PerceptionResult, PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception")


class RoadSegmenter:
    def __init__(self, config: PipelineConfig) -> None:
        weights = config.model_weights
        self._conf = config.confidence_threshold
        self._imgsz = int(config.perception_input_size)

        logger.info("Loading model weights: %s", weights)
        self._model = YOLO(weights, task="segment")

        # Exported engines have their classes compiled in; re-prompting them
        # is impossible, and fine-tuned checkpoints have fixed classes too.
        prompts = config.perception_prompts
        if prompts and Path(weights).suffix == ".pt":
            try:
                self._model.set_classes(list(prompts))
                logger.info("Open-vocab classes: %s", prompts)
            except (TypeError, AssertionError, AttributeError):
                logger.info("Model has fixed classes - skipping open-vocab prompts")

    def predict(
        self,
        frame: FramePacket,
        valid_roi: np.ndarray | None = None,
    ) -> PerceptionResult:
        """The model sees the whole image because it needs the context, and
        blanking a region out can itself confuse an open-vocabulary segmenter.
        Detections are clipped to ``valid_roi`` afterwards instead, so bodywork
        in frame cannot move the confidence or the traversable fraction.
        """
        img = frame.preprocessed
        h, w = frame.height, frame.width

        t0 = time.perf_counter()
        # Without retina_masks the masks come back at the padded network
        # resolution, and a plain resize shifts them against the frame.
        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, verbose=False, retina_masks=True
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        result = None
        if results:
            result = results[0]
        instances, raw_confs = self._extract_masks(result, h, w)

        if valid_roi is not None and valid_roi.shape != (h, w):
            logger.debug("Ignoring ROI of shape %s for a %s frame", valid_roi.shape, (h, w))
            valid_roi = None

        combined = np.zeros((h, w), dtype=bool)
        for instance in instances:
            combined |= instance

        mask = apply_roi(combined, valid_roi)
        confs = weighted_confidence(instances, raw_confs, valid_roi)

        return PerceptionResult(
            mask=mask,
            confidences=confs,
            num_detections=len(confs),
            inference_time_ms=t_ms,
            # Fusion replaces ``mask``; the model's own union is kept so the
            # debug views can show what depth changed.
            rgb_mask=mask,
            valid_roi=valid_roi,
            road_fraction=road_fraction(mask, valid_roi),
        )

    @staticmethod
    def _extract_masks(result, h: int, w: int) -> tuple[list[np.ndarray], list[float]]:
        """Per instance rather than merged, so a blob lying entirely on the ego
        body can be dropped from the confidence statistics."""
        instances: list[np.ndarray] = []
        confs: list[float] = []

        if result is None or result.masks is None:
            return instances, confs

        for i, m in enumerate(result.masks.data):
            m_np = m.cpu().numpy().astype(np.uint8)
            if m_np.shape != (h, w):
                m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
            instances.append(m_np.astype(bool))

            conf = 1.0
            if result.boxes is not None and i < len(result.boxes.conf):
                conf = float(result.boxes.conf[i].cpu())
            confs.append(conf)

        return instances, confs
