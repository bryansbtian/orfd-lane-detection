"""Traversable-road segmentation with YOLOE or fixed-class semantic weights.

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
        self._semantic = self._model.task == "semantic"

        # Exported engines have their classes compiled in; re-prompting them
        # is impossible, and fine-tuned checkpoints have fixed classes too.
        prompts = config.perception_prompts
        if self._semantic:
            from offroad_autonomy.perception.semantic_predictor import RoadSemanticPredictor

            self._predictor = RoadSemanticPredictor
            names = self._model.names
            selected = {name.strip().casefold() for name in prompts}
            self._road_ids = [
                int(idx) for idx, name in names.items() if name.strip().casefold() in selected
            ]
            if not self._road_ids:
                raise ValueError(
                    "No semantic road class matches perception.prompts. "
                    f"Set prompts to the traversable class names from {names}."
                )
            # Single-logit models use 0=background, 1=foreground even though
            # their names dictionary lists the sole foreground class as 0.
            if len(names) == 1:
                self._road_ids = [1]
            logger.info("Semantic road classes: %s (mask IDs %s)", names, self._road_ids)
        elif prompts and Path(weights).suffix == ".pt":
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
        kwargs = {"predictor": self._predictor} if self._semantic else {}
        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, verbose=False, retina_masks=True, **kwargs
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        result = None
        if results:
            result = results[0]

        if valid_roi is not None and valid_roi.shape != (h, w):
            logger.debug("Ignoring ROI of shape %s for a %s frame", valid_roi.shape, (h, w))
            valid_roi = None

        if self._semantic:
            return self._semantic_result(result, h, w, valid_roi, t_ms)
        instances, raw_confs = self._extract_masks(result, h, w)

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
            valid_roi=valid_roi,
            road_fraction=road_fraction(mask, valid_roi),
        )

    def _semantic_result(self, result, h, w, valid_roi, t_ms) -> PerceptionResult:
        mask = np.zeros((h, w), dtype=bool)
        confs = []
        if result is not None and result.semantic_mask is not None:
            labels = result.semantic_mask.data.cpu().numpy()
            confidence = result.semantic_confidence.cpu().numpy()
            if labels.shape != (h, w) or confidence.shape != (h, w):
                raise ValueError("Semantic prediction is not aligned with the camera frame")
            mask = np.isin(labels, self._road_ids) & (confidence >= self._conf)
            mask = apply_roi(mask, valid_roi)
            if mask.any():
                confs = [float(confidence[mask].mean())]
        return PerceptionResult(
            mask=mask,
            confidences=confs,
            num_detections=len(confs),
            inference_time_ms=t_ms,
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
