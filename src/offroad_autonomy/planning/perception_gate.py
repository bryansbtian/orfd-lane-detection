"""Decide whether a frame's perception is good enough to plan from.

The planner used to accept any mask with a few hundred pixels and, when it
could not find a path, extrapolate one from a Kalman state - which the
controller then drove at full target speed. With segmentation scores on
BeamNG hovering around the detection threshold, that meant the vehicle
regularly drove a path that no perception supported.

This gate sits in front of every planner mode. A frame is rejected when

* the best detection score is below ``gate_min_confidence``,
* fewer than ``min_road_pixels`` traversable pixels fall in the planner ROI,
* no road component reaches the near visible edge of the ROI (the road must start at
  the vehicle - a blob on the horizon is not somewhere we can drive to), or
* that component covers less than ``gate_min_mask_area`` of the ROI.

The ROI is the bottom ``planner_roi_height`` of the frame, so sky and
tree-line detections can never pull the path upward.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from offroad_autonomy.types import PipelineConfig, StabilizedResult

#: The road must start at the vehicle: a blob on the horizon is not somewhere
#: the vehicle can drive to.
_ANCHOR_FRACTION = 0.25


@dataclass
class GateDecision:
    ok: bool
    reason: str
    confidence: float
    road_pixels: int
    area_fraction: float
    component: np.ndarray
    roi_top: int


class PerceptionGate:
    def __init__(self, config: PipelineConfig) -> None:
        self._min_confidence = float(config.gate_min_confidence)
        self._min_area = float(config.gate_min_mask_area)
        self._min_pixels = int(config.min_road_pixels)
        self._roi_height = float(np.clip(config.planner_roi_height, 0.1, 1.0))

    def roi_top(self, height: int) -> int:
        return int(round(height * (1.0 - self._roi_height)))

    def current_evidence(self, stabilized: StabilizedResult) -> np.ndarray:
        mask = stabilized.mask.astype(bool, copy=True)
        valid = stabilized.valid_roi
        if valid is not None and valid.shape == mask.shape:
            mask &= valid
        raw = stabilized.raw_result
        if raw is not None:
            # Old EMA evidence cannot connect road that this frame cannot see.
            if raw.mask is None:
                mask[:] = False
            else:
                mask &= raw.mask
            confidence_map = raw.confidence_map
            if confidence_map is not None:
                if confidence_map.shape != mask.shape:
                    raise ValueError("Road confidence map is not aligned with the mask")
                mask &= np.isfinite(confidence_map) & (confidence_map >= self._min_confidence)
        return mask

    def connected_evidence(self, stabilized: StabilizedResult, component: np.ndarray) -> np.ndarray:
        """Keep the chosen corridor's trusted continuation beyond the gate ROI."""
        mask = self.current_evidence(stabilized)
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        seeds = np.unique(labels[component])
        return np.isin(labels, seeds[seeds > 0])

    def evaluate(self, stabilized: StabilizedResult) -> GateDecision:
        mask = self.current_evidence(stabilized)
        h, w = mask.shape[:2]
        top = self.roi_top(h)

        band = np.zeros_like(mask)
        band[top:] = mask[top:]
        valid = stabilized.valid_roi
        if valid is not None and valid.shape == mask.shape:
            band &= valid
            roi_pixels = int(valid[top:].sum())
        else:
            roi_pixels = (h - top) * w

        raw = stabilized.raw_result
        confidences: list[float] = []
        if raw is not None:
            confidences = raw.confidences
        confidence = float(max(confidences, default=0.0))
        road_pixels = int(band.sum())
        component = self._vehicle_component(band, top, valid)
        if raw is not None and raw.confidence_map is not None:
            confidence = 0.0
            if component.any():
                confidence = float(raw.confidence_map[component].mean())
        area = float(component.sum()) / max(roi_pixels, 1)

        reason = ""
        if not np.isfinite(confidence) or confidence < self._min_confidence:
            reason = f"low confidence {confidence:.2f}<{self._min_confidence:.2f}"
        elif road_pixels < self._min_pixels:
            reason = f"too few road pixels ({road_pixels})"
        elif not component.any():
            reason = "no road at vehicle"
        elif area < self._min_area:
            reason = f"road area {area:.1%}<{self._min_area:.0%}"

        return GateDecision(
            ok=not reason,
            reason=reason,
            confidence=confidence,
            road_pixels=road_pixels,
            area_fraction=area,
            component=component,
            roi_top=top,
        )

    @staticmethod
    def _vehicle_component(
        band: np.ndarray, top: int, valid: np.ndarray | None = None
    ) -> np.ndarray:
        h = band.shape[0]
        if not band.any():
            return band
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            band.astype(np.uint8), connectivity=8
        )
        anchor_top = h - max(1, int(round((h - top) * _ANCHOR_FRACTION)))
        if valid is not None and valid.shape == band.shape:
            # The hood hides the bottom of the dashcam image. Anchor each
            # column at its nearest visible ground, never inside the hood.
            rows = np.arange(h)[:, None]
            last = np.max(np.where(valid, rows, -1), axis=0)
            span = np.maximum(1, np.rint((last - top + 1) * _ANCHOR_FRACTION).astype(int))
            anchors = valid & (rows >= np.maximum(top, last - span + 1)) & (rows <= last)
            anchored = np.unique(labels[anchors])
        else:
            anchored = np.unique(labels[anchor_top:])
        anchored = anchored[anchored > 0]
        if len(anchored) == 0:
            return np.zeros_like(band)
        best = int(anchored[np.argmax(stats[anchored, cv2.CC_STAT_AREA])])
        return labels == best
