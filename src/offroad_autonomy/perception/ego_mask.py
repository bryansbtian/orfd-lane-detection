"""Exclusion of the ego vehicle's own bodywork from perception.

On the shipped bumper rig this stage is a no-op. It stays because the mask
is a property of a *pose*, not of the codebase: any mount that looks out over
the hood needs one, and the invariants below are what make such a mount safe.

Bodywork pixels are not terrain. Treating them as non-drivable would tell the
planner the vehicle is walled in by its own bonnet; treating them as drivable
would invite it to steer into itself. Both are wrong, because the right answer
is that nothing can be known about them at all.

So this stage produces a **validity** mask rather than a classification. Every
downstream consumer - segmentation, confidence, terrain, planning, the safe
stop - is restricted to the valid region, and the confidence denominator
counts valid pixels only. A camera that sees more bodywork therefore reports
the same confidence on the same road, which is the property that stops the
vehicle safe-stopping at a perfectly clear trail.

The polygon is specified in normalised coordinates so one definition covers
both the capture resolution and the 720x465 processing grid. Each camera has
its own polygon, since the same bodywork sits differently in each view.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.types import EgoMaskSpec

logger = logging.getLogger("offroad_autonomy.perception.ego_mask")


class EgoMask:
    def __init__(self, spec: EgoMaskSpec | None) -> None:
        if spec is None:
            spec = EgoMaskSpec()
        self.spec = spec
        self._cache: dict[tuple[int, int], np.ndarray] = {}

        if self.enabled:
            logger.info(
                "Ego-vehicle exclusion active: %d-point polygon, %d px margin",
                len(self.spec.polygon),
                self.spec.margin_px,
            )

    @property
    def enabled(self) -> bool:
        return bool(self.spec.enabled and len(self.spec.polygon) >= 3)

    def excluded(self, shape: tuple[int, int]) -> np.ndarray:
        key = (int(shape[0]), int(shape[1]))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._build(key)
            self._cache[key] = cached
        return cached

    def valid_roi(self, shape: tuple[int, int]) -> np.ndarray:
        return ~self.excluded(shape)

    def coverage(self, shape: tuple[int, int]) -> float:
        return float(self.excluded(shape).mean())

    def _build(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        excluded = np.zeros((height, width), dtype=np.uint8)
        if not self.enabled:
            return excluded.astype(bool)

        points = np.asarray(self.spec.polygon, dtype=np.float64)
        pixels = np.stack([points[:, 0] * (width - 1), points[:, 1] * (height - 1)], axis=1)
        cv2.fillPoly(excluded, [np.round(pixels).astype(np.int32)], 1)

        margin = int(self.spec.margin_px)
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
            excluded = cv2.dilate(excluded, kernel)

        mask = excluded.astype(bool)
        if mask.all():
            # A polygon covering everything would starve the whole stack, and
            # silently returning an empty ROI is far worse than shouting.
            raise ValueError(
                "Ego mask covers the entire frame - check cameras.<side>.ego_mask.polygon"
            )
        return mask


def apply_roi(mask: np.ndarray, valid_roi: np.ndarray | None) -> np.ndarray:
    if valid_roi is None or valid_roi.shape != mask.shape:
        return mask
    return mask & valid_roi


def road_fraction(mask: np.ndarray, valid_roi: np.ndarray | None) -> float:
    """The denominator is the valid region, never the whole frame, so masking
    more bodywork cannot change the answer for an unchanged road."""
    if valid_roi is None:
        total = mask.size
        covered = int(mask.sum())
    else:
        total = int(valid_roi.sum())
        covered = int((mask & valid_roi).sum())
    if total <= 0:
        return 0.0
    return covered / total


def weighted_confidence(
    instance_masks: list[np.ndarray],
    instance_confs: list[float],
    valid_roi: np.ndarray | None,
) -> list[float]:
    """A blob the segmenter found on the hood contributes nothing, and a road
    detection clipped by the hood is not penalised for the clipped part."""
    if valid_roi is None:
        return list(instance_confs)

    kept: list[float] = []
    for mask, conf in zip(instance_masks, instance_confs):
        if mask.shape != valid_roi.shape:
            kept.append(conf)
            continue
        inside = int((mask & valid_roi).sum())
        if inside <= 0:
            continue
        kept.append(conf)
    return kept
