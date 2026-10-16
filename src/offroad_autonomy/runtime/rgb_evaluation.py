"""Label-based RGB metrics, independent of simulator geometry or vehicle pose."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def manual_path_metrics(
    centerline: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    working_size: tuple[int, int],
) -> dict:
    """Measure a rasterized path against independent native-grid annotations."""
    h, w = truth.shape
    points = np.asarray(centerline, dtype=np.float64)
    if len(points) < 2 or not np.isfinite(points).all():
        return {"path_on_manual_road_fraction": None, "minimum_manual_clearance_px": None}
    points = points * [w / working_size[0], h / working_size[1]]
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] <= w - 1)
        & (points[:, 1] >= 0)
        & (points[:, 1] <= h - 1)
    )
    if not inside.all():
        return {"path_on_manual_road_fraction": 0.0, "minimum_manual_clearance_px": 0.0}
    raster = np.zeros((h, w), np.uint8)
    cv2.polylines(raster, [np.rint(points).astype(np.int32)], False, 1, 1)
    on_path = raster.astype(bool)
    distance = cv2.distanceTransform(np.pad((truth & valid).astype(np.uint8), 1), cv2.DIST_L2, 5)
    return {
        "path_on_manual_road_fraction": float((truth & valid)[on_path].mean()),
        "path_unknown_fraction": float((~valid)[on_path].mean()),
        "minimum_manual_clearance_px": float(distance[1:-1, 1:-1][on_path].min()),
    }


@dataclass
class MaskMetrics:
    # Rows are truth, columns prediction. Void pixels never enter the counts.
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.int64))
    frames: int = 0
    temporal_ious: list[float] = field(default_factory=list)
    change_agreements: list[float] = field(default_factory=list)
    _previous: tuple | None = None

    def reset_sequence(self) -> None:
        self._previous = None

    def add(self, prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> None:
        if prediction.shape != truth.shape or valid.shape != truth.shape:
            raise ValueError("Prediction, annotation and valid mask must share a grid")
        prediction = prediction.astype(bool, copy=False)
        truth = truth.astype(bool, copy=False)
        codes = truth[valid].astype(np.uint8) * 2 + prediction[valid]
        self.confusion += np.bincount(codes, minlength=4).reshape(2, 2)
        self.frames += 1
        if self._previous is not None:
            prev_prediction, prev_truth, prev_valid = self._previous
            both = valid & prev_valid
            union = (prediction | prev_prediction) & both
            if union.any():
                self.temporal_ious.append(
                    float((prediction & prev_prediction & both).sum() / union.sum())
                )
            # Exclude irrelevant sky pixels from change agreement. A consistently
            # wrong mask can still be stable, so always report spatial IoU too.
            active = (prediction | prev_prediction | truth | prev_truth) & both
            if active.any():
                agreement = (prediction ^ prev_prediction) == (truth ^ prev_truth)
                self.change_agreements.append(float(agreement[active].mean()))
        self._previous = prediction.copy(), truth.copy(), valid.copy()

    def report(self) -> dict:
        tn, fp, fn, tp = self.confusion.ravel().tolist()
        road_iou = None
        background_iou = None
        if tp + fp + fn:
            road_iou = tp / (tp + fp + fn)
        if tn + fp + fn:
            background_iou = tn / (tn + fp + fn)
        present = [v for v in (road_iou, background_iou) if v is not None]
        miou = None
        if present:
            miou = float(np.mean(present))
        tiou = None
        stability = None
        if self.temporal_ious:
            tiou = float(np.mean(self.temporal_ious))
        if self.change_agreements:
            stability = float(np.mean(self.change_agreements))
        return {
            "frames": self.frames,
            "confusion_tn_fp_fn_tp": [tn, fp, fn, tp],
            "road_iou": road_iou,
            "background_iou": background_iou,
            "miou": miou,
            "tiou_unwarped": tiou,
            "label_relative_change_agreement": stability,
            "temporal_pairs": len(self.temporal_ious),
        }
