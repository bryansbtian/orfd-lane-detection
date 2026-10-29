"""Observer-Only Control Diagnostics For R3 Development And Verification."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


class TrackingDiagnostics:
    def __init__(
        self, path: Path, camera, snapshot_interval_s: float, snapshots: bool = False
    ) -> None:
        self._stream = path.open("x", encoding="utf-8")
        self._camera = camera
        self._snapshots = snapshots
        self._snapshot_interval_s = snapshot_interval_s
        self._directory = path.with_suffix("")
        self._next_snapshot = 0.0

    def record(self, result, state, before: dict, after: dict) -> None:
        forward, right, keep = self._camera.image_to_ground(result.plan.centerline)
        record = {
            "before": before,
            "after": after,
            "capture_host_timestamp": result.capture.timestamp,
            "frame_id": result.capture.frame_id,
            "speed_mps": state.speed_mps,
            "confidence": max(result.perception.confidences, default=0.0),
            "confidences": result.perception.confidences,
            "road_fraction": result.stabilized.road_fraction,
            "plan_status": result.plan.status,
            "fallback_reason": result.plan.fallback_reason,
            "speed_scale": result.plan.speed_scale,
            "path_forward_m": forward[keep].tolist(),
            "path_right_m": right[keep].tolist(),
            "command": asdict(result.command),
            "timings_ms": result.timings_ms,
        }

        # Infinity means an unseen edge, not a measured large clearance. JSON
        # null retains that distinction without nonstandard NaN tokens.
        def clean(value):
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        self._stream.write(json.dumps(clean(record), allow_nan=False) + "\n")
        self._stream.flush()
        if self._snapshots and before["timestamp_s"] >= self._next_snapshot:
            self._directory.mkdir(exist_ok=True)
            label = f"{before['timestamp_s']:08.3f}"
            cv2.imwrite(str(self._directory / f"{label}.jpg"), result.frame.preprocessed)
            if result.plan.planner_mask is not None:
                cv2.imwrite(
                    str(self._directory / f"{label}_mask.png"),
                    result.plan.planner_mask.astype(np.uint8) * 255,
                )
            self._next_snapshot = before["timestamp_s"] + self._snapshot_interval_s

    def close(self) -> None:
        self._stream.close()
