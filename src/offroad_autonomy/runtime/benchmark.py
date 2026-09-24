"""Per-run benchmark recording for the A-E experiment configurations.

Everything here is measured from the live loop, not estimated. Two metrics
are proxies and are labelled as such in the report:

* **path_jitter_pct** - standard deviation of the frame-to-frame change in
  the planned look-ahead point's lateral position, as a percentage of image
  width. Lower is steadier. It is image-space, so compare it only between
  runs on the same map, spawn and segmentation view.
* **lane_departures** - how many times the ground directly ahead of the
  hood stopped looking like road for ``departure_frames`` consecutive frames.
  There is no ground-truth lane in an off-road map; this counts the moments
  the vehicle's own footprint left the mask it was following.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from offroad_autonomy.runtime.timing import MAIN_STAGES, STEREO_STAGES, RuntimeStats
from offroad_autonomy.types import PathPlan


def footprint_strip(
    valid_roi: np.ndarray, width_fraction: float = 0.24, band_fraction: float = 0.08
) -> np.ndarray:
    """Ground just above the hood, across the vehicle's width: the part of
    the mask the vehicle is about to drive over."""
    height, width = valid_roi.shape
    c0 = int(width * (0.5 - width_fraction / 2.0))
    c1 = max(c0 + 1, int(width * (0.5 + width_fraction / 2.0)))
    cols = valid_roi[:, c0:c1]
    flipped = cols[::-1]
    has_valid = flipped.any(axis=0)
    lowest = height - 1 - np.argmax(flipped, axis=0)
    band = max(2, int(height * band_fraction))

    strip = np.zeros_like(valid_roi, dtype=bool)
    for offset, (ok, bottom) in enumerate(zip(has_valid, lowest)):
        if ok:
            strip[max(0, bottom - band) : bottom + 1, c0 + offset] = True
    return strip & valid_roi


class BenchmarkRecorder:
    def __init__(
        self,
        label: str,
        config_path: str,
        valid_roi: np.ndarray,
        warmup_s: float = 5.0,
        departure_frames: int = 5,
        departure_threshold: float = 0.5,
    ) -> None:
        self.label = label
        self.config_path = config_path
        self._warmup_s = float(warmup_s)
        # Warm-up counts from the first frame, not from construction: the
        # recorder exists before BeamNG has even launched.
        self._t_origin: float | None = None
        self._t_first: float | None = None
        self._t_last: float | None = None
        self._strip = footprint_strip(valid_roi)
        self._departure_frames = int(departure_frames)
        self._departure_threshold = float(departure_threshold)
        self._off_road_run = 0
        self._departed = False

        self.frames = 0
        self.departures = 0
        self._primary_ms: list[float] = []
        self._full_ms: list[float] = []
        self._confidence: list[float] = []
        self._valid_disparity: list[float] = []
        self._depth_coverage: list[float] = []
        self._depth_used = 0
        self._targets: list[float] = []

    def add(
        self,
        now: float,
        primary_ms: float,
        full_ms: float,
        confidence: float,
        mask: np.ndarray,
        plan: PathPlan | None,
        valid_disparity: float | None,
        depth_coverage: float | None,
        depth_used: bool,
    ) -> None:
        if self._t_origin is None:
            self._t_origin = now
        if now - self._t_origin < self._warmup_s:
            return
        if self._t_first is None:
            self._t_first = now
        self._t_last = now
        self.frames += 1

        self._primary_ms.append(primary_ms)
        self._full_ms.append(full_ms)
        self._confidence.append(confidence)
        if valid_disparity is not None:
            self._valid_disparity.append(valid_disparity)
        if depth_coverage is not None:
            self._depth_coverage.append(depth_coverage)
        self._depth_used += int(depth_used)

        if plan is not None and len(plan.centerline) >= 2:
            target = plan.centerline[max(0, len(plan.centerline) // 3)]
            self._targets.append(float(target[0]) / max(mask.shape[1], 1))

        if self._strip.shape == mask.shape and self._strip.any():
            on_road = float((mask & self._strip).sum()) / float(self._strip.sum())
            if on_road < self._departure_threshold:
                self._off_road_run += 1
                if self._off_road_run >= self._departure_frames and not self._departed:
                    self.departures += 1
                    self._departed = True
            else:
                self._off_road_run = 0
                self._departed = False

    @staticmethod
    def _stat(values: list[float], fn) -> float | None:
        if not values:
            return None
        return float(fn(np.asarray(values)))

    def report(self, main: RuntimeStats, stereo: RuntimeStats | None, worker=None) -> dict:
        duration = 0.0
        if self._t_first is not None and self._t_last:
            duration = self._t_last - self._t_first
        jitter = None
        if len(self._targets) >= 3:
            jitter = float(np.std(np.diff(np.asarray(self._targets))) * 100.0)

        stereo_block: dict = {"enabled": stereo is not None}
        if stereo is not None:
            total = stereo.stage("stereo_total")
            stereo_block.update(
                fps=round(stereo.fps(), 2),
                latency_mean_ms=round(total.mean_ms, 2),
                latency_p95_ms=round(total.p95_ms, 2),
                stages={
                    name: {"mean_ms": round(s.mean_ms, 2), "p95_ms": round(s.p95_ms, 2)}
                    for name, s in stereo.summary(STEREO_STAGES).items()
                },
            )
        if worker is not None:
            stereo_block.update(
                submitted=worker.submitted,
                completed=worker.completed,
                dropped=worker.dropped,
                failed=worker.failed,
            )

        def pct(value: float | None) -> float | None:
            if value is None:
                return None
            return round(100.0 * value, 2)

        main_fps = None
        if duration > 0:
            main_fps = round(self.frames / duration, 2)
        depth_used_pct = None
        if self.frames:
            depth_used_pct = round(100.0 * self._depth_used / self.frames, 1)
        primary_mean = self._stat(self._primary_ms, np.mean)
        primary_p95 = self._stat(self._primary_ms, lambda a: np.percentile(a, 95))
        return {
            "label": self.label,
            "config": self.config_path,
            "duration_s": round(duration, 1),
            "frames": self.frames,
            "main_fps": main_fps,
            "primary_latency_mean_ms": _round(primary_mean),
            "primary_latency_p95_ms": _round(primary_p95),
            "full_loop_mean_ms": _round(self._stat(self._full_ms, np.mean)),
            "full_loop_p95_ms": _round(self._stat(self._full_ms, lambda a: np.percentile(a, 95))),
            "meets_20fps": bool(duration > 0 and self.frames / duration >= 20.0),
            "meets_50ms": bool(primary_p95 is not None and primary_p95 <= 50.0),
            "segmentation_confidence_mean": _round(self._stat(self._confidence, np.mean), 3),
            "valid_disparity_pct_mean": pct(self._stat(self._valid_disparity, np.mean)),
            "depth_coverage_pct_mean": pct(self._stat(self._depth_coverage, np.mean)),
            "depth_used_pct": depth_used_pct,
            "path_jitter_pct": _round(jitter, 3),
            "lane_departures": self.departures,
            "stereo": stereo_block,
            "main_stages": {
                name: {"mean_ms": round(s.mean_ms, 2), "p95_ms": round(s.p95_ms, 2)}
                for name, s in main.summary(MAIN_STAGES).items()
            },
            "notes": "path_jitter_pct and lane_departures are image-space proxies",
        }

    def save(self, path: str | Path, report: dict) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)
