"""Rolling per-stage timing: mean, p95 and loop rate.

One ``RuntimeStats`` per thread of work (the control loop, the stereo
worker). Samples live in bounded windows, so memory is constant however long
a session runs, and every statistic describes recent behaviour rather than a
whole-session average that hides a regression.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

MAIN_STAGES = (
    "capture",
    "vehicle_state",
    "stitching",
    "preprocess",
    "segmentation",
    "fusion",
    "postprocess",
    "planning",
    "control",
    "actuation",
    "primary_loop",
    "full_loop",
)

STEREO_STAGES = (
    "rectification",
    "stereo_matching",
    "depth_filtering",
    "terrain",
    "stereo_total",
)

#: Kept apart from ``MAIN_STAGES``: mixing them made loop latency partly a
#: measure of how fast the window could draw.
DISPLAY_STAGES = (
    "display_capture",
    "dashboard_render",
    "dashboard_show",
    "dashboard_total",
)


@dataclass(frozen=True)
class StageStats:
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    last_ms: float = 0.0
    count: int = 0


class RuntimeStats:
    def __init__(self, window: int = 300) -> None:
        self._window = int(window)
        self._lock = threading.Lock()
        self._samples: dict[str, deque[float]] = {}
        self._ticks: deque[float] = deque(maxlen=self._window)

    def record(self, stage: str, elapsed_ms: float) -> None:
        with self._lock:
            samples = self._samples.get(stage)
            if samples is None:
                samples = self._samples[stage] = deque(maxlen=self._window)
            samples.append(float(elapsed_ms))

    def record_many(self, timings: dict[str, float]) -> None:
        for stage, elapsed in timings.items():
            self.record(stage, elapsed)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0)

    def tick(self, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        with self._lock:
            self._ticks.append(now)

    def fps(self) -> float:
        with self._lock:
            count = len(self._ticks)
            if count < 2:
                return 0.0
            span = self._ticks[-1] - self._ticks[0]
        if span <= 0.0:
            return 0.0
        return (count - 1) / span

    def stage(self, stage: str) -> StageStats:
        with self._lock:
            samples = self._samples.get(stage)
            if not samples:
                return StageStats()
            values = np.fromiter(samples, dtype=np.float64)
        return StageStats(
            mean_ms=float(values.mean()),
            p95_ms=float(np.percentile(values, 95)),
            last_ms=float(values[-1]),
            count=int(values.size),
        )

    def stages(self) -> list[str]:
        with self._lock:
            return list(self._samples)

    def summary(self, order: tuple[str, ...] = ()) -> dict[str, StageStats]:
        names = [name for name in order if name in self.stages()]
        names += [name for name in self.stages() if name not in names]
        return {name: self.stage(name) for name in names}

    def format_lines(self, order: tuple[str, ...] = ()) -> list[str]:
        return [
            f"{name:<16} mean {s.mean_ms:6.1f} ms   p95 {s.p95_ms:6.1f} ms"
            for name, s in self.summary(order).items()
        ]
