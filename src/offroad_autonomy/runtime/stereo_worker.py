"""Stereo depth off the control loop, with latest-frame semantics.

The control loop must never wait for stereo. It hands each new synchronised
pair to ``submit`` - which returns immediately - and reads whatever
``latest`` holds when it plans.

There is exactly one pending slot. A pair submitted while the worker is busy
replaces the one waiting, so the queue can never grow and the worker always
starts on the freshest frames. At most three pairs exist at once: the one
being matched, the one waiting, and the one inside the published result.

``rate_hz`` caps how often a new match starts. Pairs arriving in between
simply overwrite the pending slot, so a slow rate costs freshness, not
memory or latency.

With ``asynchronous=False`` the same object runs the work inline on the
caller's thread (still rate-limited). That is the mode for unit tests and
for measuring what synchronous stereo would cost the loop.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import StereoFramePair, StereoGeometry

logger = logging.getLogger("offroad_autonomy.runtime.stereo")


@dataclass
class StereoJob:
    """Captured at submit time so the worker never reads loop state that is
    changing underneath it."""

    pair: StereoFramePair
    road_mask: np.ndarray | None = None
    valid_roi: np.ndarray | None = None


class StereoWorker:
    def __init__(
        self,
        compute: Callable[[StereoJob], StereoGeometry | None],
        rate_hz: float = 10.0,
        asynchronous: bool = True,
        stats: RuntimeStats | None = None,
    ) -> None:
        self._compute = compute
        self._period = 0.0
        if rate_hz > 0.0:
            self._period = 1.0 / rate_hz
        self._async = bool(asynchronous)
        self._cond = threading.Condition()
        self._pending: StereoJob | None = None
        self._latest: StereoGeometry | None = None
        self._stop = False
        self._thread: threading.Thread | None = None
        self._last_start = -math.inf
        self._last_error_log = -math.inf
        if stats is None:
            stats = RuntimeStats()
        self.stats = stats

        self.submitted = 0
        self.dropped = 0
        self.completed = 0
        self.failed = 0

    @property
    def asynchronous(self) -> bool:
        return self._async

    def start(self) -> None:
        if not self._async or self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="stereo-worker", daemon=True)
        self._thread.start()
        logger.info("Stereo worker started (period %.3f s, 0 = unthrottled)", self._period)

    def stop(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def submit(self, job: StereoJob) -> bool:
        """Never blocks in async mode; a waiting job is replaced, not queued,
        so the worker always starts on the freshest frames."""
        if not self._async:
            now = time.perf_counter()
            if now - self._last_start < self._period:
                return False
            self._last_start = now
            self.submitted += 1
            self._execute(job)
            return True

        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = job
            self.submitted += 1
            self._cond.notify()
        return True

    def latest(self) -> StereoGeometry | None:
        with self._cond:
            return self._latest

    def reset(self) -> None:
        with self._cond:
            self._pending = None
            self._latest = None

    @staticmethod
    def _lower_thread_priority() -> None:
        """Stereo never blocks the loop, but its numpy work still competes for
        cores with segmentation's CPU-side work; on a Jetson's few cores that
        contention is real. Lower priority makes it one-sided. Best effort:
        OpenCV's own pool threads (SGBM) are unaffected.
        """
        try:
            if sys.platform == "win32":
                import ctypes

                kernel32 = ctypes.windll.kernel32
                thread_priority_below_normal = -1
                kernel32.SetThreadPriority(
                    kernel32.GetCurrentThread(), thread_priority_below_normal
                )
            elif sys.platform.startswith("linux"):
                # Linux threads are schedulable tasks, so this renices only
                # the worker thread, not the whole process.
                os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 5)
        except (OSError, AttributeError) as exc:
            logger.debug("Could not lower stereo worker priority: %s", exc)

    def _run(self) -> None:
        self._lower_thread_priority()
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                # Waiting on the condition, not sleeping, lets newer pairs
                # replace the pending one meanwhile.
                remaining = self._last_start + self._period - time.perf_counter()
                while remaining > 0.0 and not self._stop:
                    self._cond.wait(remaining)
                    remaining = self._last_start + self._period - time.perf_counter()
                if self._stop:
                    return
                job = self._pending
                self._pending = None
                self._last_start = time.perf_counter()
            if job is not None:
                self._execute(job)

    def _execute(self, job: StereoJob) -> None:
        t0 = time.perf_counter()
        try:
            result = self._compute(job)
        except Exception:
            self.failed += 1
            if t0 - self._last_error_log > 5.0:
                logger.exception("Stereo computation failed (frame %d)", job.pair.frame_id)
                self._last_error_log = t0
            return

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            return
        result.latency_ms = elapsed_ms
        self.stats.record("stereo_total", elapsed_ms)
        self.stats.record_many(result.depth.timings_ms)
        self.stats.tick()
        with self._cond:
            self._latest = result
            self.completed += 1
