"""Render completed dashcam inference snapshots without blocking control."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import PathPlan, PipelineStepResult

logger = logging.getLogger("offroad_autonomy.runtime.display")


@dataclass
class DisplayState:
    """Everything the dashboard draws, as of one control-loop iteration.

    This is a *snapshot to read*, never a channel back into the stack: the
    worker consumes it and produces pixels. The arrays inside are the ones
    the loop has already finished with, and each iteration produces fresh
    ones, so the worker can hold a reference without copying. A redraw that
    straddles two iterations shows a one-frame-old overlay, which is a
    cosmetic outcome and the reason no lock is held while drawing.
    """

    result: PipelineStepResult
    telemetry: object
    plan: PathPlan | None = None
    valid_roi: np.ndarray | None = None
    debug_view: str = "default"
    timing_overlay: bool = False
    #: The orbit camera's latest image, for the presentation video only.
    orbit: np.ndarray | None = None


class DisplayWorker:
    def __init__(
        self,
        render: Callable[[DisplayState], np.ndarray],
        show: Callable[[np.ndarray], bool],
        read_key: Callable[[], int],
        record: Callable[[np.ndarray, float], None] | None = None,
        rate_hz: float = 20.0,
        asynchronous: bool = True,
        stats: RuntimeStats | None = None,
        open_window: Callable[[], None] | None = None,
        close_window: Callable[[], None] | None = None,
    ) -> None:
        self._render = render
        self._open_window = open_window
        self._close_window = close_window
        self._window_open = False
        self._unavailable = False
        self._show = show
        self._read_key = read_key
        self._record = record
        self._period = 0.0
        if rate_hz > 0.0:
            self._period = 1.0 / rate_hz
        self._async = bool(asynchronous)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: DisplayState | None = None
        self._keys: deque[int] = deque(maxlen=32)
        self._stop = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._last_draw_time = 0.0
        if stats is None:
            stats = RuntimeStats()
        self.stats = stats
        self.rendered = 0
        self.dropped = 0
        self.failed = 0
        self._last_error_log = 0.0

    @property
    def asynchronous(self) -> bool:
        return self._async

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def unavailable(self) -> bool:
        with self._lock:
            return self._unavailable

    def fps(self) -> float:
        return self.stats.fps()

    def start(self) -> None:
        if not self._async:
            self._open()
            return
        if self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="display-worker", daemon=True)
        self._thread.start()
        logger.info("Dashboard worker started (period %.3f s, 0 = unthrottled)", self._period)

    def stop(self, timeout: float = 2.0) -> None:
        if not self._async:
            self._close()
            return
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning("Dashboard worker did not stop within %.1f s", timeout)
                return
            self._thread = None

    def publish(self, state: DisplayState) -> None:
        """Overwrites rather than queues, so a slow dashboard goes stale
        instead of building latency."""
        if self.unavailable:
            return
        if not self._async:
            self._draw(state)
            return
        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = state
            self._cond.notify()

    def drain_keys(self) -> list[int]:
        with self._lock:
            keys = list(self._keys)
            self._keys.clear()
        return keys

    def _open(self) -> bool:
        if self._open_window is None:
            return True
        try:
            self._open_window()
        except Exception:
            logger.exception("Dashboard window could not be opened - continuing without it")
            with self._lock:
                self._unavailable = True
            return False
        self._window_open = True
        return True

    def _close(self) -> None:
        if not self._window_open or self._close_window is None:
            return
        self._window_open = False
        try:
            self._close_window()
        except Exception:
            logger.exception("Dashboard window did not close cleanly")

    def _run(self) -> None:
        if not self._open():
            return
        try:
            self._loop()
        finally:
            self._close()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                remaining = self._period - (time.perf_counter() - self._last_draw_time)
                if remaining > 0.0:
                    # New snapshots replace pending work without waking the
                    # rate limiter early and stealing extra CPU from inference.
                    self._cond.wait_for(lambda: self._stop, timeout=remaining)
                if self._stop:
                    return
                state = self._pending
                self._pending = None
            if state is not None:
                self._draw(state)
                if self.closed:
                    return

    def _draw(self, state: DisplayState) -> None:
        t0 = time.perf_counter()
        self._last_draw_time = t0
        try:
            canvas = self._render(state)
            t2 = time.perf_counter()
            alive = self._show(canvas)
            t3 = time.perf_counter()
        except Exception:
            self.failed += 1
            if t0 - self._last_error_log > 5.0:
                logger.exception("Dashboard rendering failed")
                self._last_error_log = t0
            return

        t4 = t3
        if self._record is not None:
            try:
                self._record(canvas, t0)
            except Exception:
                # A recording fault will not fix itself, and retrying it every
                # frame would flood the log while the dashboard keeps working.
                logger.exception("Video recording failed; recording disabled")
                self._record = None
            t4 = time.perf_counter()
            self.stats.record("dashboard_record", (t4 - t3) * 1000.0)

        self.stats.record("dashboard_render", (t2 - t0) * 1000.0)
        self.stats.record("dashboard_show", (t3 - t2) * 1000.0)
        self.stats.record("dashboard_total", (t4 - t0) * 1000.0)
        self.stats.tick()
        self.rendered += 1

        key = self._read_key()
        with self._lock:
            if key is not None and key >= 0:
                self._keys.append(int(key))
            if not alive:
                self._closed = True
