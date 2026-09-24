"""Tests for the stereo worker, pair synchronisation, timing and benchmarks."""

import threading
import time

import numpy as np
import pytest

from offroad_autonomy.runtime.benchmark import BenchmarkRecorder, footprint_strip
from offroad_autonomy.runtime.stereo_worker import StereoJob, StereoWorker
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.simulation.beamng_client import frame_signature, judge_pair_sync
from offroad_autonomy.types import (
    DepthResult,
    PathPlan,
    StereoFramePair,
    StereoGeometry,
)


def _job(frame_id: int) -> StereoJob:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    return StereoJob(
        pair=StereoFramePair(
            left=frame, right=frame, frame_id=frame_id, timestamp=time.perf_counter()
        )
    )


def _geometry(job: StereoJob) -> StereoGeometry:
    depth = DepthResult(
        depth_m=np.zeros((2, 2), np.float32),
        valid=np.zeros((2, 2), bool),
        points_vehicle=np.zeros((2, 2, 3), np.float32),
        cloud_vehicle=np.zeros((0, 3), np.float32),
        timings_ms={"stereo_matching": 1.0},
    )
    return StereoGeometry(
        depth=depth,
        terrain=None,
        frame_id=job.pair.frame_id,
        capture_time=job.pair.timestamp,
        completed_time=time.perf_counter(),
    )


def _wait_for(predicate, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(0.005)
    return predicate()


def test_submit_returns_immediately_while_the_worker_is_busy():
    release = threading.Event()

    def blocking(job):
        release.wait(2.0)
        return _geometry(job)

    worker = StereoWorker(blocking, rate_hz=0.0)
    worker.start()
    try:
        worker.submit(_job(1))
        t0 = time.perf_counter()
        for frame_id in range(2, 50):
            worker.submit(_job(frame_id))
        assert time.perf_counter() - t0 < 0.05
    finally:
        release.set()
        worker.stop()


def test_worker_keeps_only_the_newest_pending_pair():
    """Latest-frame semantics: a busy worker skips to the freshest pair."""
    release = threading.Event()
    started = threading.Event()
    seen: list[int] = []

    def compute(job):
        seen.append(job.pair.frame_id)
        started.set()
        release.wait(2.0)
        return _geometry(job)

    worker = StereoWorker(compute, rate_hz=0.0)
    worker.start()
    try:
        worker.submit(_job(1))
        assert started.wait(2.0)
        for frame_id in range(2, 11):
            worker.submit(_job(frame_id))
        release.set()
        assert _wait_for(lambda: worker.completed >= 2)
        assert seen == [1, 10]
        assert worker.dropped == 8
        assert worker.latest().frame_id == 10
    finally:
        release.set()
        worker.stop()


def test_worker_respects_the_rate_cap():
    starts: list[float] = []

    def compute(job):
        starts.append(time.perf_counter())
        return _geometry(job)

    worker = StereoWorker(compute, rate_hz=20.0)
    worker.start()
    try:
        deadline = time.perf_counter() + 0.5
        frame_id = 0
        while time.perf_counter() < deadline:
            frame_id += 1
            worker.submit(_job(frame_id))
            time.sleep(0.002)
    finally:
        worker.stop()

    gaps = np.diff(starts)
    assert len(starts) <= 12  # ~10 in 0.5 s at 20 Hz, never ~250
    assert gaps.min() >= 0.045


def test_a_failing_frame_does_not_kill_the_worker():
    calls = {"n": 0}

    def flaky(job):
        calls["n"] += 1
        if job.pair.frame_id == 1:
            raise RuntimeError("bad frame")
        return _geometry(job)

    worker = StereoWorker(flaky, rate_hz=0.0)
    worker.start()
    try:
        worker.submit(_job(1))
        assert _wait_for(lambda: worker.failed == 1)
        worker.submit(_job(2))
        assert _wait_for(lambda: worker.latest() is not None)
        assert worker.latest().frame_id == 2
    finally:
        worker.stop()


def test_inline_mode_runs_on_the_caller_thread():
    threads: list[str] = []

    def compute(job):
        threads.append(threading.current_thread().name)
        return _geometry(job)

    worker = StereoWorker(compute, rate_hz=0.0, asynchronous=False)
    worker.submit(_job(1))

    assert threads == [threading.current_thread().name]
    assert worker.latest().frame_id == 1
    assert worker.stats.stage("stereo_total").count == 1


def _judge(before, after, previous, skew=1.0, both_new=True):
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    accepted = []
    pair = judge_pair_sync(
        frame,
        frame,
        before=before,
        after=after,
        previous=previous,
        read_skew_ms=skew,
        max_skew_ms=8.0,
        require_both_new=both_new,
        frame_id=5,
        timestamp=1.0,
        on_accept=accepted.append,
    )
    return pair, accepted


def test_pair_from_one_render_is_synchronised():
    pair, accepted = _judge(before=(1, 2), after=(1, 2), previous=(10, 20))

    assert pair.synchronized and pair.is_new and pair.has_stereo
    assert accepted == [(1, 2)]


def test_frame_published_mid_read_is_rejected():
    pair, accepted = _judge(before=(1, 2), after=(3, 2), previous=(10, 20))

    assert not pair.synchronized
    assert pair.sync_note == "changed during read"
    assert not accepted


def test_only_one_camera_updating_is_rejected():
    pair, _ = _judge(before=(1, 20), after=(1, 20), previous=(10, 20))

    assert not pair.synchronized
    assert pair.sync_note == "only left updated"


def test_slow_read_is_rejected():
    pair, _ = _judge(before=(1, 2), after=(1, 2), previous=(10, 20), skew=20.0)

    assert not pair.synchronized
    assert "skew" in pair.sync_note


def test_unchanged_pair_is_synchronised_but_not_new():
    pair, _ = _judge(before=(1, 2), after=(1, 2), previous=(1, 2))

    assert pair.synchronized
    assert not pair.is_new


def test_frame_signature_detects_a_new_render():
    buffer = bytearray(np.random.default_rng(0).integers(0, 255, 960 * 620 * 4, dtype=np.uint8))
    before = frame_signature(buffer)
    buffer[997 * 10] ^= 0xFF

    assert frame_signature(buffer) != before
    assert frame_signature(None) is None


def test_runtime_stats_mean_p95_and_rate():
    stats = RuntimeStats(window=100)
    for value in range(1, 101):
        stats.record("stage", float(value))
    for tick in range(11):
        stats.tick(now=tick * 0.05)

    summary = stats.stage("stage")
    assert summary.mean_ms == pytest.approx(50.5)
    assert summary.p95_ms == pytest.approx(95.05)
    assert stats.fps() == pytest.approx(20.0)


def test_runtime_stats_window_is_bounded():
    stats = RuntimeStats(window=10)
    for value in range(1000):
        stats.record("stage", float(value))

    assert stats.stage("stage").count == 10
    assert stats.stage("stage").mean_ms == pytest.approx(994.5)


def test_footprint_strip_sits_just_above_the_hood():
    roi = np.ones((100, 200), dtype=bool)
    roi[80:, 60:140] = False  # hood

    strip = footprint_strip(roi)

    rows = np.flatnonzero(strip.any(axis=1))
    assert rows.max() == 79
    assert not (strip & ~roi).any()


def test_benchmark_counts_departures_and_reports_targets():
    roi = np.ones((100, 200), dtype=bool)
    roi[80:, 60:140] = False
    recorder = BenchmarkRecorder("t", "cfg.yaml", roi, warmup_s=0.0, departure_frames=3)
    road = np.zeros_like(roi)
    road[:, 50:150] = True
    plan = PathPlan(centerline=np.array([[100.0, 90.0], [100.0, 50.0], [100.0, 10.0]]))

    for step in range(30):
        mask = np.zeros_like(road)
        if (step // 10) % 2 == 0:
            mask = road
        recorder.add(
            now=step * 0.05,
            primary_ms=30.0,
            full_ms=40.0,
            confidence=0.8,
            mask=mask,
            plan=plan,
            valid_disparity=0.6,
            depth_coverage=0.5,
            depth_used=True,
        )

    report = recorder.report(RuntimeStats(), None)
    assert report["lane_departures"] == 1
    assert report["frames"] == 30
    assert report["primary_latency_p95_ms"] == pytest.approx(30.0)
    assert report["meets_50ms"] is True
    assert report["valid_disparity_pct_mean"] == pytest.approx(60.0)
    assert report["path_jitter_pct"] == pytest.approx(0.0)
