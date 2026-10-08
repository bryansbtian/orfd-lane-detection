"""Tests for capture signatures, timing and benchmarks."""

import numpy as np
import pytest

from offroad_autonomy.runtime.benchmark import BenchmarkRecorder, footprint_strip
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.simulation.beamng_client import frame_signature
from offroad_autonomy.types import PathPlan


def test_connection_failure_exits_cleanly_and_disconnects(monkeypatch):
    from unittest.mock import Mock

    import offroad_autonomy.main as app
    from offroad_autonomy.simulation.beamng_client import BeamNGConnectionError
    from offroad_autonomy.types import PipelineConfig

    client, pipeline = Mock(), Mock()
    client.connect.side_effect = BeamNGConnectionError("Set BEAMNG_HOME to start BeamNG")
    pipeline.stats = RuntimeStats()
    monkeypatch.setattr("sys.argv", ["offroad-autonomy", "--headless"])
    monkeypatch.setattr(app, "setup_logger", lambda **kwargs: None)
    error_log = Mock()
    monkeypatch.setattr(app.logger, "error", error_log)
    monkeypatch.setattr(app, "load_config", lambda path: PipelineConfig(ui_headless=True))
    monkeypatch.setattr(app, "BeamNGClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(app, "AutonomyPipeline", lambda config: pipeline)
    with pytest.raises(SystemExit) as caught:
        app.main()
    assert caught.value.code == 1
    assert "Set BEAMNG_HOME" in str(error_log.call_args.args[1])
    client.disconnect.assert_called_once()
    pipeline.step_result.assert_not_called()


def test_automatic_stop_does_not_resend_the_previous_throttle(monkeypatch):
    from unittest.mock import Mock

    import offroad_autonomy.main as app
    from offroad_autonomy.types import PipelineConfig, VehicleState
    from tests.test_dashboard import _result

    cfg = PipelineConfig(ui_headless=True)
    client, pipeline, detector = Mock(), Mock(), Mock()
    result = _result()
    result.command.throttle = 0.45
    pipeline.stats = RuntimeStats()
    pipeline.ego_coverage = 0.0
    pipeline.step_result.return_value = result
    client.capture_frame.return_value = result.capture
    client.get_vehicle_state.return_value = VehicleState()

    def stop(**kwargs):
        app._shutdown = True
        return True, "test no-road stop"

    detector.update.side_effect = stop
    monkeypatch.setattr("sys.argv", ["offroad-autonomy", "--headless"])
    monkeypatch.setattr(app, "load_config", lambda _: cfg)
    monkeypatch.setattr(app, "BeamNGClient", lambda _, **__: client)
    monkeypatch.setattr(app, "AutonomyPipeline", lambda _: pipeline)
    monkeypatch.setattr(app, "_StuckDetector", lambda **_: detector)
    monkeypatch.setattr(app.signal, "signal", Mock())
    app.main()
    client.park.assert_called_once()
    applied = client.send_controls.call_args.args[0]
    assert applied.throttle == 0 and applied.parkingbrake == 1


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
        )

    report = recorder.report(RuntimeStats())
    assert report["lane_departures"] == 1
    assert report["frames"] == 30
    assert report["primary_latency_p95_ms"] == pytest.approx(30.0)
    assert report["meets_50ms"] is True
    assert "stereo" not in report
    assert "depth_coverage_pct_mean" not in report
    assert report["path_jitter_pct"] == pytest.approx(0.0)
