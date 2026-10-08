"""Tests for the dashcam dashboard."""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.types import (
    DEBUG_VIEWS,
    DEFAULT_DASHBOARD_COLORS,
    CameraFrame,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineStepResult,
    StabilizedResult,
    SteeringDebug,
)
from offroad_autonomy.visualization.dashboard import AutonomyDashboard, DashboardTelemetry


def _result() -> PipelineStepResult:
    frame = np.random.default_rng(0).integers(0, 255, (465, 720, 3), dtype=np.uint8)
    mask = np.zeros((465, 720), dtype=bool)
    mask[250:400, 250:470] = True
    raw = np.zeros((620, 960, 3), dtype=np.uint8)
    return PipelineStepResult(
        frame=FramePacket(raw=raw, preprocessed=frame, timestamp=0.0, height=465, width=720),
        perception=PerceptionResult(mask=mask, confidences=[0.8]),
        stabilized=StabilizedResult(mask=mask),
        plan=PathPlan(centerline=np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]])),
        command=ControlCommand(),
        capture=CameraFrame(image=raw),
    )


@pytest.mark.parametrize("view", DEBUG_VIEWS)
def test_every_debug_view_renders(view):
    dashboard = AutonomyDashboard()
    result = _result()
    telemetry = DashboardTelemetry(
        speed_mph=10.0,
        steering=0.1,
        throttle=0.2,
        brake=0.0,
        perception_confidence=0.8,
        stability_score=0.9,
        kalman_active=False,
        fps=22.0,
        latency_ms=35.0,
        latency_p95_ms=44.0,
        timing_lines=["capture  1.0 ms"],
    )
    roi = np.ones((465, 720), dtype=bool)
    roi[420:] = False

    frame = dashboard.render(
        result,
        telemetry,
        plan=result.plan,
        valid_roi=roi,
        debug_view=view,
        timing_overlay=True,
    )

    assert frame.shape == (900, 1600, 3)


def test_dashboard_survives_missing_camera_frames():
    dashboard = AutonomyDashboard()
    result = replace(_result(), capture=None)
    telemetry = DashboardTelemetry(
        speed_mph=0.0,
        steering=0.0,
        throttle=0.0,
        brake=0.0,
        perception_confidence=0.0,
        stability_score=1.0,
        kalman_active=True,
        fps=0.0,
        latency_ms=0.0,
        autopilot_active=False,
    )

    for view in DEBUG_VIEWS:
        frame = dashboard.render(result, telemetry, debug_view=view)
        assert frame.shape == (900, 1600, 3)


def test_raw_view_is_the_frame_used_for_inference():
    result = _result()
    image = AutonomyDashboard()._main_view("raw", result, None, None)
    assert image is result.frame.raw


def _telemetry(**overrides) -> DashboardTelemetry:
    fields = {
        "speed_mph": 9.4,
        "steering": 0.14,
        "throttle": 0.3,
        "brake": 0.0,
        "perception_confidence": 0.7,
        "stability_score": 1.0,
        "kalman_active": False,
        "fps": 8.0,
        "latency_ms": 90.0,
        "latency_p95_ms": 110.0,
    }
    fields.update(overrides)
    return DashboardTelemetry(**fields)


def _flat_result(plan: PathPlan | None = None) -> PipelineStepResult:
    """A grey frame, so any pixel in a palette colour was drawn by the dashboard."""
    result = _result()
    flat = np.full((465, 720, 3), 90, dtype=np.uint8)
    frame = replace(result.frame, preprocessed=flat)
    mask = np.zeros((465, 720), dtype=bool)
    result = replace(
        result,
        frame=frame,
        perception=PerceptionResult(mask=mask, confidences=[0.8]),
        stabilized=StabilizedResult(mask=mask),
    )
    if plan is not None:
        result = replace(result, plan=plan)
    return result


def _count_color(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int((image == np.array(color, dtype=np.uint8)).all(axis=2).sum())


def test_safe_stop_draws_no_path_and_shows_the_banner():
    dashboard = AutonomyDashboard()
    result = _flat_result()
    path_color = DEFAULT_DASHBOARD_COLORS["PATH_CORE"]

    driving = dashboard.render(result, _telemetry(), plan=result.plan)
    stopped = dashboard.render(result, _telemetry(autopilot_active=False), plan=None)

    assert _count_color(driving, path_color) > 0
    assert _count_color(stopped, path_color) == 0
    # The banner border is drawn in the fault colour inside the viewport.
    assert _count_color(stopped, DEFAULT_DASHBOARD_COLORS["BAD"]) > 0


def test_held_path_is_dashed_in_the_warning_colour():
    dashboard = AutonomyDashboard()
    plan = replace(_result().plan, fallback_active=True, fallback_reason="holding")
    result = _flat_result(plan)

    frame = dashboard.render(result, _telemetry(), plan=plan)

    assert _count_color(frame, DEFAULT_DASHBOARD_COLORS["WARN"]) > 0
    assert _count_color(frame, DEFAULT_DASHBOARD_COLORS["PATH_CORE"]) == 0


def test_runtime_tile_border_follows_the_fps_band():
    dashboard = AutonomyDashboard()
    result = _flat_result()
    tile = dashboard._layout.autonomy_fps
    probe = (tile.y, tile.x + 40)

    on_target = dashboard.render(result, _telemetry(fps=7.0))
    amber = dashboard.render(result, _telemetry(fps=6.0))
    red = dashboard.render(result, _telemetry(fps=3.0))

    assert tuple(on_target[probe]) == DEFAULT_DASHBOARD_COLORS["CARD_BORDER"]
    assert tuple(amber[probe]) == DEFAULT_DASHBOARD_COLORS["WARN"]
    assert tuple(red[probe]) == DEFAULT_DASHBOARD_COLORS["BAD"]


def test_dashboard_fps_never_takes_a_health_colour():
    dashboard = AutonomyDashboard()
    result = _flat_result()
    tile = dashboard._layout.dashboard_fps

    slow = dashboard.render(result, _telemetry(dashboard_fps=1.0))

    region = slow[tile.y : tile.bottom, tile.x : tile.right]
    for name in ("BAD", "WARN"):
        assert _count_color(region, DEFAULT_DASHBOARD_COLORS[name]) == 0


def test_long_reason_and_chained_state_stay_inside_the_perception_card():
    dashboard = AutonomyDashboard()
    result = _flat_result()
    card = dashboard._layout.perception
    quiet = dashboard.render(result, _telemetry())

    loud = dashboard.render(
        result,
        _telemetry(
            fallback_state="SAFE STOP + GATE HOLD + KALMAN",
            fallback_reason="word " * 200,
        ),
    )

    outside = np.ones(quiet.shape[:2], dtype=bool)
    outside[card.y : card.bottom, card.x : card.right] = False
    assert np.array_equal(quiet[outside], loud[outside])
    assert not np.array_equal(quiet, loud)


def test_render_does_not_modify_the_cached_static_layer():
    dashboard = AutonomyDashboard()
    result = _flat_result()
    before = dashboard._static.copy()
    first = dashboard._static

    dashboard.render(result, _telemetry(autopilot_active=False))
    dashboard.render(result, _telemetry(fps=2.0), debug_view="pipeline")

    assert dashboard._static is first
    assert np.array_equal(dashboard._static, before)


def test_ego_exclusion_colour_is_distinct_from_mask_and_fault_colours():
    ego = np.array(DEFAULT_DASHBOARD_COLORS["EGO_EXCLUDED"], dtype=float)
    for name in ("MASK_FILL", "BAD", "GOOD", "WARN", "PATH_CORE"):
        other = np.array(DEFAULT_DASHBOARD_COLORS[name], dtype=float)
        assert np.linalg.norm(ego - other) > 100.0, name


def test_pipeline_readout_lists_controller_numbers_as_rows():
    dashboard = AutonomyDashboard()
    result = _result()
    result.command.debug = SteeringDebug(cross_track_m=0.25)

    rows = dict(dashboard._pipeline_readout(result))

    assert rows["Cross-Track"].startswith("+0.25 m")
    assert rows["Controller"] is None
