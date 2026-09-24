"""Tests for the stitched wide view, the perception view and the dashboard."""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.stitching import WideViewStitcher
from offroad_autonomy.types import (
    DEBUG_VIEWS,
    DepthResult,
    EgoMaskSpec,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    StabilizedResult,
    StereoFramePair,
    StereoRigSpec,
    ControlCommand,
    stereo_camera_spec,
)
from offroad_autonomy.visualization.dashboard import AutonomyDashboard, DashboardTelemetry


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(model_weights="dummy.pt", **overrides)


def _toed_config(toe: float) -> PipelineConfig:
    config = _config()
    rig = StereoRigSpec(toe_out_deg=toe)
    sensor = config.left_camera.sensor
    return replace(
        config,
        stereo_rig=rig,
        left_camera=stereo_camera_spec(rig, "left", "l", sensor, config.left_camera.ego_mask),
        right_camera=stereo_camera_spec(rig, "right", "r", sensor, config.right_camera.ego_mask),
    )


def test_parallel_pair_stitches_to_roughly_one_camera_width():
    """Honest about the geometry: parallel cameras add almost no width."""
    stitcher = WideViewStitcher.from_config(_config())

    assert stitcher.camera.height == 465
    assert 720 <= stitcher.camera.width <= 730


def test_toe_out_widens_the_stitched_view():
    narrow = WideViewStitcher.from_config(_toed_config(0.0))
    wide = WideViewStitcher.from_config(_toed_config(10.0))

    assert wide.camera.width > narrow.camera.width + 100
    assert wide.camera.horizontal_fov_deg > narrow.camera.horizontal_fov_deg + 15


def test_stitch_takes_each_half_from_its_own_camera():
    config = _toed_config(10.0)
    stitcher = WideViewStitcher.from_config(config)
    left = np.full((620, 960, 3), 50, dtype=np.uint8)
    right = np.full((620, 960, 3), 200, dtype=np.uint8)

    wide = stitcher.stitch(left, right)

    row = wide[wide.shape[0] // 2]
    assert row[10].mean() == pytest.approx(50, abs=1)
    assert row[-10].mean() == pytest.approx(200, abs=1)


def test_stitched_ego_mask_comes_from_both_cameras():
    config = _config(segmentation_mode="stitched")
    view = PerceptionView(config)

    assert view.valid_roi.shape == (view.camera.height, view.camera.width)
    excluded = ~view.valid_roi
    assert excluded[-1].any()  # hood across the bottom
    assert not excluded[0].any()


def test_single_camera_view_uses_that_cameras_mask():
    left_view = PerceptionView(_config(segmentation_mode="left"))
    right_view = PerceptionView(_config(segmentation_mode="right"))

    assert left_view.size == (720, 465)
    # Both bumper cameras are clear of the vehicle, so both ROIs are whole.
    assert left_view.valid_roi.all() and right_view.valid_roi.all()


def test_disabled_ego_masks_leave_the_view_valid():
    config = _config()
    config = replace(
        config,
        left_camera=replace(config.left_camera, ego_mask=EgoMaskSpec()),
    )

    assert PerceptionView(config).valid_roi.all()


def _result() -> PipelineStepResult:
    frame = np.random.default_rng(0).integers(0, 255, (465, 720, 3), dtype=np.uint8)
    mask = np.zeros((465, 720), dtype=bool)
    mask[250:400, 250:470] = True
    depth = DepthResult(
        depth_m=np.full((465, 720), 10.0, np.float32),
        valid=mask.copy(),
        points_vehicle=np.zeros((465, 720, 3), np.float32),
        cloud_vehicle=np.zeros((0, 3), np.float32),
        disparity=np.full((465, 720), 12.0, np.float32),
        depth_rect=np.full((465, 720), 10.0, np.float32),
        rectified_left=frame,
        rectified_right=frame,
        coverage=0.5,
        valid_disparity_fraction=0.6,
        median_forward_depth_m=9.0,
        min_corridor_depth_m=4.0,
    )
    raw = np.zeros((620, 960, 3), dtype=np.uint8)
    return PipelineStepResult(
        frame=FramePacket(raw=raw, preprocessed=frame, timestamp=0.0, height=465, width=720),
        perception=PerceptionResult(mask=mask, confidences=[0.8]),
        stabilized=StabilizedResult(mask=mask),
        plan=PathPlan(centerline=np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]])),
        command=ControlCommand(),
        frames=StereoFramePair(left=raw, right=raw),
        depth=depth,
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
        depth_active=True,
        depth_state="LIVE",
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
        stitched=result.frame.preprocessed,
        depth_roi=roi,
    )

    assert frame.shape == (900, 1600, 3)


def test_dashboard_survives_missing_stereo():
    dashboard = AutonomyDashboard()
    result = replace(_result(), depth=None, frames=StereoFramePair(left=None, right=None))
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
