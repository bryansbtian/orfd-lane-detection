"""Unit tests for the pipeline with mocked stages."""

import time
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from offroad_autonomy.types import (
    ControlCommand,
    DepthResult,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    StabilizedResult,
    StereoFramePair,
    TerrainAnalysis,
    VehicleState,
)

H, W = 360, 640


def _make_config(**overrides) -> PipelineConfig:
    # Inline stereo keeps these tests deterministic; the async worker has
    # its own tests below and in test_stereo_worker.py.
    overrides.setdefault("stereo_async", False)
    overrides.setdefault("stereo_rate_hz", 0.0)
    return PipelineConfig(
        model_weights="dummy.pt",
        preprocess_width=W,
        preprocess_height=H,
        **overrides,
    )


_STAGE_PATCHES = (
    "offroad_autonomy.pipeline.ImagePreprocessor",
    "offroad_autonomy.pipeline.RoadSegmenter",
    "offroad_autonomy.pipeline.TemporalStabilizer",
    "offroad_autonomy.pipeline.CenterlinePlanner",
    "offroad_autonomy.pipeline.StanleyController",
)


def _stage_outputs():
    """Canned outputs for every mocked stage, shaped like the real ones."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[200:300, 200:440] = True

    return (
        FramePacket(raw=frame, preprocessed=frame, timestamp=0.0, height=H, width=W),
        PerceptionResult(mask=mask, confidences=[0.9], num_detections=1),
        StabilizedResult(mask=mask, stability_score=0.95),
        PathPlan(centerline=np.array([[320.0, 300.0], [320.0, 100.0]])),
        ControlCommand(steering=0.0, throttle=0.3, brake=0.0),
    )


def _wire(mocks):
    """Point the five mocked stages at the canned outputs."""
    packet, perception, stabilized, plan, command = _stage_outputs()
    pre, seg, stab, planner, ctrl = mocks
    pre.return_value.process.return_value = packet
    seg.return_value.predict.return_value = perception
    stab.return_value.stabilize.return_value = stabilized
    planner.return_value.plan.return_value = plan
    ctrl.return_value.compute.return_value = command
    return packet, perception, stabilized, plan, command


def _pair(**overrides) -> StereoFramePair:
    frame = np.zeros((620, 960, 3), dtype=np.uint8)
    fields = dict(
        left=frame,
        right=frame.copy(),
        timestamp=time.perf_counter(),
        frame_id=1,
        synchronized=True,
        is_new=True,
    )
    fields.update(overrides)
    return StereoFramePair(**fields)


def _depth_result() -> DepthResult:
    return DepthResult(
        depth_m=np.ones((H, W), dtype=np.float32),
        valid=np.ones((H, W), dtype=bool),
        points_vehicle=np.zeros((H, W, 3), dtype=np.float32),
        cloud_vehicle=np.zeros((10, 3), dtype=np.float32),
    )


def _terrain(clearance: float = 5.0) -> TerrainAnalysis:
    zeros = np.zeros((H, W), dtype=np.float32)
    return TerrainAnalysis(
        height_above_ground=zeros,
        slope_rad=zeros,
        obstacle_mask=np.zeros((H, W), dtype=bool),
        clearance_m=zeros,
        traversability=zeros,
        valid=np.zeros((H, W), dtype=bool),
        min_forward_clearance_m=clearance,
    )


def _pipeline_with_stereo(stack, config, depth=None, terrain=None):
    """Build a pipeline with mocked stages and mocked stereo/terrain."""
    mocks = [stack.enter_context(patch(target)) for target in _STAGE_PATCHES]
    MockDepth = stack.enter_context(patch("offroad_autonomy.pipeline.StereoDepthEstimator"))
    MockTerrain = stack.enter_context(patch("offroad_autonomy.pipeline.TerrainAnalyzer"))
    outputs = _wire(mocks)
    if depth is None:
        depth = _depth_result()
    if terrain is None:
        terrain = _terrain()
    MockDepth.return_value.compute.return_value = depth
    MockTerrain.return_value.analyze.return_value = terrain

    from offroad_autonomy.pipeline import AutonomyPipeline

    return AutonomyPipeline(config), mocks, MockDepth, MockTerrain, outputs


def test_pipeline_step_produces_control_command():
    with ExitStack() as stack:
        pipeline, mocks, _, _, outputs = _pipeline_with_stereo(stack, _make_config())
        command = pipeline.step(np.zeros((620, 960, 3), dtype=np.uint8), VehicleState())

        assert isinstance(command, ControlCommand)
        assert command.throttle == 0.3
        for mock in mocks:
            assert mock.return_value.method_calls


def test_pipeline_step_result_exposes_stage_outputs():
    with ExitStack() as stack:
        pipeline, _, _, _, outputs = _pipeline_with_stereo(stack, _make_config())
        packet, perception, stabilized, plan, command = outputs

        result = pipeline.step_result(_pair(), VehicleState())

        assert isinstance(result, PipelineStepResult)
        assert result.frame is packet
        assert result.stabilized is stabilized
        assert result.plan is plan
        assert result.command is command
        assert "segmentation" in result.timings_ms
        assert "planning" in result.timings_ms


def test_segmentation_runs_on_the_configured_camera():
    """Mode 'right' must hand the right frame - not the left - to perception."""
    with ExitStack() as stack:
        pipeline, mocks, _, _, _ = _pipeline_with_stereo(
            stack, _make_config(segmentation_mode="right", depth_enabled=False)
        )
        left = np.zeros((620, 960, 3), dtype=np.uint8)
        right = np.full((620, 960, 3), 7, dtype=np.uint8)

        pipeline.step_result(_pair(left=left, right=right), VehicleState())

        assert mocks[0].return_value.process.call_args.args[0] is right


def test_pipeline_runs_depth_when_a_synchronised_pair_is_present():
    config = _make_config()
    with ExitStack() as stack:
        mock_fuse = stack.enter_context(patch("offroad_autonomy.pipeline.fuse_rgb_depth"))
        pipeline, mocks, MockDepth, MockTerrain, outputs = _pipeline_with_stereo(stack, config)
        mock_fuse.return_value = outputs[1]

        result = pipeline.step_result(_pair(), VehicleState())

        MockDepth.return_value.compute.assert_called_once()
        MockTerrain.return_value.analyze.assert_called_once()
        assert MockTerrain.return_value.analyze.call_args.args[0] is result.depth
        mock_fuse.assert_called_once()
        assert result.depth_fallback == ""
        assert result.traversability is not None
        # The planner must be told about the terrain it is planning over.
        planner = mocks[3]
        assert planner.return_value.plan.call_args.kwargs["terrain"] is result.terrain


def test_stereo_receives_the_original_frames_not_a_derivative():
    with ExitStack() as stack:
        pipeline, _, MockDepth, _, _ = _pipeline_with_stereo(stack, _make_config())
        pair = _pair()

        pipeline.step_result(pair, VehicleState())

        args = MockDepth.return_value.compute.call_args.args
        assert args[0] is pair.left
        assert args[1] is pair.right


def test_unsynchronised_pair_skips_stereo_but_not_control():
    with ExitStack() as stack:
        pipeline, _, MockDepth, MockTerrain, outputs = _pipeline_with_stereo(stack, _make_config())

        result = pipeline.step_result(
            _pair(synchronized=False, sync_note="only left updated"), VehicleState()
        )

        MockDepth.return_value.compute.assert_not_called()
        MockTerrain.return_value.analyze.assert_not_called()
        assert result.depth is None
        assert result.depth_fallback == "warming up"
        assert result.command is outputs[4]


def test_missing_right_frame_still_segments_the_left():
    with ExitStack() as stack:
        pipeline, _, MockDepth, _, outputs = _pipeline_with_stereo(stack, _make_config())
        pair = _pair(right=None, synchronized=False, sync_note="camera missing")

        assert pipeline.has_input(pair)
        result = pipeline.step_result(pair, VehicleState())

        MockDepth.return_value.compute.assert_not_called()
        assert result.command is outputs[4]


def test_repeated_frames_are_not_matched_again():
    with ExitStack() as stack:
        pipeline, _, MockDepth, _, _ = _pipeline_with_stereo(stack, _make_config())

        pipeline.step_result(_pair(), VehicleState())
        pipeline.step_result(_pair(is_new=False), VehicleState())

        assert MockDepth.return_value.compute.call_count == 1


def test_pipeline_skips_geometry_when_stereo_returns_nothing():
    with ExitStack() as stack:
        pipeline, _, MockDepth, MockTerrain, outputs = _pipeline_with_stereo(stack, _make_config())
        MockDepth.return_value.compute.return_value = None

        result = pipeline.step_result(_pair(), VehicleState())

        MockTerrain.return_value.analyze.assert_not_called()
        assert result.terrain is None
        assert result.command is outputs[4]


def test_stale_depth_is_not_used():
    with ExitStack() as stack:
        pipeline, mocks, _, _, _ = _pipeline_with_stereo(stack, _make_config(stereo_max_age_s=0.05))
        old = _pair(timestamp=time.perf_counter() - 1.0)

        result = pipeline.step_result(old, VehicleState())

        assert result.depth is None
        assert result.depth_fallback == "stale"
        assert mocks[3].return_value.plan.call_args.kwargs["terrain"] is None


def test_clearance_is_discounted_by_distance_driven_since_capture():
    with ExitStack() as stack:
        pipeline, mocks, _, _, _ = _pipeline_with_stereo(
            stack, _make_config(stereo_max_age_s=1.0), terrain=_terrain(clearance=10.0)
        )
        pair = _pair(timestamp=time.perf_counter() - 0.2)

        result = pipeline.step_result(pair, VehicleState(speed_mps=10.0))

        planned = mocks[3].return_value.plan.call_args.kwargs["terrain"]
        # ~0.2 s at 10 m/s = ~2 m closer than the stereo frame reported.
        assert planned.min_forward_clearance_m == pytest.approx(8.0, abs=0.3)
        assert result.traversability.forward_clearance_m == pytest.approx(
            planned.min_forward_clearance_m
        )


def test_async_stereo_never_blocks_the_control_loop():
    """A slow stereo computation must not add to the loop's latency."""
    config = _make_config(stereo_async=True)
    with ExitStack() as stack:
        pipeline, _, MockDepth, _, outputs = _pipeline_with_stereo(stack, config)

        def slow_compute(*args, **kwargs):
            time.sleep(0.3)
            return _depth_result()

        MockDepth.return_value.compute.side_effect = slow_compute
        try:
            t0 = time.perf_counter()
            first = pipeline.step_result(_pair(frame_id=1), VehicleState())
            elapsed = time.perf_counter() - t0

            assert elapsed < 0.1
            assert first.depth is None
            assert first.depth_fallback == "warming up"
            assert first.command is outputs[4]

            deadline = time.perf_counter() + 3.0
            while pipeline.stereo_worker.latest() is None and time.perf_counter() < deadline:
                time.sleep(0.01)
            later = pipeline.step_result(_pair(frame_id=2), VehicleState())
            assert later.depth is not None
        finally:
            pipeline.close()


def test_depth_disabled_config_skips_the_stereo_stage_entirely():
    config = _make_config(depth_enabled=False)

    with ExitStack() as stack:
        for target in _STAGE_PATCHES:
            stack.enter_context(patch(target))
        MockDepth = stack.enter_context(patch("offroad_autonomy.pipeline.StereoDepthEstimator"))

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)

        MockDepth.assert_not_called()
        assert pipeline.depth_estimator is None
        assert pipeline.stereo_worker is None


def test_invalid_rig_disables_depth_without_killing_the_pipeline():
    """A bad camera config must degrade to mono, not crash on start-up."""
    config = _make_config()

    with ExitStack() as stack:
        for target in _STAGE_PATCHES:
            stack.enter_context(patch(target))
        stack.enter_context(
            patch(
                "offroad_autonomy.pipeline.StereoDepthEstimator",
                side_effect=ValueError("optical axes"),
            )
        )

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)

        assert pipeline.depth_estimator is None
        assert pipeline.terrain_analyzer is None
        assert pipeline.stereo_worker is None


def test_stitched_mode_needs_both_frames():
    config = _make_config(segmentation_mode="stitched", depth_enabled=False)
    with ExitStack() as stack:
        pipeline, _, _, _, _ = _pipeline_with_stereo(stack, config)

        assert pipeline.has_input(_pair())
        assert not pipeline.has_input(_pair(right=None))
        # The stitched view is wider than one camera, never narrower.
        assert pipeline.view.size[0] >= W
        assert pipeline.valid_roi.shape == (pipeline.view.size[1], pipeline.view.size[0])


def test_invalid_segmentation_mode_is_refused():
    with ExitStack() as stack:
        with pytest.raises(ValueError, match="segmentation"):
            _pipeline_with_stereo(stack, replace(_make_config(), segmentation_mode="center"))
