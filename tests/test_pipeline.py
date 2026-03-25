"""Unit tests for the pipeline with mocked stages."""

from unittest.mock import MagicMock, patch

import numpy as np

from offroad_autonomy.types import (
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    StabilizedResult,
    VehicleState,
)


def _make_config() -> PipelineConfig:
    return PipelineConfig(
        model_weights="dummy.pt",
        preprocess_width=640,
        preprocess_height=360,
    )


def test_pipeline_step_produces_control_command():
    """Verify the pipeline step chains all stages and returns a ControlCommand."""
    config = _make_config()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=bool)
    mask[200:300, 200:440] = True

    mock_frame_packet = FramePacket(
        raw=frame, preprocessed=frame, timestamp=0.0, height=360, width=640,
    )
    mock_perception = PerceptionResult(
        mask=mask, confidences=[0.9], num_detections=1, inference_time_ms=15.0,
    )
    mock_stabilized = StabilizedResult(mask=mask, stability_score=0.95)
    mock_plan = PathPlan(
        centerline=np.array([[320.0, 300.0], [320.0, 100.0]]),
        heading_rad=0.0,
    )
    mock_command = ControlCommand(steering=0.0, throttle=0.3, brake=0.0)

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor") as MockPre,
        patch("offroad_autonomy.pipeline.RoadSegmenter") as MockSeg,
        patch("offroad_autonomy.pipeline.TemporalStabilizer") as MockStab,
        patch("offroad_autonomy.pipeline.CenterlinePlanner") as MockPlan,
        patch("offroad_autonomy.pipeline.StanleyController") as MockCtrl,
    ):
        MockPre.return_value.process.return_value = mock_frame_packet
        MockSeg.return_value.predict.return_value = mock_perception
        MockStab.return_value.stabilize.return_value = mock_stabilized
        MockPlan.return_value.plan.return_value = mock_plan
        MockCtrl.return_value.compute.return_value = mock_command

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)
        result = pipeline.step(frame, VehicleState())

        assert isinstance(result, ControlCommand)
        assert result.throttle == 0.3

        MockPre.return_value.process.assert_called_once()
        MockSeg.return_value.predict.assert_called_once()
        MockStab.return_value.stabilize.assert_called_once()
        MockPlan.return_value.plan.assert_called_once()
        MockCtrl.return_value.compute.assert_called_once()


def test_pipeline_step_result_exposes_stage_outputs():
    """Verify the debug pipeline result keeps intermediate stage outputs."""
    config = _make_config()

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=bool)
    mask[200:300, 200:440] = True

    mock_frame_packet = FramePacket(
        raw=frame, preprocessed=frame, timestamp=0.0, height=360, width=640,
    )
    mock_perception = PerceptionResult(
        mask=mask, confidences=[0.9], num_detections=1, inference_time_ms=15.0,
    )
    mock_stabilized = StabilizedResult(mask=mask, stability_score=0.95)
    mock_plan = PathPlan(
        centerline=np.array([[320.0, 300.0], [320.0, 100.0]]),
        heading_rad=0.0,
    )
    mock_command = ControlCommand(steering=0.0, throttle=0.3, brake=0.0)

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor") as MockPre,
        patch("offroad_autonomy.pipeline.RoadSegmenter") as MockSeg,
        patch("offroad_autonomy.pipeline.TemporalStabilizer") as MockStab,
        patch("offroad_autonomy.pipeline.CenterlinePlanner") as MockPlan,
        patch("offroad_autonomy.pipeline.StanleyController") as MockCtrl,
    ):
        MockPre.return_value.process.return_value = mock_frame_packet
        MockSeg.return_value.predict.return_value = mock_perception
        MockStab.return_value.stabilize.return_value = mock_stabilized
        MockPlan.return_value.plan.return_value = mock_plan
        MockCtrl.return_value.compute.return_value = mock_command

        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)
        result = pipeline.step_result(frame, VehicleState())

        assert isinstance(result, PipelineStepResult)
        assert result.frame is mock_frame_packet
        assert result.perception is mock_perception
        assert result.stabilized is mock_stabilized
        assert result.plan is mock_plan
        assert result.command is mock_command
