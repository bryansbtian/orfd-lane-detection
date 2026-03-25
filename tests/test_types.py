"""Unit tests for shared types."""

import numpy as np

from offroad_autonomy.types import (
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    StabilizedResult,
    VehicleState,
)


def test_frame_packet_creation():
    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    fp = FramePacket(raw=raw, preprocessed=raw, timestamp=0.0, height=720, width=1280)
    assert fp.height == 720
    assert fp.width == 1280
    assert fp.raw.shape == (720, 1280, 3)


def test_perception_result_defaults():
    mask = np.zeros((360, 640), dtype=bool)
    pr = PerceptionResult(mask=mask)
    assert pr.num_detections == 0
    assert pr.confidences == []
    assert pr.inference_time_ms == 0.0


def test_control_command_defaults():
    cmd = ControlCommand()
    assert cmd.steering == 0.0
    assert cmd.throttle == 0.0
    assert cmd.brake == 0.0


def test_vehicle_state_defaults():
    vs = VehicleState()
    assert vs.speed_mps == 0.0
    assert vs.heading_rad == 0.0


def test_path_plan_fields():
    pts = np.array([[320.0, 300.0], [320.0, 200.0], [320.0, 100.0]])
    plan = PathPlan(centerline=pts, heading_rad=0.1, curvature=0.01)
    assert plan.centerline.shape == (3, 2)
    assert not plan.kalman_active


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert cfg.beamng_port == 64256
    assert cfg.confidence_threshold == 0.25
    assert cfg.perception_prompts == [
        "traversable road",
        "dirt road",
        "off-road trail",
        "drivable terrain",
        "gravel path",
    ]
    assert cfg.ema_alpha == 0.7
