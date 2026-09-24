"""Unit tests for shared types."""

import numpy as np
import pytest

from offroad_autonomy.types import (
    GMSL2_SENSOR,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    StereoFramePair,
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


def test_gmsl2_sensor_defaults():
    """The rig's one camera model, as specified."""
    assert GMSL2_SENSOR.model == "GMSL2"
    assert (GMSL2_SENSOR.width, GMSL2_SENSOR.height) == (2880, 1860)
    assert GMSL2_SENSOR.fov_x_deg == 120.0
    assert GMSL2_SENSOR.target_fps >= 28.0


def test_camera_specs_expose_sensor_properties():
    """A mount reads its intrinsics through the shared sensor."""
    cfg = PipelineConfig()
    sensor = cfg.left_camera.sensor

    for spec in (cfg.left_camera, cfg.right_camera):
        assert spec.width == sensor.width
        assert spec.height == sensor.height
        assert spec.fov_x_deg == GMSL2_SENSOR.fov_x_deg
        assert spec.fov_y_deg == pytest.approx(GMSL2_SENSOR.fov_y_deg)
        assert spec.target_fps == sensor.target_fps


def test_stereo_frame_pair_only_offers_synchronised_stereo():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert StereoFramePair(left=frame, right=frame).has_stereo
    assert not StereoFramePair(left=frame, right=frame, synchronized=False).has_stereo
    assert not StereoFramePair(left=frame, right=None).has_stereo
    assert StereoFramePair(left=frame, right=None).frame("left") is frame


def test_working_resolution_defaults_are_a_clean_fraction():
    cfg = PipelineConfig()
    capture = cfg.left_camera.sensor

    assert (cfg.preprocess_width, cfg.preprocess_height) == (720, 465)
    assert (cfg.stereo_width, cfg.stereo_height) == (720, 465)
    # 3/4 of the capture, 1/4 of the imager - uniform in both axes.
    assert cfg.preprocess_width * 4 == capture.width * 3
    assert cfg.preprocess_height * 4 == capture.height * 3
    assert GMSL2_SENSOR.width // cfg.preprocess_width == 4
