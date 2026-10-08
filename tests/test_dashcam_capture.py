"""Single-camera attachment and shared inference/display capture."""

from dataclasses import replace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from offroad_autonomy.simulation.beamng_client import BeamNGClient, BeamNGConnectionError
from offroad_autonomy.types import PipelineConfig


@pytest.mark.parametrize(
    "host,launch,hint",
    [
        ("localhost", False, "BEAMNG_HOME"),
        ("192.168.1.50", False, "firewall"),
        ("localhost", True, "Automatic launch"),
    ],
)
@pytest.mark.parametrize("error_type", ["disconnected", "socket"])
def test_connection_failure_explains_endpoint_and_recovery(host, launch, hint, error_type):
    from beamngpy.logging import BNGDisconnectedError

    cfg = PipelineConfig(beamng_host=host, beamng_port=65432, beamng_launch=launch)
    client = BeamNGClient(cfg)
    error = BNGDisconnectedError("refused") if error_type == "disconnected" else OSError("refused")
    with patch("beamngpy.BeamNGpy") as bng, patch("beamngpy.Scenario") as scenario:
        bng.return_value.open.side_effect = error
        with pytest.raises(BeamNGConnectionError) as caught:
            client.connect()
        message = str(caught.value)
        assert f"{host}:65432" in message
        assert "-tcom -tport 65432" in message
        assert hint in message
        assert caught.value.__cause__ is error
        scenario.assert_not_called()
        client.disconnect()
        if launch:
            bng.return_value.close.assert_called_once()
        else:
            bng.return_value.disconnect.assert_called_once()


@pytest.mark.parametrize("headless", [True, False])
@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_connect_attaches_only_the_dashcam(headless, transport):
    cfg = PipelineConfig(ui_headless=headless, beamng_camera_transport=transport)
    client = BeamNGClient(cfg)
    with (
        patch("beamngpy.BeamNGpy"),
        patch("beamngpy.Scenario"),
        patch("beamngpy.Vehicle") as vehicle_cls,
        patch("beamngpy.sensors.camera.Camera") as camera,
        patch("offroad_autonomy.simulation.beamng_client.time.sleep"),
    ):
        vehicle_cls.return_value.queue_lua_command.return_value = "manualGearbox"
        client.connect()
        camera.assert_called_once()
        args = camera.call_args.kwargs
        assert args["name"] == "dashcam"
        assert args["pos"] == cfg.camera.pos
        assert args["resolution"] == (960, 620)
        assert args["field_of_view_y"] == pytest.approx(cfg.camera.fov_y_deg)
        assert args["is_render_colours"]
        assert not args["is_render_annotations"]
        assert not args["is_render_instance"]
        assert not args["is_render_depth"]
        assert args["is_streaming"] == (transport == "shared_memory")
        assert set(client._cameras) == {"dashcam"}
        client._vehicle.set_shift_mode.assert_called_once_with("realistic_automatic")
        client._vehicle.control.assert_any_call(gear=1)
        client.disconnect()
        camera.return_value.remove.assert_called_once()


@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_capture_reads_one_sensor_and_owns_its_pixels(transport):
    cfg = PipelineConfig(beamng_camera_transport=transport)
    cfg.camera = replace(cfg.camera, sensor=replace(cfg.camera.sensor, width=4, height=2))
    client = BeamNGClient(cfg)
    sensor = MagicMock()
    rgba = np.zeros((2, 4, 4), dtype=np.uint8)
    rgba[:] = (11, 22, 33, 255)
    buffer = bytearray(rgba.tobytes())
    sensor.colour_shmem.read.return_value = memoryview(buffer)
    sensor.poll_raw.return_value = {"colour": buffer}
    client._cameras = {"dashcam": sensor}

    first = client.capture_frame()
    assert first.is_new
    assert first.image[0, 0].tolist() == [33, 22, 11]
    assert not client.capture_frame().is_new
    buffer[0] = 99
    third = client.capture_frame()
    assert third.is_new
    assert third.frame_id == first.frame_id + 2
    assert first.image[0, 0].tolist() == [33, 22, 11]
    assert third.image[0, 0].tolist() == [33, 22, 99]
    if transport == "socket":
        assert sensor.poll_raw.call_count == 3
        sensor.colour_shmem.read.assert_not_called()
    else:
        assert sensor.colour_shmem.read.call_count == 3
        sensor.poll_raw.assert_not_called()


def test_missing_or_malformed_dashcam_frame_is_rejected():
    client = BeamNGClient(PipelineConfig())
    assert client.capture_frame() is None
    sensor = MagicMock()
    sensor.colour_shmem.read.return_value = b"bad frame"
    client._cameras = {"dashcam": sensor}
    assert client.capture_frame() is None


@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_annotation_depth_and_orbit_buffers_cannot_replace_dashcam_rgb(transport):
    cfg = PipelineConfig(beamng_camera_transport=transport)
    cfg.camera = replace(cfg.camera, sensor=replace(cfg.camera.sensor, width=4, height=2))
    client = BeamNGClient(cfg)
    sensor, orbit = MagicMock(), MagicMock()
    rgb = np.full((2, 4, 4), [11, 22, 33, 255], dtype=np.uint8).tobytes()
    labels = np.full((2, 4, 4), 255, dtype=np.uint8).tobytes()
    sensor.colour_shmem.read.return_value = rgb
    sensor.annotation_shmem.read.side_effect = AssertionError("ground-truth annotations read")
    sensor.instance_shmem.read.side_effect = AssertionError("ground-truth instances read")
    sensor.depth_shmem.read.side_effect = AssertionError("ground-truth depth read")
    sensor.poll_raw.return_value = {
        "colour": rgb,
        "annotation": labels,
        "instance": labels,
        "depth": labels,
    }
    client._cameras = {"dashcam": sensor, "orbit": orbit}
    captured = client.capture_frame()
    np.testing.assert_array_equal(captured.image, np.full((2, 4, 3), [33, 22, 11], dtype=np.uint8))
    sensor.colour_shmem.read.return_value = None
    sensor.poll_raw.return_value.pop("colour")
    assert client.capture_frame() is None  # No fallback to the available privileged buffers.
    sensor.annotation_shmem.read.assert_not_called()
    sensor.instance_shmem.read.assert_not_called()
    sensor.depth_shmem.read.assert_not_called()
    assert not orbit.mock_calls


def test_presentation_camera_also_disables_ground_truth_render_buffers():
    client = BeamNGClient(PipelineConfig())
    camera = MagicMock()
    client._attach_camera(camera, MagicMock(), client._config.orbit_camera)
    flags = camera.call_args.kwargs
    assert flags["is_render_colours"]
    assert not flags["is_render_annotations"]
    assert not flags["is_render_instance"]
    assert not flags["is_render_depth"]


@pytest.mark.parametrize("kind,gear", [("manualGearbox", 1), ("automaticGearbox", 2)])
def test_resume_selects_forward_before_releasing_brakes(kind, gear):
    client = BeamNGClient(PipelineConfig())
    vehicle = MagicMock()
    client._vehicle = vehicle
    vehicle.queue_lua_command.return_value = kind
    client.release_park()
    assert vehicle.mock_calls[:5] == [
        call.control(throttle=0, brake=0, parkingbrake=1),
        call.set_shift_mode("realistic_automatic"),
        call.queue_lua_command(
            'return powertrain.getDevice("gearbox").type',
            response=True,
        ),
        call.control(gear=gear),
        call.control(steering=0, throttle=0, brake=0, parkingbrake=0),
    ]


@pytest.mark.parametrize("failure_at", ["set_shift_mode", "queue_lua_command"])
def test_failed_forward_setup_keeps_parking_brake_applied(failure_at):
    client = BeamNGClient(PipelineConfig())
    vehicle = MagicMock()
    client._vehicle = vehicle
    getattr(vehicle, failure_at).side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError, match="connection lost"):
        client.release_park()
    assert not any(c.kwargs.get("parkingbrake") == 0 for c in vehicle.control.call_args_list)


def test_unknown_gearbox_is_not_released():
    client = BeamNGClient(PipelineConfig())
    client._vehicle = MagicMock()
    client._vehicle.queue_lua_command.return_value = "unsupported"
    with pytest.raises(RuntimeError, match="Unsupported forward-drive gearbox"):
        client.release_park()
    assert not any(
        c.kwargs.get("parkingbrake") == 0 for c in client._vehicle.control.call_args_list
    )
