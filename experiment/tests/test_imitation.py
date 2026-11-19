"""Signal joins, leakage prevention, timestamped video, and fail-closed learned controls."""

import json
import math
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from experiment.config import ExperimentConfig
from experiment.imitation.comparison import route_geometry, summarize_trial
from experiment.imitation.config import ImitationConfig
from experiment.imitation.controller import (
    ImitationController,
    SteeringCalibration,
    match_camera_fov,
)
from experiment.imitation.learning.recordings import discover_recordings
from experiment.imitation.learning.synchronization import (
    aligned_indices,
    steering_evidence,
    temporal_split,
)
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, VehicleState


def calibration():
    return SteeringCalibration(
        {
            "version": 1,
            "simulation_only": True,
            "measured": True,
            "camera_mount_verified": True,
            "measurement_source": "Synthetic Test Fixture",
            "recorded_rwa_radians_positive_left_verified": True,
            "real_wheelbase_m": 2.815,
            "simulator_wheelbase_m": 2.6,
            "wheel_angle_left_rad": [-0.5, 0.0, 0.5],
            "beamng_commands": [1.0, 0.0, -1.0],
        }
    )


class Predictor:
    def __init__(self, angle=0.2):
        self.metadata = {"camera_calibration": {"width": 2880, "k": [1714.20536]}}
        self.angle = angle
        self.calls = 0

    def predict(self, image, speed):
        self.calls += 1
        return self.angle


def plan(config):
    camera = CameraModel(config.camera, config.preprocess_width, config.preprocess_height)
    forward = np.linspace(2, 14, 20)
    return PathPlan(centerline=camera.ground_to_image(forward, np.zeros_like(forward))[::-1])


def test_timestamp_join_separates_causal_input_from_nearest_target():
    settings = ImitationConfig()
    times = np.array([100_000_000, 133_000_000, 166_000_000])
    assert aligned_indices(times, 125_000_000, settings) == (0, 1)
    assert aligned_indices(times, 133_000_000, settings) == (1, 1)
    assert aligned_indices(times, 90_000_000, settings) is None
    assert aligned_indices(times, 250_000_000, settings) is None


def test_temporal_splits_exclude_boundary_guard_intervals():
    settings = ImitationConfig()
    assert temporal_split(20, 100, settings) == "train"
    assert temporal_split(60, 100, settings) == "discard"
    assert temporal_split(70, 100, settings) == "validation"
    assert temporal_split(80, 100, settings) == "discard"
    assert temporal_split(95, 100, settings) == "test"


def test_dynamics_check_rejects_wrong_units_and_keeps_human_provenance_unknown():
    settings = ImitationConfig()
    rows = []
    for angle in np.linspace(-0.2, 0.2, 100):
        rows.append(
            {
                "speed_mps": 5.0,
                "raw_rwa": angle,
                "yaw_rate": 5 * math.tan(angle) / settings.real_wheelbase_m,
            }
        )
    result = steering_evidence(rows, settings)
    assert result["passed"]
    assert not result["human_control_confirmed"]
    assert not result["independent_reference_confirmed"]
    for row in rows:
        row["raw_rwa"] *= 180 / math.pi
    assert not steering_evidence(rows, settings)["passed"]


def test_archive_discovery_skips_exports_and_duplicate_recordings(tmp_path):
    paths = []
    for name in ("original.zip", "copy.zip", "exports.zip"):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            if name == "exports.zip":
                archive.writestr("frames.csv", "export")
            else:
                archive.writestr("bag/bag_0.mcap", b"fixture")
        paths.append(path)
    sessions, skipped = discover_recordings(paths)
    assert len(sessions) == 1
    assert len(skipped) == 2


def test_loose_mcap_segments_are_grouped_as_one_recording(tmp_path):
    paths = [tmp_path / "drive_1.mcap", tmp_path / "drive_0.mcap"]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]))
    sessions, skipped = discover_recordings(paths)
    assert len(sessions) == 1
    assert sessions[0]["members"] == ["drive_0.mcap", "drive_1.mcap"]
    assert skipped == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("image_width", "bad"),
        ("pretrained", "false"),
        ("sample_hz", 0),
        ("workers", -1),
        ("comparison_steps", 1.2),
    ],
)
def test_bad_imitation_configuration_is_rejected(key, value):
    with pytest.raises((ValueError, TypeError)):
        replace(ImitationConfig(), **{key: value})


def test_uncalibrated_actuation_is_refused_and_left_turn_maps_negative():
    data = calibration().metadata.copy()
    data["measured"] = False
    with pytest.raises(ValueError, match="Measured"):
        SteeringCalibration(data)
    with pytest.raises(ValueError, match="calibration_path"):
        SteeringCalibration.load("")
    assert calibration().command(0.25) == pytest.approx(-0.5)
    assert calibration().command(-100) == pytest.approx(1)
    with pytest.raises(ValueError, match="finite"):
        calibration().command(float("nan"))


def test_virtual_camera_crop_matches_fov_and_refuses_missing_coverage():
    image = np.zeros((620, 960, 3), dtype=np.uint8)
    matched = match_camera_fov(image, 120, 80)
    assert matched.shape[0] < 620
    assert matched.shape[1] < 960
    assert matched.shape[1] / matched.shape[0] == pytest.approx(960 / 620, abs=0.01)
    with pytest.raises(ValueError, match="field of view"):
        match_camera_fov(image, 70, 80)


def test_learned_control_obeys_time_slew_and_missing_observation_stops():
    config = ExperimentConfig()
    now = [0.0]
    predictor = Predictor()
    controller = ImitationController(
        config, predictor=predictor, calibration=calibration(), clock=lambda: now[0]
    )
    image = np.zeros((620, 960, 3), dtype=np.uint8)
    state = VehicleState(speed_mps=2)
    path = plan(config)
    first = controller.compute_observation(path, state, image)
    assert first.steering == 0
    now[0] = 0.1
    second = controller.compute_observation(path, state, image)
    assert -config.imitation.steering_slew_per_s * 0.1 <= second.steering < 0
    assert controller.last_angle_rad == 0.2
    stopped = controller.compute(path, state)
    assert stopped.throttle == 0 and stopped.parkingbrake == 1
    controller.reset()
    assert controller.compute_observation(path, state, image).steering == 0


def test_gate_stop_bypasses_inference_and_invalid_prediction_parks():
    config = ExperimentConfig()
    predictor = Predictor(float("nan"))
    controller = ImitationController(config, predictor=predictor, calibration=calibration())
    path = plan(config)
    image = np.zeros((620, 960, 3), dtype=np.uint8)
    path.speed_scale = 0
    result = controller.compute_observation(path, VehicleState(speed_mps=2), image)
    assert result.throttle == 0 and result.brake > 0
    assert predictor.calls == 0
    path.speed_scale = 1
    result = controller.compute_observation(path, VehicleState(speed_mps=2), image)
    assert result.throttle == 0 and result.parkingbrake == 1


def test_comparison_metrics_use_reference_geometry_and_detect_unsafe_commands():
    case = {
        "radius_m": 9,
        "straight_m": 10,
        "node_spacing_m": 0.5,
        "width_m": 4,
        "completion_margin_m": 1,
    }
    reference = route_geometry(case)
    assert reference.shape[1] == 2
    assert len(reference) > 20
    rows = []
    for index, error in enumerate((0.1, 1.5, 0.2)):
        rows.append(
            {
                "center_error_m": error,
                "steering": 0.1 * index,
                "remaining_m": 0,
                "control_ms": 2,
                "stop_required": True,
                "throttle": 0.1,
                "brake": 0,
                "parkingbrake": 0,
                "simulation_time_s": index,
            }
        )
    result = summarize_trial(rows, case, ExperimentConfig())
    assert result["completed"]
    assert result["trail_departures"] == 1
    assert not result["safe_stop_compliance"]


def test_cli_offers_rl_and_imitation_controller_options():
    from experiment.run import build_parser

    assert build_parser().parse_args(["--controller", "imitation"]).controller == "imitation"
    assert build_parser().parse_args(["--rl-policy", "p.npz"]).rl_policy == "p.npz"


def test_imitation_cli_refuses_unmeasured_adapter_before_connecting(monkeypatch):
    from experiment.run import main

    def unexpected(*args, **kwargs):
        pytest.fail("Uncalibrated imitation must not create a simulator client")

    monkeypatch.setattr("offroad_autonomy.main.BeamNGClient", unexpected)
    with pytest.raises(SystemExit, match="calibration_path"):
        main(["--controller", "imitation"])


def test_inference_over_budget_stops_instead_of_using_late_controls(monkeypatch):
    from types import SimpleNamespace

    config = ExperimentConfig()
    controller = ImitationController(config, predictor=Predictor(), calibration=calibration())
    timestamps = iter([0.0, 0.2])
    monkeypatch.setattr(
        "experiment.imitation.controller.time",
        SimpleNamespace(perf_counter=lambda: next(timestamps)),
    )
    result = controller.compute_observation(
        plan(config), VehicleState(speed_mps=2), np.zeros((62, 96, 3), dtype=np.uint8)
    )
    assert result.parkingbrake == 1
    assert result.throttle == 0


def test_unavailable_simulator_produces_no_fabricated_comparison(monkeypatch, tmp_path):
    from experiment.imitation.comparison import compare_suite

    def unavailable(*args, **kwargs):
        raise ConnectionRefusedError("Synthetic Unavailable Endpoint")

    monkeypatch.setattr("experiment.imitation.comparison.socket.create_connection", unavailable)
    config = ExperimentConfig()
    config.imitation = replace(config.imitation, run_dir=str(tmp_path))
    suite = tmp_path / "suite.yaml"
    suite.write_text("cases: []\n")
    report = compare_suite(config, suite)
    assert report["status"] == "Unavailable"
    assert report["trials"] == []


def test_h264_decoder_preserves_source_pts_through_flush():
    av = pytest.importorskip("av")
    from experiment.imitation.learning.video import TimestampedDecoder

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width = 64
    encoder.height = 32
    encoder.pix_fmt = "yuv420p"
    from fractions import Fraction

    encoder.time_base = Fraction(1, 30)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "23"}
    decoder = TimestampedDecoder(1)
    expected = []
    outputs = []
    for index in range(5):
        frame = av.VideoFrame.from_ndarray(
            np.full((32, 64, 3), index * 40, dtype=np.uint8), format="rgb24"
        )
        frame.pts = index
        for packet in encoder.encode(frame):
            timestamp = 1_790_000_000_000_000_000 + index * 33_333_333
            expected.append(timestamp)
            outputs.extend(decoder.push(bytes(packet), {"image_header_ns": timestamp}))
    for packet in encoder.encode(None):
        raise AssertionError("Test encoder should not delay packets")
    outputs.extend(decoder.flush())
    assert [row["image_header_ns"] for _, row in outputs] == expected
    assert not decoder.pending


def test_streaming_ros2_reader_roundtrip_from_compressed_mcap_inside_zip(tmp_path):
    pytest.importorskip("mcap")
    pytest.importorskip("mcap_ros2")
    import io
    import struct

    from mcap.writer import CompressionType, Writer

    from experiment.imitation.learning.recordings import messages

    stream = io.BytesIO()
    writer = Writer(stream, compression=CompressionType.ZSTD)
    writer.start(profile="ros2")
    schema = writer.register_schema(
        name="std_msgs/msg/String", encoding="ros2msg", data=b"string data\n"
    )
    channel = writer.register_channel(topic="/test", message_encoding="cdr", schema_id=schema)
    payload = b"\x00\x01\x00\x00" + struct.pack("<I", 6) + b"hello\x00"
    writer.add_message(channel, 123, payload, 123)
    writer.finish()
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("session/session_0.mcap", stream.getvalue())
    sessions, _ = discover_recordings([archive_path])
    decoded = list(messages(sessions[0], ["/test"]))
    assert len(decoded) == 1
    assert decoded[0][2].log_time == 123
    assert decoded[0][3].data == "hello"
    assert list(messages(sessions[0], ["/missing"])) == []


def test_extractor_decodes_synchronizes_and_builds_disjoint_splits(monkeypatch, tmp_path):
    av = pytest.importorskip("av")
    from fractions import Fraction
    from types import SimpleNamespace

    from experiment.imitation.learning.extraction import build_dataset

    source = tmp_path / "drive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("drive/drive_0.mcap", b"Fixture Reader Is Replaced Below")
    settings = replace(
        ImitationConfig(), image_width=64, image_height=32, split_guard_s=0.02, cpu_threads=1
    )
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 64, 32, "yuv420p"
    encoder.time_base = Fraction(1, 30)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
    packets = []
    telemetry = []
    for index in range(60):
        timestamp = 1_000_000_000 + index * 33_333_333
        sec, nanosec = divmod(timestamp, 1_000_000_000)
        header = SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
        angle = float(0.1 * math.sin(index / 10))
        telemetry.append(
            SimpleNamespace(
                header=header,
                rwa=angle,
                spd_mps=2.0,
                driverbraking=0,
                cdcu_ctrl_mode=0,
                error=False,
                spd_error=False,
                fix=SimpleNamespace(latitude=0.0, longitude=0.0),
                imu=SimpleNamespace(
                    imu=SimpleNamespace(
                        angular_velocity=SimpleNamespace(
                            z=2 * math.tan(angle) / settings.real_wheelbase_m
                        )
                    )
                ),
            )
        )
        image = np.random.default_rng(index).integers(0, 255, (32, 64, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = index
        encoded = encoder.encode(frame)
        assert len(encoded) == 1
        packets.append(SimpleNamespace(header=header, format="h264", data=bytes(encoded[0])))

    def messages(session, topics):
        if settings.can_topic in topics:
            calibration = SimpleNamespace(
                width=64,
                height=32,
                k=[40, 0, 32, 0, 40, 16, 0, 0, 1],
                d=[],
                distortion_model="plumb_bob",
            )
            yield (
                "drive_0.mcap",
                "/image_processing/camera_info",
                SimpleNamespace(log_time=1_000_000_000),
                calibration,
            )
            for value in telemetry:
                yield (
                    "drive_0.mcap",
                    settings.can_topic,
                    SimpleNamespace(
                        log_time=value.header.stamp.sec * 1_000_000_000 + value.header.stamp.nanosec
                    ),
                    value,
                )
        else:
            for value in packets:
                yield (
                    "drive_0.mcap",
                    settings.image_topic,
                    SimpleNamespace(
                        log_time=value.header.stamp.sec * 1_000_000_000 + value.header.stamp.nanosec
                    ),
                    value,
                )

    monkeypatch.setattr("experiment.imitation.learning.extraction.messages", messages)
    output = tmp_path / "dataset"
    result = build_dataset([source], output, settings)
    assert all(result["counts"][name] > 0 for name in ("train", "validation", "test"))
    assert result["source_files_unchanged"]
    assert not result["human_control_confirmed"]
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert all(row["speed_header_ns"] <= row["image_header_ns"] for row in rows)
    assert all(
        row["throttle_target"] is None and row["brake_intensity_target"] is None for row in rows
    )
    with pytest.raises(ValueError, match="not empty"):
        build_dataset([source], output, settings)
