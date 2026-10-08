"""Unit tests for configuration loading."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from offroad_autonomy.types import DEFAULT_CAMERA, DashboardThresholds, PipelineConfig
from offroad_autonomy.utils import environment
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.environment import PlatformFacts

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"


def test_load_config_reads_perception_prompts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "perception:",
                '  model_weights: "dummy.pt"',
                "  prompts:",
                '    - "trail"',
                '    - "road"',
                "visualization:",
                "  dashboard:",
                "    colors:",
                "      BG: [1, 2, 3]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model_weights == "dummy.pt"
    assert config.perception_prompts == ["trail", "road"]
    assert config.dashboard_colors["BG"] == (1, 2, 3)


def test_legacy_depth_settings_cannot_enable_depth(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("depth:\n  enabled: true\nterrain:\n  fusion_weight: 1.0\n", encoding="utf-8")
    config = load_config(path)
    assert not hasattr(config, "depth_enabled")
    assert not hasattr(config, "depth_fusion_weight")


def test_vehicle_width_is_configured_with_control(tmp_path):
    path = tmp_path / "width.yaml"
    path.write_text("control:\n  vehicle_half_width_m: 1.2\n", encoding="utf-8")
    assert load_config(path).vehicle_half_width_m == pytest.approx(1.2)


def test_camera_offset_is_shared_vehicle_geometry(tmp_path):
    path = tmp_path / "geometry.yaml"
    path.write_text("control:\n  camera_ahead_of_rear_axle_m: 1.8\n", encoding="utf-8")

    assert load_config(path).camera_ahead_of_rear_axle_m == pytest.approx(1.8)
    assert load_config(DEFAULT_YAML).camera_ahead_of_rear_axle_m == pytest.approx(
        PipelineConfig().camera_ahead_of_rear_axle_m
    )


def test_controller_selection_settings_are_refused(tmp_path):
    path = tmp_path / "controller.yaml"
    path.write_text("control:\n  controller: stanley\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Remove control.controller"):
        load_config(path)


def test_invalid_segmentation_mode_is_refused(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("perception:\n  segmentation_mode: center\n", encoding="utf-8")

    with pytest.raises(ValueError, match="segmentation_mode"):
        load_config(config_path)


def test_jetson_config_extends_the_default(monkeypatch):
    monkeypatch.setenv("BEAMNG_HOST", "192.168.1.50")
    jetson = PlatformFacts(os_name="linux", default_gateway="192.168.1.1")
    with patch.object(environment, "detect_platform", return_value=jetson):
        config = load_config(REPO / "configs" / "jetson.yaml")
        base = load_config(DEFAULT_YAML)

    assert config.beamng_host == "192.168.1.50"
    assert config.beamng_launch is False
    assert config.beamng_camera_transport == "socket"
    assert config.ui_headless is True
    # Everything the overlay does not mention is inherited.
    assert config.camera == base.camera
    assert config.map_spawns == base.map_spawns


def test_extends_cycle_is_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_dashcam_defaults_match_shipped_config():
    cfg = load_config(DEFAULT_YAML)
    assert cfg.camera == DEFAULT_CAMERA == PipelineConfig().camera
    assert cfg.camera.name == "dashcam"
    assert cfg.camera.pos == (0.0, -0.30, 1.85)
    assert cfg.camera.sensor.model == "GMSL2"
    assert cfg.camera.fov_x_deg == 120.0
    assert not hasattr(cfg, "left_camera")
    assert not hasattr(cfg, "right_camera")
    assert not hasattr(cfg, "display_camera")


def test_dashcam_mount_and_sensor_are_configurable(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text(
        "beamng:\n  camera:\n    height_m: 2.1\n    pitch_deg: -12\n"
        "    sensor:\n      width: 1440\n      height: 930\n"
    )
    cfg = load_config(path)
    assert cfg.camera.pos == (0.0, -0.30, 2.1)
    assert cfg.camera.dir[2] == pytest.approx(-0.20791169)
    assert cfg.camera.width == 1440
    assert cfg.camera.fov_x_deg == 120.0


def test_old_multi_camera_config_requires_migration(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("beamng:\n  cameras:\n    left: {}\n")
    with pytest.raises(ValueError, match="beamng.camera"):
        load_config(path)


def test_dashboard_thresholds_default_to_the_shipped_config():
    thresholds = load_config(DEFAULT_YAML).dashboard_thresholds

    assert thresholds == DashboardThresholds()


def test_dashboard_floors_come_from_the_gate_and_safe_stop(tmp_path):
    path = tmp_path / "floors.yaml"
    path.write_text(
        "planning:\n  gate:\n    confidence_threshold: 0.3\nsafety:\n  min_road_fraction: 0.05\n"
        "visualization:\n  dashboard:\n    target_fps: 10\n    latency_budget_ms: 100\n",
        encoding="utf-8",
    )

    thresholds = load_config(path).dashboard_thresholds

    assert thresholds.confidence_floor == pytest.approx(0.3)
    assert thresholds.road_floor == pytest.approx(0.05)
    assert thresholds.target_fps == pytest.approx(10.0)
    assert thresholds.latency_budget_ms == pytest.approx(100.0)


@pytest.mark.parametrize(
    "block",
    [
        "target_fps: 0",
        "latency_budget_ms: -1",
        "fps_warn_fraction: 1.0",
        "confidence_good: 0.1",
        "fps_bar_scale: 1.0",
        "road_bar_full_scale: 0.01",
    ],
)
def test_invalid_dashboard_thresholds_are_refused(tmp_path, block):
    path = tmp_path / "bad.yaml"
    path.write_text(f"visualization:\n  dashboard:\n    {block}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="visualization.dashboard"):
        load_config(path)


def test_recording_defaults_match_the_shipped_config():
    config = load_config(DEFAULT_YAML)
    defaults = PipelineConfig()

    assert config.recording_fps == defaults.recording_fps
    assert config.recording_crf == defaults.recording_crf
    assert config.recording_preset == defaults.recording_preset
    assert config.recording_queue_frames == defaults.recording_queue_frames


@pytest.mark.parametrize(
    "block",
    ["  preset: turbo", "  crf: 60", "  fps: 0", "  queue_frames: 0"],
)
def test_invalid_recording_settings_are_refused(tmp_path, block):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("recording:\n" + block + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recording"):
        load_config(config_path)


def test_grid_profile_selects_the_grid_planner():
    config = load_config(REPO / "configs" / "grid.yaml")

    assert config.planner_mode == "grid"
    assert config.grid.arc_count == PipelineConfig().grid.arc_count


@pytest.mark.parametrize(
    ("block", "match"),
    [("    arc_size: 3", "arc_size"), ("    arc_count: 40", "odd"), ("    cell_m: 0", "cell_m")],
)
def test_invalid_grid_settings_are_refused(tmp_path, block, match):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("planning:\n  grid:\n" + block + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(config_path)


def test_orbit_camera_defaults_match_the_shipped_config():
    config = load_config(DEFAULT_YAML)
    defaults = PipelineConfig().orbit_camera

    assert config.orbit_camera.pos == pytest.approx(defaults.pos)
    assert config.orbit_camera.dir == pytest.approx(defaults.dir)
    assert config.orbit_camera.sensor == defaults.sensor


def test_orbit_camera_pose_and_sensor_are_configurable(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "presentation:\n"
        "  orbit_camera:\n"
        "    pos: [0.0, 8.0, 3.0]\n"
        "    pitch_deg: -20.0\n"
        "    sensor:\n"
        "      width: 640\n"
        "      height: 480\n"
        "      fov_h: 80.0\n",
        encoding="utf-8",
    )

    orbit = load_config(config_path).orbit_camera

    assert orbit.pos == (0.0, 8.0, 3.0)
    assert orbit.dir[2] == pytest.approx(-math.sin(math.radians(20.0)))
    assert (orbit.width, orbit.height, orbit.fov_x_deg) == (640, 480, 80.0)


@pytest.mark.parametrize(
    "block",
    [
        "    pos: [0.0, 6.0]",
        "    pitch_deg: 95",
        "    sensor:\n      width: 0",
        "    sensor:\n      fov_h: 180",
        "    sensor:\n      target_fps: 0",
    ],
)
def test_invalid_orbit_camera_settings_are_refused(tmp_path, block):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("presentation:\n  orbit_camera:\n" + block + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="presentation.orbit_camera"):
        load_config(config_path)
