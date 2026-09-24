"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from offroad_autonomy.perception.camera_geometry import StereoRig, build_camera_models
from offroad_autonomy.types import (
    DEFAULT_LEFT_CAMERA,
    DEFAULT_RIGHT_CAMERA,
    GMSL2_CAPTURE_SENSOR,
)
from offroad_autonomy.utils.config import load_config

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


def test_load_config_derives_both_mounts_from_the_rig(tmp_path):
    """Intrinsics are declared once; mounts come from one rig definition."""
    config_path = tmp_path / "rig.yaml"
    config_path.write_text(
        "\n".join(
            [
                "beamng:",
                "  cameras:",
                "    sensor:",
                "      model: GMSL2",
                "      width: 1920",
                "      height: 1080",
                "      fov_h: 100.0",
                "      target_fps: 30.0",
                "    stereo_rig:",
                "      baseline_m: 0.5",
                "      lateral_offset_m: 0.0",
                "      forward_offset_m: -0.1",
                "      height_m: 1.6",
                "      pitch_deg: -5.0",
                "    left:",
                "      name: port",
                "    right:",
                "      name: starboard",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.left_camera.name == "port"
    assert config.left_camera.pos == pytest.approx((0.25, -0.1, 1.6))
    assert config.right_camera.pos == pytest.approx((-0.25, -0.1, 1.6))
    assert config.left_camera.dir == pytest.approx(config.right_camera.dir)
    assert config.stereo_rig.pitch_deg == -5.0

    assert config.left_camera.sensor is config.right_camera.sensor
    for spec in (config.left_camera, config.right_camera):
        assert (spec.width, spec.height) == (1920, 1080)
        assert spec.fov_x_deg == 100.0
        assert spec.target_fps == 30.0


def test_shipped_config_is_a_two_camera_gmsl2_bumper_pair():
    """The config in the repo must describe the front pair as specified."""
    config = load_config(DEFAULT_YAML)

    for spec in (config.left_camera, config.right_camera):
        assert spec.sensor.model == "GMSL2"
        assert (spec.width, spec.height) == (960, 620)
        assert spec.fov_x_deg == 120.0
        assert spec.target_fps >= 28.0
        # Ahead of all bodywork, so there is nothing to exclude.
        assert not spec.ego_mask.enabled

    assert config.left_camera.pos == pytest.approx((0.3, -1.95, 0.95))
    assert config.right_camera.pos == pytest.approx((-0.3, -1.95, 0.95))
    assert 0.4 <= config.stereo_rig.baseline_m <= 0.8
    assert config.stereo_rig.toe_out_deg == 0.0
    assert config.segmentation_mode in ("left", "right")


def test_shipped_config_has_a_separate_display_only_camera():
    """Three cameras, and the third is not a member of the stereo pair."""
    config = load_config(DEFAULT_YAML)

    assert config.display_rig.enabled
    display = config.display_camera
    assert display.name not in (config.left_camera.name, config.right_camera.name)
    assert display.pos != config.left_camera.pos
    # It is deliberately cheaper than a perception stream: it is only rendered.
    assert display.width * display.height <= config.left_camera.width * config.left_camera.height
    assert display.target_fps <= config.left_camera.target_fps
    # Nothing routes it into perception: the segmentation camera is one of
    # the pair, whatever the mode.
    assert config.segmentation_camera in (config.left_camera, config.right_camera)


def test_stereo_mount_is_tunable_by_named_scalars(tmp_path):
    """Height, forward position and baseline each move on their own line."""
    path = tmp_path / "mount.yaml"
    path.write_text(
        "beamng:\n  cameras:\n    stereo_rig:\n"
        "      baseline_m: 0.5\n      height_m: 1.10\n"
        "      forward_offset_m: -2.10\n      pitch_deg: -4.0\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.stereo_rig.height_m == pytest.approx(1.10)
    assert config.stereo_rig.forward_offset_m == pytest.approx(-2.10)
    assert config.left_camera.pos == pytest.approx((0.25, -2.10, 1.10))
    assert config.right_camera.pos == pytest.approx((-0.25, -2.10, 1.10))
    # Both cameras still share height, pitch and roll - only X differs.
    assert config.left_camera.dir == pytest.approx(config.right_camera.dir)


def test_display_mount_is_tunable_independently(tmp_path):
    """Moving the display camera must not disturb the perception pair."""
    path = tmp_path / "display.yaml"
    path.write_text(
        "beamng:\n  cameras:\n    visualization:\n"
        "      height_m: 2.40\n      forward_offset_m: 0.50\n      pitch_deg: -12.0\n",
        encoding="utf-8",
    )

    config = load_config(path)
    reference = load_config(DEFAULT_YAML)

    assert config.display_camera.pos == pytest.approx((0.0, 0.50, 2.40))
    assert config.left_camera.pos == pytest.approx(reference.left_camera.pos)
    assert config.right_camera.pos == pytest.approx(reference.right_camera.pos)


def test_display_camera_can_be_switched_off(tmp_path):
    """Without it the stack still runs - the dashboard just loses the panel."""
    path = tmp_path / "nodisplay.yaml"
    path.write_text(
        "beamng:\n  cameras:\n    visualization:\n      enabled: false\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert not config.display_rig.enabled
    assert config.left_camera.pos == pytest.approx((0.3, -1.95, 0.95))


def test_working_resolution_is_an_exact_fraction_of_the_capture():
    """A uniform resize is what keeps the scaled intrinsics valid."""
    config = load_config(DEFAULT_YAML)
    sensor = config.left_camera.sensor

    for width, height in (
        (config.preprocess_width, config.preprocess_height),
        (config.stereo_width, config.stereo_height),
    ):
        assert width / sensor.width == pytest.approx(height / sensor.height)


def test_camera_fields_fall_back_to_rig_defaults(tmp_path):
    """A partial camera block keeps the defaults for everything it omits."""
    config_path = tmp_path / "partial.yaml"
    config_path.write_text(
        "beamng:\n  cameras:\n    left:\n      name: only_a_name\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.left_camera.name == "only_a_name"
    assert config.left_camera.pos == pytest.approx(DEFAULT_LEFT_CAMERA.pos)
    assert config.right_camera.name == DEFAULT_RIGHT_CAMERA.name
    assert config.left_camera.sensor == GMSL2_CAPTURE_SENSOR


def test_explicit_mount_overrides_the_rig(tmp_path):
    config_path = tmp_path / "override.yaml"
    config_path.write_text(
        "beamng:\n  cameras:\n    left:\n      pos: [0.5, -1.7, 0.6]\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.left_camera.pos == (0.5, -1.7, 0.6)
    assert config.right_camera.pos == pytest.approx(DEFAULT_RIGHT_CAMERA.pos)


def test_disparity_range_defaults_to_derived():
    config = load_config(DEFAULT_YAML)

    assert config.stereo_num_disparities == 0
    assert config.stereo_max_disparities >= 16


def test_load_config_reads_depth_and_terrain_sections(tmp_path):
    config_path = tmp_path / "depth.yaml"
    config_path.write_text(
        "\n".join(
            [
                "depth:",
                "  enabled: false",
                "  async: false",
                "  rate_hz: 5",
                "  num_disparities: 96",
                "  max_depth_m: 45.0",
                "  roi:",
                "    enabled: false",
                "    row_top: 0.4",
                "  calibration:",
                "    D_left: [0.1, 0.0, 0.0, 0.0, 0.0]",
                "    T: [-0.55, 0.0, 0.0]",
                "terrain:",
                "  obstacle_height_m: 0.5",
                "  max_slope_deg: 18.0",
                "  fusion_weight: 0.9",
                "planning:",
                "  min_clearance_m: 1.4",
                "control:",
                "  clearance_stop_m: 3.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.depth_enabled is False
    assert config.stereo_async is False
    assert config.stereo_rate_hz == 5.0
    assert config.stereo_num_disparities == 96
    assert config.stereo_max_depth_m == 45.0
    assert config.depth_roi_enabled is False
    assert config.depth_roi_row_top == 0.4
    assert config.stereo_D_left == (0.1, 0.0, 0.0, 0.0, 0.0)
    assert config.stereo_T == (-0.55, 0.0, 0.0)
    assert config.stereo_K_left is None
    assert config.obstacle_height_m == 0.5
    assert config.max_slope_deg == 18.0
    assert config.depth_fusion_weight == 0.9
    assert config.planner_min_clearance_m == 1.4
    assert config.clearance_stop_m == 3.0


def test_invalid_segmentation_mode_is_refused(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("perception:\n  segmentation_mode: center\n", encoding="utf-8")

    with pytest.raises(ValueError, match="segmentation_mode"):
        load_config(config_path)


def test_shipped_default_config_builds_a_valid_stereo_rig():
    """The config in the repo must describe a usable pair, not just parse."""
    config = load_config(DEFAULT_YAML)

    left, right = build_camera_models(config)
    rig = StereoRig(left, right)

    assert rig.baseline_m == pytest.approx(config.stereo_rig.baseline_m)
    assert rig.axis_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_jetson_config_extends_the_default():
    config = load_config(REPO / "configs" / "jetson.yaml")
    base = load_config(DEFAULT_YAML)

    assert config.beamng_launch is False
    assert config.beamng_camera_transport == "socket"
    assert config.ui_headless is True
    # Everything the overlay does not mention is inherited.
    assert config.left_camera.pos == base.left_camera.pos
    assert config.segmentation_mode == base.segmentation_mode
    assert config.map_spawns == base.map_spawns


def test_extends_cycle_is_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")
