"""Unit tests for the stereo pair's camera models and rig geometry."""

import math
from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import (
    CameraModel,
    StereoRig,
    build_camera_models,
    relative_pose,
)
from offroad_autonomy.types import (
    DEFAULT_LEFT_CAMERA,
    DEFAULT_RIGHT_CAMERA,
    GMSL2_CAPTURE_SENSOR,
    GMSL2_SENSOR,
    PipelineConfig,
    StereoRigSpec,
    stereo_camera_spec,
)


def _rig(width: int = 720, height: int = 465) -> StereoRig:
    """The shipped pair, modelled at the default working resolution."""
    return StereoRig(
        CameraModel(DEFAULT_LEFT_CAMERA, width, height),
        CameraModel(DEFAULT_RIGHT_CAMERA, width, height),
    )


def _pair(rig: StereoRigSpec, sensor=GMSL2_CAPTURE_SENSOR):
    from offroad_autonomy.types import EgoMaskSpec

    return (
        stereo_camera_spec(rig, "left", "l", sensor, EgoMaskSpec()),
        stereo_camera_spec(rig, "right", "r", sensor, EgoMaskSpec()),
    )


def test_camera_axes_follow_vehicle_convention():
    """+X is the vehicle's left, so camera-right must be vehicle -X."""
    model = CameraModel(
        replace(DEFAULT_LEFT_CAMERA, dir=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0)),
        width=720,
        height=465,
    )

    right, down, forward = model.rotation
    assert np.allclose(right, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(down, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.allclose(forward, [0.0, -1.0, 0.0], atol=1e-6)


def test_focal_length_is_derived_from_horizontal_fov():
    """f = (w/2) / tan(hfov/2) - never a hard-coded constant."""
    sensor = replace(GMSL2_SENSOR, width=640, height=360, fov_x_deg=90.0)
    model = CameraModel(replace(DEFAULT_LEFT_CAMERA, sensor=sensor), 640, 360)

    # At 90 degrees tan(45) == 1, so f is exactly half the width.
    assert model.focal_px == pytest.approx(320.0)
    assert model.cx == pytest.approx(320.0)
    assert model.cy == pytest.approx(180.0)


def test_gmsl2_focal_length_matches_the_sensor_geometry():
    """The 2880x1860 @ 120 deg imager, at full resolution."""
    model = CameraModel(replace(DEFAULT_LEFT_CAMERA, sensor=GMSL2_SENSOR))

    expected = 1440.0 / math.tan(math.radians(60.0))
    assert model.focal_px == pytest.approx(expected, rel=1e-9)
    assert model.focal_px == pytest.approx(831.38, abs=0.01)
    assert model.horizontal_fov_deg == pytest.approx(120.0, abs=1e-6)
    # Vertical angle is a consequence of the aspect ratio, not a free knob.
    assert model.vertical_fov_deg == pytest.approx(96.41, abs=0.01)


def test_capture_sensor_is_an_exact_fraction_of_the_imager():
    """Same lens at 1/3 resolution: same angles, focal scaled exactly."""
    assert GMSL2_CAPTURE_SENSOR.width * 3 == GMSL2_SENSOR.width
    assert GMSL2_CAPTURE_SENSOR.height * 3 == GMSL2_SENSOR.height
    assert GMSL2_CAPTURE_SENSOR.fov_y_deg == pytest.approx(GMSL2_SENSOR.fov_y_deg)
    assert GMSL2_CAPTURE_SENSOR.focal_px == pytest.approx(GMSL2_SENSOR.focal_px / 3.0)


def test_focal_length_scales_exactly_with_downsampling():
    """Downscaling must scale the intrinsics, not invalidate them."""
    full = CameraModel(DEFAULT_LEFT_CAMERA)
    work = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)

    assert work.scale == pytest.approx(0.75)
    assert work.focal_px == pytest.approx(full.focal_px * 0.75)
    assert work.cx == pytest.approx(full.cx * 0.75)
    assert work.cy == pytest.approx(full.cy * 0.75)
    assert work.horizontal_fov_deg == pytest.approx(full.horizontal_fov_deg)


def test_both_cameras_share_one_gmsl2_sensor():
    """The pair is one camera model twice over."""
    for spec in (DEFAULT_LEFT_CAMERA, DEFAULT_RIGHT_CAMERA):
        assert spec.sensor.model == "GMSL2"
        assert spec.fov_x_deg == 120.0
        assert spec.sensor.target_fps >= 28.0
    assert DEFAULT_LEFT_CAMERA.sensor == DEFAULT_RIGHT_CAMERA.sensor
    assert CameraModel(DEFAULT_LEFT_CAMERA, 720, 465).focal_px == pytest.approx(
        CameraModel(DEFAULT_RIGHT_CAMERA, 720, 465).focal_px
    )


def test_beamngpy_receives_the_derived_vertical_fov():
    """beamngpy takes fov_y; passing the horizontal angle would widen the lens."""
    sensor = DEFAULT_LEFT_CAMERA.sensor

    assert sensor.fov_x_deg == 120.0
    assert sensor.fov_y_deg == pytest.approx(96.41, abs=0.01)


def test_sensor_frame_interval_matches_target_rate():
    sensor = GMSL2_CAPTURE_SENSOR
    assert sensor.frame_interval_s == pytest.approx(1.0 / sensor.target_fps)
    assert sensor.frame_interval_s <= 1.0 / 28.0


def test_project_backproject_round_trip():
    model = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    points = np.array([[-1.0, -10.0, 0.0], [2.0, -25.0, 0.6], [0.0, -4.0, -0.3]], dtype=np.float32)

    recovered = model.to_vehicle(model.backproject(*model.project(model.from_vehicle(points))))

    assert np.allclose(recovered, points, atol=1e-4)


def test_point_to_the_right_projects_right_of_centre():
    model = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    # +X is the vehicle's left, so -X is to the right (and the left camera
    # itself sits at +0.3).
    right_point = np.array([[-2.0, -12.0, 0.0]], dtype=np.float32)

    u, _, _ = model.project(model.from_vehicle(right_point))

    assert u[0] > model.cx


def test_pair_differs_only_by_the_baseline():
    """Same height, pitch, roll and direction; offset purely sideways."""
    left, right = DEFAULT_LEFT_CAMERA, DEFAULT_RIGHT_CAMERA

    assert left.pos[0] - right.pos[0] == pytest.approx(0.6)
    assert left.pos[1:] == right.pos[1:]
    assert left.dir == pytest.approx(right.dir)
    assert left.up == pytest.approx(right.up)


def test_baseline_is_configurable():
    for baseline in (0.4, 0.6, 0.8):
        left, right = _pair(StereoRigSpec(baseline_m=baseline))
        rig = StereoRig(CameraModel(left, 720, 465), CameraModel(right, 720, 465))
        assert rig.baseline_m == pytest.approx(baseline, abs=1e-9)


def test_pitch_tilts_both_cameras_down_equally():
    left, right = _pair(StereoRigSpec(pitch_deg=-10.0))

    for spec in (left, right):
        assert math.degrees(math.asin(spec.dir[2])) == pytest.approx(-10.0)
        # Up stays perpendicular to the view direction.
        assert float(np.dot(spec.dir, spec.up)) == pytest.approx(0.0, abs=1e-9)


def test_roll_is_applied_to_both_cameras():
    left, right = _pair(StereoRigSpec(roll_deg=5.0))
    level, _ = _pair(StereoRigSpec(roll_deg=0.0))

    assert left.up == pytest.approx(right.up)
    # Roll is a rotation about the optical axis, measured from the level up.
    assert left.dir == pytest.approx(level.dir)
    cos_roll = float(np.dot(left.up, level.up))
    assert math.degrees(math.acos(min(1.0, cos_roll))) == pytest.approx(5.0)


def test_toe_out_turns_each_camera_away_from_the_other():
    left, right = _pair(StereoRigSpec(toe_out_deg=4.0))

    # +X is left: the left camera yaws toward +X, the right toward -X.
    assert left.dir[0] > 0.0
    assert right.dir[0] < 0.0
    rig = StereoRig(CameraModel(left, 720, 465), CameraModel(right, 720, 465))
    # 2 x 4 deg of yaw; the -2 deg pitch shaves a few thousandths off.
    pitch = math.radians(-2.0)
    expected = math.degrees(
        math.acos(math.cos(pitch) ** 2 * math.cos(math.radians(8.0)) + math.sin(pitch) ** 2)
    )
    assert rig.axis_angle_deg == pytest.approx(expected, abs=1e-6)


def test_relative_pose_matches_opencv_convention():
    """X_right = R @ X_left + T, and T_x = -baseline for a parallel pair."""
    left = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    right = CameraModel(DEFAULT_RIGHT_CAMERA, 720, 465)
    rotation, translation = relative_pose(left, right)

    assert np.allclose(rotation, np.eye(3), atol=1e-9)
    assert np.allclose(translation, [-0.6, 0.0, 0.0], atol=1e-9)

    point = np.array([1.0, -15.0, 0.2])
    in_left = left.from_vehicle(point.reshape(1, 3))[0]
    in_right = right.from_vehicle(point.reshape(1, 3))[0]
    assert np.allclose(rotation @ in_left + translation, in_right, atol=1e-5)


def test_stereo_rig_reports_physical_baseline():
    assert _rig().baseline_m == pytest.approx(0.6, abs=1e-6)


def test_baseline_is_unaffected_by_the_sensor_change():
    """Baseline is extrinsic; swapping lenses must not move it."""
    wide = replace(GMSL2_SENSOR, fov_x_deg=150.0, width=1920, height=1080)
    rig = StereoRig(
        CameraModel(replace(DEFAULT_LEFT_CAMERA, sensor=wide), 960, 540),
        CameraModel(replace(DEFAULT_RIGHT_CAMERA, sensor=wide), 960, 540),
    )

    assert rig.baseline_m == pytest.approx(0.6, abs=1e-6)


def test_disparity_range_is_derived_from_the_rig():
    """numDisparities follows from f, B and the near limit - not a guess."""
    rig = _rig()

    needed = rig.disparity_for_depth(2.0)
    chosen = rig.disparity_range_for(2.0)

    assert chosen % 16 == 0
    assert chosen >= needed
    assert chosen - needed < 16
    assert rig.disparity_range_for(1.0) > chosen


def test_disparity_and_depth_are_inverse():
    rig = _rig()

    disparity = rig.disparity_for_depth(10.0)
    recovered = rig.depth_from_disparity(np.array([disparity], dtype=np.float32))

    assert recovered[0] == pytest.approx(10.0, rel=1e-5)


def test_zero_disparity_yields_no_depth():
    depth = _rig().depth_from_disparity(np.array([0.0], dtype=np.float32))

    assert np.isnan(depth[0])


def test_stereo_rig_accepts_a_small_misalignment():
    """Rectification absorbs a slight toe-in; the rig must not refuse it."""
    left = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    nudged = CameraModel(replace(DEFAULT_RIGHT_CAMERA, dir=(-0.03, -1.0, -0.035)), 720, 465)

    rig = StereoRig(left, nudged)

    assert 0.0 < rig.axis_angle_deg < 5.0


def test_stereo_rig_rejects_barely_overlapping_views():
    left = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    splayed = CameraModel(replace(DEFAULT_RIGHT_CAMERA, dir=(-0.6, -1.0, -0.035)), 720, 465)

    with pytest.raises(ValueError, match="optical axes"):
        StereoRig(left, splayed)


def test_stereo_rig_rejects_mismatched_focal_length():
    """A pair with different lenses cannot share one Z = fB/d."""
    zoomed_sensor = replace(GMSL2_CAPTURE_SENSOR, fov_x_deg=60.0)
    left = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    zoomed = CameraModel(replace(DEFAULT_RIGHT_CAMERA, sensor=zoomed_sensor), 720, 465)

    with pytest.raises(ValueError, match="resolution and field of view"):
        StereoRig(left, zoomed)


def test_stereo_rig_rejects_swapped_cameras():
    """Naming the right-hand camera 'left' inverts the sign of every depth."""
    left = CameraModel(DEFAULT_LEFT_CAMERA, 720, 465)
    right = CameraModel(DEFAULT_RIGHT_CAMERA, 720, 465)

    with pytest.raises(ValueError, match="Swap the"):
        StereoRig(right, left)


def test_default_rig_builds_from_config():
    config = PipelineConfig()

    left, right = build_camera_models(config)
    rig = StereoRig(left, right)

    assert (left.width, left.height) == (config.stereo_width, config.stereo_height)
    assert rig.baseline_m == pytest.approx(0.6, abs=1e-6)
    assert "GMSL2" in rig.describe()


def test_vertical_fov_follows_aspect_ratio():
    sensor = replace(GMSL2_SENSOR, width=640, height=360, fov_x_deg=90.0)
    model = CameraModel(replace(DEFAULT_LEFT_CAMERA, sensor=sensor), 640, 360)
    expected = math.degrees(2.0 * math.atan(180.0 / 320.0))

    assert model.vertical_fov_deg == pytest.approx(expected)
