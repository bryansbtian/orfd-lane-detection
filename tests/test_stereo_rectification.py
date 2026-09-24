"""Rectification and end-to-end stereo depth on rendered scenes.

The scene is an analytic ground plane carrying a band-limited random texture
(per-texel noise aliases below a pixel at range and SGBM then fails in a way
that looks exactly like a code bug). Both cameras render it through their own
real camera models at capture resolution, so the whole chain - rectification
maps, SGBM, Z = fB/d, filtering and reprojection - is checked against exact
geometry.
"""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.stereo_depth import StereoDepthEstimator
from offroad_autonomy.perception.stereo_rectification import (
    StereoRectifier,
    draw_epipolar_pair,
)
from offroad_autonomy.types import (
    EgoMaskSpec,
    PipelineConfig,
    StereoRigSpec,
    stereo_camera_spec,
)

_TEXEL_M = 0.02
_X_RANGE = (-20.0, 20.0)
_Y_RANGE = (-60.0, 2.0)


def _config(rig: StereoRigSpec | None = None, **overrides) -> PipelineConfig:
    overrides.setdefault("depth_roi_enabled", False)
    config = PipelineConfig(model_weights="dummy.pt", **overrides)
    if rig is not None:
        sensor = config.left_camera.sensor
        config = replace(
            config,
            stereo_rig=rig,
            left_camera=stereo_camera_spec(rig, "left", "l", sensor, EgoMaskSpec()),
            right_camera=stereo_camera_spec(rig, "right", "r", sensor, EgoMaskSpec()),
        )
    return config


@pytest.fixture(scope="module")
def ground_texture() -> np.ndarray:
    rng = np.random.default_rng(11)
    width = int((_X_RANGE[1] - _X_RANGE[0]) / _TEXEL_M)
    height = int((_Y_RANGE[1] - _Y_RANGE[0]) / _TEXEL_M)
    noise = rng.random((height, width), dtype=np.float32)
    # ~0.1 m and ~0.4 m correlation: fine detail near, coarse detail far.
    fine = cv2.GaussianBlur(noise, (0, 0), 0.1 / _TEXEL_M)
    coarse = cv2.GaussianBlur(noise, (0, 0), 0.4 / _TEXEL_M)
    tex = 0.5 * (fine - fine.mean()) / fine.std() + 0.5 * (coarse - coarse.mean()) / coarse.std()
    return np.clip(128 + 40 * tex, 0, 255).astype(np.uint8)


def _render(camera: CameraModel, texture: np.ndarray) -> np.ndarray:
    """Ray-cast the textured plane z = 0; sky is flat grey."""
    uu, vv = camera.pixel_grid()
    dirs = np.stack(
        [(uu - camera.cx) / camera.focal_px, (vv - camera.cy) / camera.focal_px, np.ones_like(uu)],
        axis=-1,
    ) @ camera.rotation.astype(np.float32)
    origin = camera.position.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs[..., 2] < -1e-6, -origin[2] / dirs[..., 2], -1.0)
    hit = t > 0
    x = origin[0] + dirs[..., 0] * t
    y = origin[1] + dirs[..., 1] * t
    map_x = ((x - _X_RANGE[0]) / _TEXEL_M).astype(np.float32)
    map_y = ((y - _Y_RANGE[0]) / _TEXEL_M).astype(np.float32)
    gray = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderValue=128)
    gray[~hit] = 128
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _truth_depth(camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """Along-axis depth of the ground for every pixel of ``camera``."""
    uu, vv = camera.pixel_grid()
    dirs = np.stack(
        [(uu - camera.cx) / camera.focal_px, (vv - camera.cy) / camera.focal_px, np.ones_like(uu)],
        axis=-1,
    ) @ camera.rotation.astype(np.float32)
    origin = camera.position.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs[..., 2] < -1e-6, -origin[2] / dirs[..., 2], -1.0)
    hit = t > 0
    # dirs has unit z-component in camera space before rotation, so t is
    # exactly the along-axis depth.
    return np.where(hit, t, 0.0).astype(np.float32), hit


@pytest.mark.parametrize("toe_out", [0.0, 3.0])
def test_rectified_rows_line_up(toe_out):
    """After rectification a world point lands on the same row in both views."""
    config = _config(StereoRigSpec(toe_out_deg=toe_out))
    rect = StereoRectifier(config)
    left = CameraModel(config.left_camera)
    right = CameraModel(config.right_camera)

    rng = np.random.default_rng(0)
    points = np.stack(
        [rng.uniform(-6, 6, 200), rng.uniform(-40, -5, 200), rng.uniform(-1.0, 2.0, 200)], axis=1
    )

    def project(model, rotation, projection):
        cam = model.from_vehicle(points).astype(np.float64) @ rotation.T
        pix = cam @ projection[:, :3].T + projection[:, 3]
        return pix[:, :2] / pix[:, 2:3]

    in_left = project(left, rect.R1, rect.P1)
    in_right = project(right, rect.R2, rect.P2)

    assert np.abs(in_left[:, 1] - in_right[:, 1]).max() < 1e-3
    # Right-camera features sit to the left (positive disparity).
    assert (in_left[:, 0] - in_right[:, 0] > 0).all()


def test_parallel_rig_rectifies_to_identity():
    rect = StereoRectifier(_config())

    assert np.allclose(rect.R1, np.eye(3), atol=1e-9)
    assert rect.baseline_m == pytest.approx(0.6, abs=1e-9)
    # Capture intrinsics scaled to the working size.
    assert rect.focal_px == pytest.approx(207.8, abs=0.2)


def test_calibration_override_is_used(tmp_path):
    """A real calibration from the config must replace the derived one."""
    config = _config(stereo_T=(-0.5, 0.0, 0.0))
    rect = StereoRectifier(config)

    assert rect.baseline_m == pytest.approx(0.5, abs=1e-9)
    assert np.allclose(rect.T, [-0.5, 0.0, 0.0])


def test_maps_are_built_once_and_reused():
    rect = StereoRectifier(_config())
    maps_before = rect._map_left[0]
    frame = np.zeros((620, 960, 3), dtype=np.uint8)

    rect.rectify(frame, frame)
    rect.rectify(frame, frame)

    assert rect._map_left[0] is maps_before


def test_epipolar_debug_view_has_guide_lines():
    left = np.zeros((100, 120), dtype=np.uint8)
    view = draw_epipolar_pair(left, left, spacing_px=20)

    assert view.shape == (100, 240, 3)
    assert view[10].any()  # a guide line crosses row 10 in both halves
    assert view[10, 10].any() and view[10, 200].any()


def _run(config, texture, **compute_kwargs):
    estimator = StereoDepthEstimator(config)
    left = _render(CameraModel(config.left_camera), texture)
    right = _render(CameraModel(config.right_camera), texture)
    return estimator, estimator.compute(left, right, **compute_kwargs)


def test_ground_depth_is_recovered_through_the_full_chain(ground_texture):
    config = _config()
    estimator, depth = _run(config, ground_texture)
    assert depth is not None

    truth, hit = _truth_depth(estimator.rectified)
    usable = (depth.depth_rect > 0) & hit & (truth > 4.0) & (truth < 25.0)
    error = np.abs(depth.depth_rect[usable] - truth[usable]) / truth[usable]

    assert usable.sum() > 20000
    assert np.median(error) < 0.03
    assert np.percentile(error, 90) < 0.10
    assert depth.valid_disparity_fraction > 0.3


def test_invalid_disparity_never_becomes_depth(ground_texture):
    config = _config()
    _, depth = _run(config, ground_texture)

    # Sky (textureless) and SGBM's left border produce no depth at all.
    assert (depth.disparity[depth.depth_rect > 0] > 0).all()
    assert (depth.depth_rect[depth.disparity <= 0] == 0).all()
    measured = depth.depth_rect[depth.depth_rect > 0]
    assert measured.min() >= config.stereo_min_depth_m - 1e-3
    assert measured.max() <= config.stereo_max_depth_m + 1e-3


def test_depth_lands_on_the_segmentation_grid(ground_texture):
    config = _config()
    estimator, depth = _run(config, ground_texture)

    assert depth.depth_m.shape == (config.preprocess_height, config.preprocess_width)
    truth, hit = _truth_depth(estimator.target)
    both = depth.valid & hit & (truth < 25.0)
    error = np.abs(depth.depth_m[both] - truth[both]) / truth[both]
    assert both.sum() > 20000
    assert np.median(error) < 0.03
    # Reprojected points sit on the ground plane.
    assert np.median(np.abs(depth.points_vehicle[both][:, 2])) < 0.1


def test_roi_band_crops_matching_and_gates_depth(ground_texture):
    config = _config(depth_roi_enabled=True, depth_roi_row_top=0.5, depth_roi_use_mask=True)
    estimator = StereoDepthEstimator(config)
    left = _render(CameraModel(config.left_camera), ground_texture)
    right = _render(CameraModel(config.right_camera), ground_texture)

    road = np.zeros((config.preprocess_height, config.preprocess_width), dtype=bool)
    road[300:, 300:420] = True
    depth = estimator.compute(left, right, road_mask=road)
    assert depth is not None

    # Nothing is matched above the band...
    cutoff = int(0.5 * config.stereo_height)
    assert (depth.disparity[:cutoff] <= 0).all()
    # ...and depth is only kept near the road mask.
    roi = estimator.roi_preview(road, None)
    assert not (depth.valid & ~roi).any()
    assert depth.valid.any()


def test_roi_corridor_limits_the_cloud(ground_texture):
    config = _config(
        depth_roi_enabled=True,
        depth_roi_use_mask=False,
        depth_roi_corridor_half_width_m=2.0,
        depth_roi_corridor_length_m=15.0,
    )
    _, depth = _run(config, ground_texture)

    assert depth is not None and len(depth.cloud_vehicle)
    assert np.abs(depth.cloud_vehicle[:, 0]).max() <= 2.0 + 1e-3
    # The corridor is measured from the rig, not from the vehicle origin, so
    # with the pair on the bumper the two frames differ by the mount offset.
    rig_y = config.left_camera.pos[1]
    assert (rig_y - depth.cloud_vehicle[:, 1]).max() <= 15.0 + 0.1


def test_corridor_statistics_are_reported(ground_texture):
    _, depth = _run(_config(), ground_texture)

    assert np.isfinite(depth.median_forward_depth_m)
    assert np.isfinite(depth.min_corridor_depth_m)
    assert depth.min_corridor_depth_m <= depth.median_forward_depth_m
    assert set(depth.timings_ms) >= {"rectification", "stereo_matching", "depth_filtering"}
