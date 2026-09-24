"""Unit tests for stereo depth recovery and terrain analysis.

The scenes here are analytic: a ground plane, optionally tilted, with a raised
slab standing on it. Because the geometry is known exactly, the tests assert on
recovered metres rather than on "it ran without crashing".
"""

import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.stereo_depth import StereoDepthEstimator
from offroad_autonomy.perception.fusion import fuse_rgb_depth
from offroad_autonomy.perception.terrain_analyzer import TerrainAnalyzer
from offroad_autonomy.types import DepthResult, PerceptionResult, PipelineConfig


def _config(**overrides) -> PipelineConfig:
    # Geometry tests look at the whole frame; the ROI has its own tests.
    overrides.setdefault("depth_roi_enabled", False)
    return PipelineConfig(model_weights="dummy.pt", **overrides)


def _ray_directions(camera):
    """Unit view rays per pixel, in vehicle space."""
    uu, vv = camera.pixel_grid()
    dirs = np.stack(
        [
            (uu - camera.cx) / camera.focal_px,
            (vv - camera.cy) / camera.focal_px,
            np.ones_like(uu),
        ],
        axis=-1,
    ) @ camera.rotation.astype(np.float32)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def _cast_scene(camera, slope_a=0.0, slope_b=0.0, slab=None):
    """Intersect every view ray with a tilted plane plus an optional slab.

    Returns the vehicle-space hit points, the along-axis depth, a hit mask and
    the mask of rays that landed on the slab.
    """
    dirs = _ray_directions(camera)
    origin = camera.position.astype(np.float32)

    # Plane z = a*x + b*y.
    denominator = dirs[..., 2] - slope_a * dirs[..., 0] - slope_b * dirs[..., 1]
    numerator = slope_a * origin[0] + slope_b * origin[1] - origin[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(denominator) > 1e-6, numerator / denominator, -1.0)
    t = np.where(t > 0, t, np.inf)

    on_slab = np.zeros(t.shape, dtype=bool)
    if slab is not None:
        centre, half, height = slab
        # Top face of an axis-aligned slab standing on the plane.
        top_z = slope_a * centre[0] + slope_b * centre[1] + height
        with np.errstate(divide="ignore", invalid="ignore"):
            t_top = np.where(np.abs(dirs[..., 2]) > 1e-6, (top_z - origin[2]) / dirs[..., 2], -1.0)
        t_top = np.where(t_top > 0, t_top, np.inf)
        hit_pts = origin + dirs * np.where(np.isfinite(t_top), t_top, 0.0)[..., None]
        inside = (
            (np.abs(hit_pts[..., 0] - centre[0]) <= half)
            & (np.abs(hit_pts[..., 1] - centre[1]) <= half)
            & np.isfinite(t_top)
        )
        on_slab = inside & (t_top < t)
        t = np.where(on_slab, t_top, t)

    hit = np.isfinite(t)
    points = origin + dirs * np.where(hit, t, 0.0)[..., None]
    depth = np.where(hit, (points - origin) @ camera.rotation[2].astype(np.float32), 0.0)
    return points.astype(np.float32), depth.astype(np.float32), hit, on_slab


def _depth_result(config, slope_a=0.0, slope_b=0.0, slab=None):
    """Build a DepthResult the way the estimator would, from exact geometry.

    Depth is cast in the *rectified* left camera - the frame SGBM measures
    in - and then goes through the estimator's own reprojection.
    """
    estimator = StereoDepthEstimator(config)
    left = estimator.rectified

    _, depth_left, hit, _ = _cast_scene(left, slope_a, slope_b, slab)
    usable = (
        hit & (depth_left >= config.stereo_min_depth_m) & (depth_left <= config.stereo_max_depth_m)
    )
    cloud, points, depth_image, valid = estimator._reproject(
        np.where(usable, depth_left, np.nan).astype(np.float32), usable
    )
    return DepthResult(
        depth_m=depth_image,
        valid=valid,
        points_vehicle=points,
        cloud_vehicle=cloud,
        coverage=float(valid.mean()),
    )


def test_reprojected_points_land_on_the_ground_plane():
    """Depth reprojected into the segmentation view must reconstruct z = 0."""
    config = _config()
    depth = _depth_result(config)

    heights = depth.points_vehicle[depth.valid][:, 2]

    assert depth.valid.any()
    assert np.abs(heights).max() < 0.02


def test_reprojection_into_segmentation_view_is_geometrically_correct():
    config = _config()
    _, view_truth, hit, _ = _cast_scene(
        CameraModel(config.segmentation_camera, config.preprocess_width, config.preprocess_height)
    )
    depth = _depth_result(config)

    overlap = depth.valid & hit
    error = np.abs(depth.depth_m[overlap] - view_truth[overlap])

    assert overlap.sum() > 1000
    # Same optical centre as the rectified frame, so this is near-exact.
    assert np.median(error) < 0.05


def test_depth_estimator_returns_none_on_textureless_pair():
    """Flat grey renders give SGBM nothing to match, and that must not crash."""
    config = _config()
    estimator = StereoDepthEstimator(config)
    blank = np.full((360, 640, 3), 128, dtype=np.uint8)

    assert estimator.compute(blank, blank) is None


def test_disparity_rounds_up_to_multiple_of_sixteen():
    estimator = StereoDepthEstimator(_config(stereo_num_disparities=100))

    assert estimator._matcher.getNumDisparities() % 16 == 0
    assert estimator.num_disparities >= 100


def test_disparity_range_is_derived_when_not_pinned():
    """With num_disparities unset the range comes from f, B and min depth."""
    config = _config()
    assert config.stereo_num_disparities == 0

    estimator = StereoDepthEstimator(config)
    rig = estimator.rig

    assert estimator.num_disparities == rig.disparity_range_for(config.stereo_min_depth_m)
    # It must actually reach the configured near limit.
    assert estimator.num_disparities >= rig.disparity_for_depth(config.stereo_min_depth_m)
    assert estimator._matcher.getNumDisparities() == estimator.num_disparities


def test_derived_disparity_tracks_the_lens():
    """A different focal length must re-derive the range, not reuse a constant."""
    from dataclasses import replace as _replace

    from offroad_autonomy.types import GMSL2_SENSOR

    narrow = _replace(GMSL2_SENSOR, fov_x_deg=60.0)
    config = _config()
    wide_range = StereoDepthEstimator(config).num_disparities

    narrow_config = _config(
        left_camera=_replace(config.left_camera, sensor=narrow),
        right_camera=_replace(config.right_camera, sensor=narrow),
        stereo_max_disparities=2048,
    )
    narrow_range = StereoDepthEstimator(narrow_config).num_disparities

    # A narrower lens means a longer focal length, so more disparity per metre.
    assert narrow_range > wide_range


def test_disparity_range_is_capped():
    config = _config(stereo_min_depth_m=0.05, stereo_max_disparities=64)

    assert StereoDepthEstimator(config).num_disparities == 64


def test_rectification_ingests_capture_resolution_frames():
    """Frames arrive at capture size; one remap rectifies AND downscales."""
    config = _config()
    estimator = StereoDepthEstimator(config)
    sensor = config.left_camera.sensor

    rng = np.random.default_rng(3)
    full = rng.integers(0, 255, (sensor.height, sensor.width, 3), dtype=np.uint8)

    rect_left, rect_right = estimator.rectifier.rectify(full, full)

    assert rect_left.shape == (config.stereo_height, config.stereo_width, 3)
    assert rect_right.shape == rect_left.shape
    # The working model is the native one scaled, not a re-derived guess.
    assert estimator.rig.left.scale == pytest.approx(config.stereo_width / sensor.width)
    assert estimator.rig.left.focal_px == pytest.approx(sensor.focal_px * estimator.rig.left.scale)


def test_depth_uses_the_working_focal_length():
    """Z = f*B/d must use the downscaled f, or every depth is 4x wrong."""
    estimator = StereoDepthEstimator(_config())
    rig = estimator.rig

    disparity = np.array([rig.disparity_for_depth(12.0)], dtype=np.float32)

    assert rig.depth_from_disparity(disparity)[0] == pytest.approx(12.0, rel=1e-4)
    assert rig.left.focal_px == pytest.approx(rig.left.sensor.focal_px * 0.75)
    # The rectified focal length is what Z = f*B/d actually uses.
    assert estimator.focal_px == pytest.approx(rig.left.focal_px, rel=1e-3)


def test_ground_plane_recovered_on_flat_terrain():
    config = _config()
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config))

    assert terrain.ground_slope_deg < 0.5
    assert terrain.ground_plane[2] == pytest.approx(0.0, abs=0.05)


def test_ground_plane_tracks_a_tilted_surface():
    """On a climb, height must be measured against the slope, not the horizon."""
    config = _config()
    slope_b = 0.12  # rises toward -Y, i.e. uphill ahead
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config, slope_b=slope_b))

    assert terrain.ground_plane[1] == pytest.approx(slope_b, abs=0.02)
    assert terrain.ground_slope_deg == pytest.approx(np.degrees(np.arctan(slope_b)), abs=1.0)
    # The slope itself must not read as an obstacle field.
    assert terrain.obstacle_fraction < 0.05


def test_raised_slab_is_flagged_as_an_obstacle():
    config = _config()
    slab = ((0.0, -9.0), 0.9, 0.8)  # centre (x, y), half-width, height
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config, slab=slab))

    assert terrain.obstacle_mask.any()
    heights = terrain.height_above_ground[terrain.obstacle_mask]
    assert np.median(heights) > config.obstacle_height_m


def test_flat_ground_produces_no_obstacles():
    config = _config()
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config))

    assert terrain.obstacle_fraction < 0.02


def test_corridor_clearance_reports_distance_to_the_slab():
    config = _config()
    slab = ((0.0, -9.0), 0.9, 0.8)
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config, slab=slab))

    # Slab spans y -9.9..-8.1, so its near face is ~8.1 m ahead.
    assert terrain.min_forward_clearance_m == pytest.approx(8.1, abs=1.0)


def test_corridor_is_clear_without_obstacles():
    config = _config()
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config))

    assert not np.isfinite(terrain.min_forward_clearance_m)


def test_ground_fill_densifies_the_near_field():
    """Where stereo has no depth (here: nearer than min_depth_m, beyond
    max_depth_m), the ground-plane fill covers it without touching measured
    pixels."""
    config = _config()
    depth = _depth_result(config)
    terrain = TerrainAnalyzer(config).analyze(depth)

    assert terrain.inferred is not None
    assert terrain.coverage > terrain.measured_coverage + 0.05
    assert not (terrain.inferred & terrain.measured).any()
    # Fill supplements triangulation, it does not replace it: most of what
    # the pair reports is still directly measured. The bumper mount lowered
    # this - much more of its frame is ground within a couple of metres - which
    # is why depth.min_depth_m came down to 1.5 m with the move.
    assert terrain.measured_coverage > 0.3 * terrain.coverage
    # Inferred pixels sit on the plane, so their height is zero by construction.
    assert np.abs(terrain.height_above_ground[terrain.inferred]).max() < 0.01


def test_ground_fill_never_creates_obstacles():
    config = _config()
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config, slab=((0.0, -9.0), 0.9, 0.8)))

    assert not (terrain.obstacle_mask & terrain.inferred).any()


def test_ground_fill_can_be_disabled():
    config = _config(ground_fill_enabled=False)
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config))

    assert not terrain.inferred.any()
    assert terrain.coverage == pytest.approx(terrain.measured_coverage)


def _terrain_with_slab(config):
    return TerrainAnalyzer(config).analyze(_depth_result(config, slab=((0.0, -9.0), 0.9, 0.8)))


def test_fusion_removes_obstacle_pixels_from_the_mask():
    config = _config()
    terrain = _terrain_with_slab(config)
    mask = np.ones_like(terrain.valid, dtype=bool)
    perception = PerceptionResult(mask=mask)

    fused = fuse_rgb_depth(perception, terrain, config)

    assert fused.mask.sum() < mask.sum()
    assert not (fused.mask & terrain.obstacle_mask).any()


def test_fusion_never_adds_road_the_segmenter_did_not_see():
    """Geometry may veto appearance, never invent traversable ground."""
    config = _config()
    terrain = _terrain_with_slab(config)
    mask = np.zeros_like(terrain.valid, dtype=bool)
    mask[200:260, 280:360] = True
    perception = PerceptionResult(mask=mask)

    fused = fuse_rgb_depth(perception, terrain, config)

    assert not (fused.mask & ~mask).any()


def test_fusion_keeps_the_original_mask_where_depth_is_missing():
    config = _config(ground_fill_enabled=False)
    terrain = TerrainAnalyzer(config).analyze(_depth_result(config))
    mask = np.ones_like(terrain.valid, dtype=bool)

    fused = fuse_rgb_depth(PerceptionResult(mask=mask), terrain, config)

    unmeasured = ~terrain.valid
    assert fused.mask[unmeasured].all()
    # Unmeasured pixels fall back to the configured trust level.
    assert fused.traversability[unmeasured].max() == pytest.approx(
        (1.0 - config.depth_fusion_weight)
        + config.depth_fusion_weight * config.depth_unknown_support,
        abs=1e-5,
    )


def test_fusion_is_a_no_op_on_shape_mismatch():
    config = _config()
    terrain = _terrain_with_slab(config)
    perception = PerceptionResult(mask=np.ones((10, 10), dtype=bool))

    assert fuse_rgb_depth(perception, terrain, config) is perception


def test_fusion_preserves_inference_metadata():
    config = _config()
    terrain = _terrain_with_slab(config)
    perception = PerceptionResult(
        mask=np.ones_like(terrain.valid, dtype=bool),
        confidences=[0.8, 0.6],
        num_detections=2,
        inference_time_ms=12.5,
    )

    fused = fuse_rgb_depth(perception, terrain, config)

    assert fused.confidences == [0.8, 0.6]
    assert fused.num_detections == 2
    assert fused.inference_time_ms == 12.5
    assert fused.rgb_mask is not None
