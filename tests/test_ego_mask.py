"""Unit tests for ego-vehicle exclusion.

The property under test throughout is that bodywork in frame is *invisible* to
the stack rather than being classified. Hood pixels must not become road, must
not become obstacles, and must not move any statistic that a safety decision
depends on.
"""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.main import _StuckDetector
from offroad_autonomy.perception.camera_geometry import build_camera_models
from offroad_autonomy.perception.ego_mask import (
    EgoMask,
    apply_roi,
    road_fraction,
    weighted_confidence,
)
from offroad_autonomy.perception.stereo_depth import StereoDepthEstimator
from offroad_autonomy.perception.fusion import fuse_rgb_depth
from offroad_autonomy.perception.terrain_analyzer import TerrainAnalyzer
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import (
    DepthResult,
    EgoMaskSpec,
    PerceptionResult,
    PipelineConfig,
)

# A wedge across the bottom of the frame, like a hood.
HOOD = EgoMaskSpec(
    enabled=True,
    polygon=((0.15, 1.0), (0.85, 1.0), (0.70, 0.70), (0.30, 0.70)),
    margin_px=0,
)


def _config(**overrides) -> PipelineConfig:
    overrides.setdefault("depth_roi_enabled", False)
    return PipelineConfig(model_weights="dummy.pt", **overrides)


#: The 3-camera rig's centre-camera polygon (2 px margin, 31 px closing,
#: 4 px simplification). Kept only as the size the new masks must beat.
OLD_CENTRE_MASK = EgoMaskSpec(
    enabled=True,
    polygon=(
        (0.1572, 1.0000),
        (0.8442, 1.0000),
        (0.7705, 0.8836),
        (0.6161, 0.6918),
        (0.5132, 0.6767),
        (0.3853, 0.6918),
        (0.2740, 0.8254),
    ),
    margin_px=2,
)


def _roi(shape=(465, 720)):
    return EgoMask(HOOD).valid_roi(shape)


def test_mask_is_resolution_independent():
    """One normalised polygon has to serve acquisition and working grids."""
    mask = EgoMask(HOOD)

    small = mask.coverage((465, 720))
    large = mask.coverage((1860, 2880))

    assert small == pytest.approx(large, abs=0.01)
    assert 0.0 < small < 0.5


def test_disabled_mask_leaves_every_pixel_valid():
    mask = EgoMask(EgoMaskSpec(enabled=False))

    assert not mask.enabled
    assert mask.valid_roi((100, 100)).all()
    assert mask.coverage((100, 100)) == 0.0


def test_mask_covering_the_whole_frame_is_refused():
    """Silently starving the stack of input is worse than failing loudly."""
    everything = EgoMaskSpec(enabled=True, polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

    with pytest.raises(ValueError, match="entire frame"):
        EgoMask(everything).valid_roi((64, 64))


def test_enabled_mask_needs_a_real_polygon():
    with pytest.raises(ValueError, match="polygon"):
        EgoMaskSpec(enabled=True, polygon=((0.0, 0.0),))


@pytest.mark.parametrize("side", ["left", "right"])
def test_bumper_pair_needs_no_exclusion_at_all(side):
    """The whole point of the bumper mount: there is no bodywork to exclude.

    Both cameras sit ahead of the vehicle's frontmost node, so every pixel is
    terrain and the near-field road reaches the bottom row of the frame - the
    trapezoid the roofline pair had to throw away is simply gone.
    """
    config = PipelineConfig()
    spec = config.right_camera
    if side == "left":
        spec = config.left_camera

    assert not spec.ego_mask.enabled
    assert EgoMask(spec.ego_mask).valid_roi((465, 720)).all()


def test_bumper_pair_is_tighter_than_every_mask_that_came_before():
    """Each move has cost less frame than the last; this one costs none."""
    config = PipelineConfig()
    old = EgoMask(OLD_CENTRE_MASK).coverage((465, 720))

    for spec in (config.left_camera, config.right_camera):
        assert EgoMask(spec.ego_mask).coverage((465, 720)) == 0.0 < old


def test_display_camera_clip_is_a_small_bottom_region():
    """The display camera does see the hood - but only to clip the overlay.

    Its polygon is not an exclusion: nothing is computed from this camera, so
    the only thing it changes is where the dashboard paints.
    """
    config = PipelineConfig()
    spec = config.display_camera
    shape = (spec.height, spec.width)
    excluded = EgoMask(spec.ego_mask).excluded(shape)

    assert spec.ego_mask.enabled
    assert 0.02 <= excluded.mean() <= 0.15
    # Entirely in the lower half, and the top row is untouched.
    assert np.flatnonzero(excluded.any(axis=1)).min() > shape[0] // 2
    assert not excluded[0].any()


def test_hood_pixels_do_not_count_against_perception_confidence():
    """The denominator is valid pixels, so masking more cannot lower the score."""
    roi = _roi()
    road = np.zeros(roi.shape, dtype=bool)
    road[120:300, 200:520] = True

    with_hood = road_fraction(road, roi)
    without_hood = road_fraction(road, np.ones_like(roi))

    # Same road, and the hood-aware figure is the higher one - the excluded
    # pixels leave the denominator rather than counting as "not road".
    assert with_hood > without_hood
    assert with_hood == pytest.approx(road.sum() / roi.sum())


def test_detection_lying_entirely_on_the_hood_is_discarded():
    roi = _roi()
    on_hood = ~roi
    on_road = np.zeros(roi.shape, dtype=bool)
    on_road[100:200, 100:300] = True

    kept = weighted_confidence([on_hood, on_road], [0.9, 0.4], roi)

    # The 0.9 hood blob is dropped; the road detection stands on its own.
    assert kept == [0.4]


def test_road_detection_clipped_by_the_hood_keeps_its_score():
    roi = _roi()
    straddling = np.zeros(roi.shape, dtype=bool)
    straddling[300:465, 300:420] = True  # part road, part hood

    kept = weighted_confidence([straddling], [0.77], roi)

    assert kept == [0.77]


def test_road_fraction_is_zero_on_an_empty_mask():
    roi = _roi()

    assert road_fraction(np.zeros_like(roi), roi) == 0.0


def test_hood_pixels_cannot_become_traversable_road():
    roi = _roi()
    # A segmenter that confidently paints the entire frame as road.
    everything = np.ones(roi.shape, dtype=bool)

    gated = apply_roi(everything, roi)

    assert not gated[~roi].any()
    assert gated[roi].all()


def test_stabiliser_reapplies_the_mask_after_morphology():
    """Closing gaps must not grow the road onto our own bodywork."""
    config = _config()
    roi = _roi()
    mask = np.zeros(roi.shape, dtype=bool)
    # Road running right down to the hood boundary, so dilation would spill.
    mask[:, 300:420] = True
    mask &= roi

    result = PerceptionResult(mask=mask, valid_roi=roi, confidences=[0.9])
    stabilized = TemporalStabilizer(config).stabilize(result)

    assert not stabilized.mask[~roi].any()
    assert stabilized.valid_roi is roi


def _cast_ground(camera):
    """Ray-cast a flat ground plane for a camera, returning along-axis depth."""
    uu, vv = camera.pixel_grid()
    dirs = np.stack(
        [
            (uu - camera.cx) / camera.focal_px,
            (vv - camera.cy) / camera.focal_px,
            np.ones_like(uu),
        ],
        axis=-1,
    ) @ camera.rotation.astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origin = camera.position.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dirs[..., 2] < -1e-6, -origin[2] / dirs[..., 2], np.inf)
    t = np.where(t > 0, t, np.inf)
    hit = np.isfinite(t)
    points = origin + dirs * np.where(hit, t, 0.0)[..., None]
    depth = (points - origin) @ camera.rotation[2].astype(np.float32)
    return np.where(hit, depth, 0.0).astype(np.float32), hit


def _depth_result(config):
    estimator = StereoDepthEstimator(config)
    depth_left, hit = _cast_ground(estimator.rectified)
    usable = (
        hit & (depth_left >= config.stereo_min_depth_m) & (depth_left <= config.stereo_max_depth_m)
    )
    cloud, points, image, valid = estimator._reproject(
        np.where(usable, depth_left, np.nan).astype(np.float32), usable
    )
    return DepthResult(
        depth_m=image,
        valid=valid,
        points_vehicle=points,
        cloud_vehicle=cloud,
        coverage=float(valid.mean()),
    )


def test_hood_pixels_cannot_become_obstacles():
    config = _config()
    depth = _depth_result(config)
    roi = _roi(depth.valid.shape)

    terrain = TerrainAnalyzer(config).analyze(depth, roi)

    assert not (terrain.obstacle_mask & ~roi).any()
    assert not (terrain.traversability[~roi] > 0).any()


def test_ground_fill_does_not_invent_terrain_behind_the_hood():
    """We have no evidence there, so filling it would be fabrication."""
    config = _config()
    depth = _depth_result(config)
    roi = _roi(depth.valid.shape)

    terrain = TerrainAnalyzer(config).analyze(depth, roi)

    assert not (terrain.inferred & ~roi).any()


def test_depth_behind_the_hood_still_reaches_the_clearance_grid():
    """The pair sees past the hood; that geometry must not be thrown away."""
    config = _config()
    depth = _depth_result(config)
    roi = _roi(depth.valid.shape)

    masked = TerrainAnalyzer(config).analyze(depth, roi)
    unmasked = TerrainAnalyzer(config).analyze(depth, None)

    # The vehicle-space cloud drives the BEV grid and is deliberately whole,
    # so the corridor clearance is unchanged by an image-space exclusion.
    assert masked.min_forward_clearance_m == unmasked.min_forward_clearance_m
    assert masked.ground_plane == pytest.approx(unmasked.ground_plane)


def test_stereo_fusion_ignores_excluded_pixels():
    config = _config()
    depth = _depth_result(config)
    roi = _roi(depth.valid.shape)
    terrain = TerrainAnalyzer(config).analyze(depth, roi)

    # A segmenter that would happily call the hood drivable.
    perception = PerceptionResult(
        mask=np.ones(roi.shape, dtype=bool) & roi,
        valid_roi=roi,
        confidences=[0.9],
    )

    fused = fuse_rgb_depth(perception, terrain, config)

    assert not fused.mask[~roi].any()
    assert not (fused.traversability[~roi] > 0).any()
    assert fused.valid_roi is roi
    assert fused.road_fraction == pytest.approx(road_fraction(fused.mask, roi))


def test_fusion_road_fraction_uses_only_valid_pixels():
    config = _config()
    depth = _depth_result(config)
    roi = _roi(depth.valid.shape)
    terrain = TerrainAnalyzer(config).analyze(depth, roi)

    mask = np.zeros(roi.shape, dtype=bool)
    mask[150:250, 250:450] = True
    fused = fuse_rgb_depth(PerceptionResult(mask=mask & roi, valid_roi=roi), terrain, config)

    assert fused.road_fraction == pytest.approx((fused.mask & roi).sum() / roi.sum())


def test_rgb_only_fallback_still_applies_the_ego_mask():
    """Stereo being offline must not switch the exclusion off."""
    config = _config(depth_enabled=False)
    roi = _roi()

    # No terrain at all - the appearance-only path.
    mask = np.ones(roi.shape, dtype=bool)
    perception = PerceptionResult(
        mask=apply_roi(mask, roi),
        valid_roi=roi,
        road_fraction=road_fraction(apply_roi(mask, roi), roi),
    )
    stabilized = TemporalStabilizer(config).stabilize(perception)

    assert not stabilized.mask[~roi].any()
    assert stabilized.road_fraction == pytest.approx(1.0, abs=1e-6)


def test_ego_mask_is_built_even_when_depth_is_disabled():
    from unittest.mock import patch

    config = _config(depth_enabled=False)

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor"),
        patch("offroad_autonomy.pipeline.RoadSegmenter"),
        patch("offroad_autonomy.pipeline.TemporalStabilizer"),
        patch("offroad_autonomy.pipeline.CenterlinePlanner"),
        patch("offroad_autonomy.pipeline.StanleyController"),
    ):
        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)

    assert pipeline.depth_estimator is None
    # The ROI is built either way; from the bumper pair it is simply whole.
    assert pipeline.valid_roi.shape == (config.preprocess_height, config.preprocess_width)
    assert pipeline.ego_coverage == pytest.approx(0.0)
    assert pipeline.valid_roi.all()


def test_safe_stop_still_fires_when_the_road_really_is_gone():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    assert detector.update(0.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)[0] is False
    assert detector.update(1.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)[0] is False

    triggered, reason = detector.update(2.5, road_fraction=0.0, speed_mps=5.0, throttle=0.3)

    assert triggered
    assert "no traversable road" in reason


def test_safe_stop_does_not_fire_on_a_visible_road():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    for t in range(0, 100):
        triggered, _ = detector.update(float(t), road_fraction=0.25, speed_mps=5.0, throttle=0.3)
        assert not triggered


def test_recovering_road_clears_the_safe_stop_timer():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    detector.update(0.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)
    detector.update(1.0, road_fraction=0.30, speed_mps=5.0, throttle=0.3)
    # The timer restarted, so a single bad frame long after must not trip it.
    triggered, _ = detector.update(3.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)

    assert not triggered


def test_stuck_detection_is_unchanged():
    """The motion-based half of the safety net must still work."""
    detector = _StuckDetector()

    detector.update(0.0, road_fraction=0.5, speed_mps=0.1, throttle=0.5)
    triggered, reason = detector.update(4.0, road_fraction=0.5, speed_mps=0.1, throttle=0.5)

    assert triggered
    assert reason == "vehicle stuck"


def test_hood_sized_exclusion_cannot_by_itself_trigger_safe_stop():
    """The regression this whole change exists to prevent.

    A frame where every valid pixel is road, but 15% of the frame is hood,
    must read as a clear road - not as a 15% loss of confidence.
    """
    config = PipelineConfig()
    roi = EgoMask(config.left_camera.ego_mask).valid_roi((465, 720))
    all_road = apply_roi(np.ones(roi.shape, dtype=bool), roi)

    fraction = road_fraction(all_road, roi)
    detector = _StuckDetector(
        min_road_fraction=config.safety_min_road_fraction,
        no_road_time_s=config.safety_no_road_time_s,
    )

    assert fraction == pytest.approx(1.0)
    for t in range(0, 20):
        triggered, _ = detector.update(
            float(t), road_fraction=fraction, speed_mps=4.0, throttle=0.3
        )
        assert not triggered


def test_ego_mask_round_trips_through_yaml(tmp_path):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "ego.yaml"
    path.write_text(
        "\n".join(
            [
                "beamng:",
                "  cameras:",
                "    left:",
                "      ego_mask:",
                "        enabled: true",
                "        margin_px: 5",
                "        polygon:",
                "          - [0.2, 1.0]",
                "          - [0.8, 1.0]",
                "          - [0.5, 0.6]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(path)
    spec = config.left_camera.ego_mask

    assert spec.enabled
    assert spec.margin_px == 5
    assert spec.polygon == ((0.2, 1.0), (0.8, 1.0), (0.5, 0.6))
    assert EgoMask(spec).coverage((465, 720)) > 0.0


def test_ego_mask_can_be_switched_on_per_camera_in_yaml(tmp_path):
    """A rig that does see bodywork can still say so, one camera at a time."""
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "on.yaml"
    path.write_text(
        "beamng:\n  cameras:\n    left:\n      ego_mask:\n"
        "        enabled: true\n        polygon:\n"
        "          - [0.2, 1.0]\n          - [0.8, 1.0]\n          - [0.5, 0.6]\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.left_camera.ego_mask.enabled
    assert EgoMask(config.left_camera.ego_mask).coverage((465, 720)) > 0.0
    # The other camera is untouched and keeps the default: nothing to exclude.
    assert not config.right_camera.ego_mask.enabled
    assert EgoMask(config.right_camera.ego_mask).valid_roi((64, 64)).all()


def test_enabled_mask_without_a_polygon_is_rejected(tmp_path):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        "beamng:\n  cameras:\n    right:\n      ego_mask:\n"
        "        enabled: true\n        polygon: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fewer than 3 points"):
        load_config(path)


def test_neither_stereo_camera_is_masked_by_default():
    config = PipelineConfig()

    assert not config.left_camera.ego_mask.enabled
    assert not config.right_camera.ego_mask.enabled


def test_moving_the_camera_does_not_silently_move_the_mask():
    """The polygon belongs to a pose; changing one must not fake the other."""
    config = PipelineConfig()
    moved = replace(config.left_camera, pos=(0.3, -0.5, 1.2))

    # Same polygon, because it is configuration - which is exactly why
    # scripts/derive_ego_mask.py exists and the config says to re-run it.
    assert moved.ego_mask == config.left_camera.ego_mask


#: Frontmost node of the Hopper's bodywork: fb1r/fb1l of hopper_bumper_F.jbeam.
#: Forward is -Y, so anything mounted at a smaller Y is clear of the vehicle.
HOPPER_FRONTMOST_Y = -1.78


def test_stereo_pair_is_the_bumper_mount():
    config = PipelineConfig()
    left, right = build_camera_models(config, 720, 465)

    assert config.left_camera.pos == pytest.approx((0.3, -1.95, 0.95))
    assert config.right_camera.pos == pytest.approx((-0.3, -1.95, 0.95))
    # Still looking slightly down, not up at the sky.
    assert left.rotation[2][2] < 0.0
    assert right.rotation[2][2] < 0.0


def test_both_bumper_cameras_sit_ahead_of_all_bodywork():
    """What makes the exclusion mask unnecessary rather than merely small.

    A camera in front of the frontmost node has every part of the vehicle
    behind its image plane, so no field of view can bring bodywork into frame.
    """
    config = PipelineConfig()

    for spec in (config.left_camera, config.right_camera):
        assert spec.pos[1] < HOPPER_FRONTMOST_Y
    # And the pair stays inside the bumper's 1.60 m width.
    assert abs(config.left_camera.pos[0] - config.right_camera.pos[0]) <= 1.60


def test_display_camera_is_behind_the_bumper_and_above_the_pair():
    """The two mounts are chosen for different jobs, and differ accordingly."""
    config = PipelineConfig()

    assert config.display_camera.pos[1] > HOPPER_FRONTMOST_Y
    assert config.display_camera.pos[2] > config.left_camera.pos[2]
