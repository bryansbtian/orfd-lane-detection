"""Failures must be visible at the RGB planning boundary, without actuation."""

import time
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.corridor_validation import validate_corridor
from offroad_autonomy.planning.perception_gate import PerceptionGate
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.runtime.rgb_evaluation import MaskMetrics, manual_path_metrics
from offroad_autonomy.types import (
    DEFAULT_CAMERA,
    CameraFrame,
    PerceptionResult,
    PipelineConfig,
    StabilizedResult,
)


def test_high_confidence_distant_blob_cannot_certify_a_low_confidence_corridor():
    mask = np.zeros((100, 160), bool)
    mask[50:, 50:110] = True
    mask[51:60, 120:150] = True
    probability = np.zeros_like(mask, dtype=np.float32)
    probability[50:, 50:110] = 0.2
    probability[51:60, 120:150] = 0.99
    raw = PerceptionResult(mask=mask, confidences=[0.99], confidence_map=probability)
    result = PerceptionGate(PipelineConfig(gate_min_confidence=0.6)).evaluate(
        StabilizedResult(mask=mask, raw_result=raw)
    )
    assert not result.ok


def test_ema_cannot_bridge_a_corridor_absent_from_current_frame():
    mask = np.ones((100, 160), bool)
    raw = PerceptionResult(mask=np.zeros_like(mask), confidences=[0.99])
    result = PerceptionGate(PipelineConfig()).evaluate(StabilizedResult(mask=mask, raw_result=raw))
    assert not result.ok
    assert not result.component.any()


def test_grid_evidence_excludes_disconnected_and_low_confidence_road():
    mask = np.zeros((100, 160), bool)
    mask[30:, 60:100] = True
    mask[30:40, 120:150] = True
    probability = mask.astype(np.float32) * 0.9
    probability[:, 60:65] = 0.1
    raw = PerceptionResult(mask=mask, confidences=[0.9], confidence_map=probability)
    stabilized = StabilizedResult(mask=mask, raw_result=raw)
    gate = PerceptionGate(PipelineConfig())
    decision = gate.evaluate(stabilized)
    assert decision.ok
    observation = gate.connected_evidence(stabilized, decision.component)
    assert not observation[:, 60:65].any()
    assert not observation[:, 120:].any()
    assert observation[35, 80]


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_confidence_fails_closed(invalid):
    mask = np.ones((100, 160), bool)
    raw = PerceptionResult(mask=mask, confidences=[invalid])
    assert (
        not PerceptionGate(PipelineConfig())
        .evaluate(StabilizedResult(mask=mask, raw_result=raw))
        .ok
    )


def test_validation_checks_between_waypoints_and_requires_clearance():
    mask = np.zeros((100, 160), bool)
    mask[10:95, 40:120] = True
    path = np.array([[80, 15], [80, 90]], dtype=float)
    assert validate_corridor(path, mask, 3).ok
    mask[50:55, 70:90] = False
    assert not validate_corridor(path, mask).ok
    near_boundary = np.array([[40, 15], [40, 90]], dtype=float)
    assert not validate_corridor(near_boundary, mask, 3).ok


def test_validation_rejects_nonfinite_and_out_of_frame_paths():
    mask = np.ones((100, 160), bool)
    assert not validate_corridor(np.array([[80, 10], [np.nan, 90]]), mask).ok
    assert not validate_corridor(np.array([[-1, 10], [80, 90]]), mask).ok


def test_independent_manual_path_metric_detects_a_gap_between_waypoints():
    truth = np.ones((100, 160), bool)
    truth[50:55, 75:85] = False
    path = np.array([[80, 15], [80, 90]])
    result = manual_path_metrics(path, truth, np.ones_like(truth), (160, 100))
    assert result["path_on_manual_road_fraction"] < 1
    assert result["minimum_manual_clearance_px"] == 0


def test_calibrated_intrinsics_survive_nonuniform_resize():
    spec = replace(DEFAULT_CAMERA, intrinsics=(700, 680, 470, 300))
    camera = CameraModel(spec, 480, 200)
    assert camera.focal_px == 350
    assert camera.focal_y_px == pytest.approx(680 * 200 / 620)
    assert camera.cx == 235
    assert camera.cy == pytest.approx(300 * 200 / 620)
    forward = np.array([4.0, 7.0, 12.0])
    right = np.array([0.2, -0.5, 1.0])
    pixels = camera.ground_to_image(forward, right)
    actual_f, actual_r, valid = camera.image_to_ground(pixels)
    assert valid.all()
    np.testing.assert_allclose(actual_f, forward, atol=1e-5)
    np.testing.assert_allclose(actual_r, right, atol=1e-5)


def test_projection_rejects_far_rays_under_pitch_uncertainty():
    camera = CameraModel(DEFAULT_CAMERA, 720, 465)
    pixels = camera.ground_to_image(np.array([4.0, 28.0]), np.zeros(2))
    _, _, valid = camera.reliable_ground(pixels, pitch_uncertainty_deg=2, max_relative_error=0.3)
    np.testing.assert_array_equal(valid, [True, False])


def test_disabled_ema_does_not_allocate_a_float_history():
    stabilizer = TemporalStabilizer(PipelineConfig(enable_ema=False, enable_morphology=False))
    mask = np.ones((100, 160), bool)
    np.testing.assert_array_equal(stabilizer.stabilize(PerceptionResult(mask=mask)).mask, mask)
    assert stabilizer._accum is None


def test_centerline_inside_a_narrow_road_cannot_certify_vehicle_width():
    import cv2

    camera = CameraModel(DEFAULT_CAMERA, 720, 465)
    forward = np.linspace(4, 12, 20)
    path = camera.ground_to_image(forward, np.zeros(20))[::-1]
    mask = np.zeros((465, 720), np.uint8)
    left = camera.ground_to_image(np.array([3.0, 15.0]), np.array([-0.5, -0.5]))
    right = camera.ground_to_image(np.array([15.0, 3.0]), np.array([0.5, 0.5]))
    cv2.fillPoly(mask, [np.rint(np.vstack([left, right])).astype(np.int32)], 1)
    assert validate_corridor(path, mask.astype(bool)).ok
    assert not validate_corridor(path, mask.astype(bool), camera=camera, half_width_m=0.95).ok


def test_margin_alone_does_not_claim_vehicle_width_was_validated():
    from offroad_autonomy.planning.centerline_planner import CenterlinePlanner

    config = PipelineConfig(path_half_width_m=0, path_boundary_margin_m=0.2)
    mask = np.ones((465, 720), bool)
    raw = PerceptionResult(mask=mask, confidences=[0.9])
    plan = CenterlinePlanner(config).plan(StabilizedResult(mask=mask, raw_result=raw))
    assert plan.valid
    assert plan.status == "centerline_only"
    assert not plan.vehicle_width_validated


def test_metrics_exclude_void_and_do_not_reward_empty_road_predictions():
    truth = np.array([[True, False], [False, True]])
    valid = np.array([[True, True], [False, True]])
    metrics = MaskMetrics()
    metrics.add(truth, truth, valid)
    assert metrics.report()["miou"] == 1
    assert metrics.report()["temporal_pairs"] == 0
    metrics.add(np.zeros_like(truth), truth, valid)
    report = metrics.report()
    assert report["tiou_unwarped"] == 0
    assert report["label_relative_change_agreement"] == 0
    assert sum(report["confusion_tn_fp_fn_tp"]) == 6
    metrics.reset_sequence()
    metrics.add(truth, truth, valid)
    assert metrics.report()["temporal_pairs"] == 1


@pytest.mark.parametrize("age", [1.0, -1.0, float("nan"), float("inf")])
def test_rgb_planning_has_no_actuation_and_invalidates_stale_output(age):
    from offroad_autonomy.pipeline import RGBPlanningPipeline

    config = PipelineConfig(max_frame_age_ms=50)
    with patch("offroad_autonomy.pipeline.RoadSegmenter") as segmenter:
        segmenter.return_value.predict.return_value = PerceptionResult(
            mask=np.zeros((465, 720), bool)
        )
        pipeline = RGBPlanningPipeline(config)
        result = pipeline.plan_result(
            CameraFrame(np.zeros((620, 960, 3), np.uint8), time.perf_counter() - age)
        )
    assert not hasattr(pipeline, "controller")
    assert not result.plan.valid
    assert result.plan.status == "stale"
    if np.isfinite(age):
        assert abs(result.frame_age_ms) >= 900
        assert result.frame.timestamp == result.capture.timestamp


def test_real_camera_calibration_and_safety_settings_load_without_simulator(tmp_path):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "recorded.yaml"
    path.write_text(
        "camera:\n  intrinsics: [700, 680, 470, 300]\nplanning:\n  path_min_clearance_px: 3\nruntime:\n  max_frame_age_ms: 80\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.camera.intrinsics == (700, 680, 470, 300)
    assert config.path_min_clearance_px == 3
    assert config.max_frame_age_ms == 80


@pytest.mark.parametrize(
    "setting",
    [
        "path_min_clearance_px: -1",
        "projection_max_range_m: 0",
        "projection_max_relative_error: .nan",
    ],
)
def test_invalid_planning_limits_are_refused(tmp_path, setting):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "invalid.yaml"
    path.write_text(f"planning:\n  {setting}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
