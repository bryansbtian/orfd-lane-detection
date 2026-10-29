"""Visibility-Aware Planner And Controller Regressions."""

from dataclasses import replace

import cv2
import numpy as np

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.baseline_planner import BaselinePlanner
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState


def test_unobserved_near_road_uses_pursuit_instead_of_stanley_gains():
    cfg = PipelineConfig()
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    f = np.linspace(3.3, 13.0, 24)
    plan = PathPlan(centerline=camera.ground_to_image(f, 0.015 * f**2)[::-1])
    first = StanleyController(cfg, camera).compute(plan, VehicleState(speed_mps=2.0))
    changed = replace(cfg, stanley_gain_k=4.0, curvature_feedforward_gain=0.0)
    second = StanleyController(changed, camera).compute(plan, VehicleState(speed_mps=2.0))
    assert first.debug.feedforward_steering == 0
    assert first.steering == second.steering


def test_visible_edge_fit_recovers_center_of_a_cropped_wide_road():
    cfg = PipelineConfig(baseline_require_visible_edges=True)
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    f = np.linspace(1.5, 25.0, 500)
    center = 1.0 + 0.015 * f**2
    left = camera.ground_to_image(f, center - 4)
    right = camera.ground_to_image(f, center + 4)
    polygon = np.r_[left, right[::-1]].astype(np.int32)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    strict = BaselinePlanner(cfg, camera).centerline(mask.astype(bool), 232)
    assert len(strict) > 2
    sf, sr, keep = camera.image_to_ground(strict)
    assert keep.all()
    assert np.mean(np.abs(sr - (1 + 0.015 * sf**2))) < 0.12
    loose = BaselinePlanner(replace(cfg, baseline_require_visible_edges=False), camera).centerline(
        mask.astype(bool), 232
    )
    lf, lr, keep = camera.image_to_ground(loose)
    assert np.mean(np.abs(lr[keep] - (1 + 0.015 * lf[keep] ** 2))) > 0.12


def test_cropped_only_road_cannot_establish_a_centerline():
    cfg = PipelineConfig(baseline_require_visible_edges=True)
    mask = np.ones((cfg.preprocess_height, cfg.preprocess_width), dtype=bool)
    assert len(BaselinePlanner(cfg).centerline(mask, 232)) == 0


def test_weighted_cropped_rows_reduce_bias_without_removing_continuity():
    cfg = PipelineConfig(baseline_censored_row_weight=0.1)
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    f = np.linspace(1.5, 25.0, 500)
    center = 1.0 + 0.015 * f**2
    polygon = np.r_[
        camera.ground_to_image(f, center - 4), camera.ground_to_image(f, center + 4)[::-1]
    ].astype(np.int32)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    errors = []
    for config in (replace(cfg, baseline_censored_row_weight=1.0), cfg):
        points = BaselinePlanner(config, camera).centerline(mask.astype(bool), 232)
        forward, right, keep = camera.image_to_ground(points)
        assert len(points) == 24 and keep.all()
        errors.append(np.mean(np.abs(right - (1 + 0.015 * forward**2))))
    assert errors[1] < errors[0] * 0.75


def test_visible_preview_keeps_adaptation_and_stays_within_observed_path():
    cfg = PipelineConfig(lookahead_from_visible_path=True)
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    forward = np.linspace(3.3, 13.0, 24)
    targets = []
    for right in (np.full(24, 0.2), 0.03 * forward**2):
        plan = PathPlan(centerline=camera.ground_to_image(forward, right)[::-1])
        command = StanleyController(cfg, camera).compute(plan, VehicleState(speed_mps=2.0))
        target, _, keep = camera.image_to_ground([command.debug.lookahead_px])
        assert keep[0]
        assert 3.3 + cfg.lookahead_min_m - 0.05 <= target[0] <= forward[-1]
        targets.append(target[0])
    assert targets[1] < targets[0]
    short = np.linspace(3.3, 4.0, 24)
    command = StanleyController(cfg, camera).compute(
        PathPlan(centerline=camera.ground_to_image(short, np.full(24, 0.2))[::-1]),
        VehicleState(speed_mps=2.0),
    )
    target, _, _ = camera.image_to_ground([command.debug.lookahead_px])
    assert target[0] <= short[-1] + 1e-5
