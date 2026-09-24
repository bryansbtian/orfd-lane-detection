"""Tests for the display-only camera and its reprojection.

Two properties are worth pinning down, because both are easy to break by
accident and neither shows up as a crash:

1. **The display camera is display only.** Its frame must not reach
   segmentation, stereo, depth, planning or control by any route.
2. **The main panel reuses the perception result rather than recomputing it.**
   The mask and path drawn on the display camera come from the bumper
   pipeline through a homography; nothing is inferred a second time.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import CameraModel, build_camera_models
from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker
from offroad_autonomy.types import (
    DEFAULT_DISPLAY_RIG,
    DisplayRigSpec,
    PipelineConfig,
    display_camera_spec,
)
from offroad_autonomy.visualization.path_projector import (
    GroundProjector,
    ground_plane_matrix,
)


def _models(config: PipelineConfig | None = None):
    """The (perception source, display target) pair the dashboard uses."""
    if config is None:
        config = PipelineConfig(model_weights="dummy.pt")
    source = CameraModel(config.left_camera, config.preprocess_width, config.preprocess_height)
    spec = config.display_camera
    return source, CameraModel(spec, spec.width, spec.height)


def test_display_camera_is_not_part_of_the_stereo_pair():
    """``build_camera_models`` is the only door into stereo, and it has two."""
    config = PipelineConfig(model_weights="dummy.pt")
    left, right = build_camera_models(config, 720, 465)

    for model in (left, right):
        assert model.spec.name != config.display_camera.name
        assert tuple(model.position) != tuple(config.display_camera.pos)


def test_segmentation_camera_is_never_the_display_camera():
    """Whatever the mode, the segmented frame comes from a bumper camera."""
    for mode in ("left", "right"):
        config = PipelineConfig(model_weights="dummy.pt", segmentation_mode=mode)
        assert config.segmentation_camera is not config.display_camera
        assert config.segmentation_camera in (config.left_camera, config.right_camera)


def test_display_camera_is_absent_from_the_stereo_frame_pair():
    """There is no third slot on the type stereo and segmentation read from."""
    from offroad_autonomy.types import StereoFramePair

    fields = set(StereoFramePair.__dataclass_fields__)
    assert {"left", "right"} <= fields
    assert not {"display", "visualization", "path_view"} & fields


def test_disabling_the_display_camera_leaves_perception_untouched():
    config = PipelineConfig(
        model_weights="dummy.pt", display_rig=replace(DEFAULT_DISPLAY_RIG, enabled=False)
    )
    reference = PipelineConfig(model_weights="dummy.pt")

    assert config.left_camera.pos == reference.left_camera.pos
    assert config.right_camera.pos == reference.right_camera.pos


def test_ground_matrix_agrees_with_the_camera_it_came_from():
    """The homography's building block must project like the camera does."""
    _, target = _models()
    matrix = ground_plane_matrix(target, ground_z=0.0)

    for x, y in ((0.0, -10.0), (2.0, -25.0), (-3.0, -6.0)):
        point = np.array([[x, y, 0.0]], dtype=np.float64)
        u_ref, v_ref, z_ref = target.project(target.from_vehicle(point))
        assert z_ref[0] > 0.0

        homogeneous = matrix @ np.array([x, y, 1.0])
        assert homogeneous[2] == pytest.approx(z_ref[0], rel=1e-4)
        assert homogeneous[0] / homogeneous[2] == pytest.approx(u_ref[0], abs=0.01)
        assert homogeneous[1] / homogeneous[2] == pytest.approx(v_ref[0], abs=0.01)


def test_a_ground_point_lands_in_the_same_place_in_both_views():
    """The whole reprojection rests on this: one world point, two pictures."""
    source, target = _models()
    projector = GroundProjector(source, target)

    for x, y in ((0.0, -8.0), (1.5, -14.0), (-2.0, -20.0)):
        point = np.array([[x, y, 0.0]], dtype=np.float64)
        u_s, v_s, z_s = source.project(source.from_vehicle(point))
        u_t, v_t, z_t = target.project(target.from_vehicle(point))
        assert z_s[0] > 0.0 and z_t[0] > 0.0

        moved = projector.project_points(np.array([[u_s[0], v_s[0]]]))
        assert len(moved) == 1
        assert moved[0][0] == pytest.approx(u_t[0], abs=1.0)
        assert moved[0][1] == pytest.approx(v_t[0], abs=1.0)


def test_warped_mask_keeps_road_on_the_road():
    """A mask covering the ground corridor must still cover it afterwards."""
    source, target = _models()
    projector = GroundProjector(source, target)

    # Everything the source camera sees of the ground, marked as road.
    mask = projector.ground_valid.copy()
    warped = projector.warp_mask(mask)

    assert warped.any()
    # The transferred region sits below the target's horizon, never in the sky.
    horizon = target.cy - target.focal_px * math.tan(math.radians(-DEFAULT_DISPLAY_RIG.pitch_deg))
    rows = np.flatnonzero(warped.any(axis=1))
    assert rows.min() >= horizon - 2


def test_sky_is_never_transferred():
    """Pixels above the horizon have no ground point and must be dropped."""
    source, target = _models()
    projector = GroundProjector(source, target)

    # The top row of the source looks at or above the horizon.
    assert not projector.ground_valid[0].any()

    sky = np.zeros((source.height, source.width), dtype=bool)
    sky[: source.height // 4] = True
    assert not projector.warp_mask(sky).any()


def test_projection_survives_an_empty_path():
    source, target = _models()
    projector = GroundProjector(source, target)

    assert len(projector.project_points(np.empty((0, 2)))) == 0


def test_a_camera_aimed_at_the_sky_is_refused_not_silently_wrong():
    """No ground plane, no homography - and the dashboard falls back instead."""
    config = PipelineConfig(model_weights="dummy.pt")
    source, _ = _models(config)
    up_rig = DisplayRigSpec(pitch_deg=45.0)
    spec = display_camera_spec(
        up_rig, config.display_camera.sensor, replace(config.display_camera.ego_mask, enabled=False)
    )
    skyward = CameraModel(spec, spec.width, spec.height)

    with pytest.raises(ValueError):
        GroundProjector(skyward, source)


def _worker(**kwargs) -> tuple[DisplayWorker, list]:
    drawn: list = []

    def render(state, frame):
        drawn.append((state, frame))
        return np.zeros((4, 4, 3), dtype=np.uint8)

    kwargs.setdefault("show", lambda frame: True)
    kwargs.setdefault("read_key", lambda: -1)
    return DisplayWorker(render=render, asynchronous=False, rate_hz=0.0, **kwargs), drawn


def test_display_frame_reaches_rendering_and_nothing_else():
    """The captured frame goes to the renderer - it is not handed back."""
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    worker, drawn = _worker(capture=lambda: frame)
    state = DisplayState(result=object(), telemetry=object())

    worker.publish(state)

    assert len(drawn) == 1
    assert drawn[0][0] is state
    assert drawn[0][1] is frame


def test_a_failing_dashboard_does_not_raise_into_the_control_loop():
    """Rendering is best effort: a broken window must not stop the vehicle."""

    def explode(state, frame):
        raise RuntimeError("no window")

    worker = DisplayWorker(
        render=explode,
        show=lambda f: True,
        read_key=lambda: -1,
        asynchronous=False,
        rate_hz=0.0,
    )

    worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.failed == 1
    assert worker.rendered == 0


def test_keys_are_queued_for_the_control_loop_to_drain():
    keys = iter([ord("e"), ord("p"), -1])
    worker, _ = _worker(read_key=lambda: next(keys))

    for _ in range(3):
        worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.drain_keys() == [ord("e"), ord("p")]
    assert worker.drain_keys() == []


def test_publishing_never_blocks_when_asynchronous():
    """Snapshots are overwritten, not queued, so the loop cannot back up."""
    worker, _ = _worker()
    worker._async = True  # without starting the thread, nothing consumes

    for _ in range(50):
        worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.dropped == 49
    assert worker._pending is not None


def test_no_display_camera_means_no_capture_call():
    """With the camera disabled the worker renders on the perception view."""
    worker, drawn = _worker(capture=None)

    worker.publish(DisplayState(result=object(), telemetry=object()))

    assert drawn[0][1] is None


def test_worker_draws_on_a_background_thread_without_the_caller_waiting():
    """The whole point: publish() returns, and drawing happens elsewhere."""
    import threading

    drew = threading.Event()
    seen: dict = {}

    def render(state, frame):
        seen["thread"] = threading.current_thread().name
        drew.set()
        return np.zeros((4, 4, 3), dtype=np.uint8)

    worker = DisplayWorker(
        render=render,
        show=lambda f: True,
        read_key=lambda: -1,
        asynchronous=True,
        rate_hz=0.0,
    )
    worker.start()
    try:
        worker.publish(DisplayState(result=object(), telemetry=object()))
        assert drew.wait(timeout=5.0), "worker never drew"
    finally:
        worker.stop()

    assert seen["thread"] != threading.current_thread().name
    assert seen["thread"] == "display-worker"
    assert worker.rendered >= 1
    assert worker.stats.stage("dashboard_total").count >= 1
