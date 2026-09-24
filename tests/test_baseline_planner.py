"""Perception gate, baseline planner and the hold -> stop fallback.

The regression these guard against: an empty or low-confidence mask used to
produce a Kalman-extrapolated path that the controller drove at full target
speed. No frame that fails the gate may produce a new path, and once the
held path runs out the controller must brake.
"""

import numpy as np

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.planning.perception_gate import PerceptionGate
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import (
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    StabilizedResult,
    VehicleState,
)

H, W = 465, 720


def _config(**overrides) -> PipelineConfig:
    overrides.setdefault("gate_hold_frames", 3)
    return PipelineConfig(model_weights="dummy.pt", **overrides)


def _road(center: int = W // 2, half_width: int = 150, top: int = 230) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[top:, center - half_width : center + half_width] = True
    return mask


def _stabilized(mask: np.ndarray, confidences=(0.6,)) -> StabilizedResult:
    raw = PerceptionResult(mask=mask, confidences=list(confidences))
    return StabilizedResult(mask=mask, raw_result=raw)


def test_gate_accepts_a_road_in_front_of_the_vehicle():
    decision = PerceptionGate(_config()).evaluate(_stabilized(_road()))
    assert decision.ok, decision.reason
    assert decision.component.any()


def test_gate_rejects_low_confidence():
    decision = PerceptionGate(_config()).evaluate(_stabilized(_road(), confidences=(0.1,)))
    assert not decision.ok
    assert "confidence" in decision.reason


def test_gate_rejects_empty_mask():
    decision = PerceptionGate(_config()).evaluate(
        _stabilized(np.zeros((H, W), bool), confidences=())
    )
    assert not decision.ok


def test_gate_ignores_blobs_that_do_not_reach_the_vehicle():
    mask = np.zeros((H, W), dtype=bool)
    mask[240:300, 100:600] = True  # a strip in the distance, detached
    decision = PerceptionGate(_config()).evaluate(_stabilized(mask))
    assert not decision.ok
    assert decision.reason == "no road at vehicle"


def test_gate_ignores_sky_above_the_roi():
    mask = _road()
    mask[:100, :] = True  # vegetation / sky false positive
    gate = PerceptionGate(_config())
    decision = gate.evaluate(_stabilized(mask))
    assert decision.ok
    assert not decision.component[: gate.roi_top(H)].any()


def test_baseline_path_follows_the_road_centre():
    planner = CenterlinePlanner(_config())
    plan = planner.plan(_stabilized(_road(center=450)))
    assert not plan.fallback_active
    assert len(plan.centerline) >= 10
    np.testing.assert_allclose(plan.centerline[:, 0], 450, atol=2.0)
    # Far -> near, so the controller's centerline[-1] is the nearest point.
    assert np.all(np.diff(plan.centerline[:, 1]) > 0)


def test_small_holes_do_not_split_the_road():
    """Depth-vetoed specks used to split a row and throw the path sideways."""
    mask = _road()
    rng = np.random.default_rng(0)
    for _ in range(40):
        y = int(rng.integers(240, H))
        x = int(rng.integers(W // 2 - 140, W // 2 + 140))
        mask[y : y + 3, x : x + 4] = False
    plan = CenterlinePlanner(_config()).plan(_stabilized(mask))
    assert not plan.fallback_active, plan.fallback_reason
    np.testing.assert_allclose(plan.centerline[:, 0], W / 2, atol=8.0)


def test_path_stops_where_the_road_jumps_sideways():
    """A detached far strip is not a continuation of the road ahead."""
    mask = np.zeros((H, W), dtype=bool)
    mask[300:, W // 2 - 150 : W // 2 + 150] = True  # road ahead, near field
    mask[230:300, W - 120 :] = True  # patch far to the right
    mask[295:300, W // 2 + 150 : W - 120] = True  # joined, so one component
    plan = CenterlinePlanner(_config()).plan(_stabilized(mask))
    assert not plan.fallback_active, plan.fallback_reason
    assert np.all(np.abs(plan.centerline[:, 0] - W / 2) < 40)


def test_failed_gate_holds_last_path_then_stops():
    planner = CenterlinePlanner(_config(gate_hold_frames=2, gate_hold_speed_scale=0.4))
    good = planner.plan(_stabilized(_road(center=400)))
    empty = _stabilized(np.zeros((H, W), bool), confidences=())

    held = [planner.plan(empty) for _ in range(2)]
    for plan in held:
        assert plan.fallback_active
        assert plan.speed_scale == 0.4
        np.testing.assert_array_equal(plan.centerline, good.centerline)

    stopped = planner.plan(empty)
    assert stopped.fallback_active
    assert stopped.speed_scale == 0.0
    assert len(stopped.centerline) == 0

    # And perception coming back resumes planning immediately.
    assert not planner.plan(_stabilized(_road())).fallback_active


def test_no_previous_path_means_stop_not_extrapolate():
    plan = CenterlinePlanner(_config()).plan(_stabilized(np.zeros((H, W), bool), confidences=()))
    assert plan.fallback_active and plan.speed_scale == 0.0
    assert len(plan.centerline) == 0


def test_advanced_mode_is_gated_too():
    planner = CenterlinePlanner(_config(planner_mode="advanced"))
    plan = planner.plan(_stabilized(_road(), confidences=(0.05,)))
    assert plan.fallback_active
    assert not plan.kalman_active


def test_controller_brakes_when_there_is_no_path():
    controller = StanleyController(_config())
    stop = PathPlan(centerline=np.empty((0, 2), np.float32), fallback_active=True, speed_scale=0.0)
    command = controller.compute(stop, VehicleState(speed_mps=5.0))
    assert command.throttle == 0.0
    assert command.brake > 0.5


def test_controller_slows_while_holding():
    config = _config()
    line = np.stack([np.full(20, W / 2.0), np.linspace(240, 440, 20)], axis=1).astype(np.float32)
    full = StanleyController(config).compute(PathPlan(centerline=line), VehicleState(speed_mps=3.0))
    held = StanleyController(config).compute(
        PathPlan(centerline=line, fallback_active=True, speed_scale=0.4),
        VehicleState(speed_mps=3.0),
    )
    assert full.throttle > 0.0
    assert held.throttle == 0.0 and held.brake > 0.0


def test_stabiliser_stages_can_be_switched_off():
    config = _config(enable_ema=False, enable_morphology=False)
    stabilizer = TemporalStabilizer(config)
    mask = _road()
    mask[300, 10] = True  # an isolated pixel morphology would remove
    first = stabilizer.stabilize(PerceptionResult(mask=mask))
    assert first.mask[300, 10]
    # Without EMA an empty frame is empty - nothing is carried over.
    second = stabilizer.stabilize(PerceptionResult(mask=np.zeros((H, W), bool)))
    assert not second.mask.any()
