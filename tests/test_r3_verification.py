"""Physics Envelope And Verification Accounting; Synthetic Fixtures Only."""

import csv

import numpy as np
import pytest

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry
from offroad_autonomy.runtime.r3_verification import analyze_physics, envelope_corners


def test_envelope_contains_full_rotated_volume_and_reference_offset():
    corners = envelope_corners([10, 20, 2], [0, 1, 0], [0, 0, 1], [-2, 3, -1, 1, -0.5, 1.5])[0]
    assert corners.shape == (8, 3)
    assert corners.min(axis=0) == pytest.approx([9, 18, 1.5])
    assert corners.max(axis=0) == pytest.approx([11, 23, 3.5])


def fixture_record(
    tmp_path, offsets, *, interventions=0, faults=0, sequence=None, native_end=None, finish=None
):
    names = [
        "sequence",
        "timestamp_s",
        "dt_s",
        "x",
        "y",
        "z",
        "fx",
        "fy",
        "fz",
        "ux",
        "uy",
        "uz",
        "vx",
        "vy",
        "vz",
        "low_f",
        "high_f",
        "low_r",
        "high_r",
        "low_u",
        "high_u",
        "steering",
        "throttle",
        "brake",
        "parkingbrake",
        "manual_interventions",
        "guard_faults",
        "damage",
    ]
    path = tmp_path / "run_01.physics.csv"
    if sequence is None:
        sequence = range(1, len(offsets) + 1)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for i, (offset, seq) in enumerate(zip(offsets, sequence)):
            writer.writerow(
                [
                    seq,
                    (i + 1) * 0.0005,
                    0.0005,
                    5,
                    offset,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    -2,
                    2,
                    -1,
                    1,
                    0,
                    2,
                    0,
                    0.1,
                    0,
                    0,
                    interventions,
                    faults,
                    0,
                ]
            )
    metadata = {
        "run": "run_01",
        "route_completed": True,
        "end_reason": "Point B Reached",
        "native_end": {
            "sequence": len(offsets),
            "guard_faults": faults,
            "interventions": interventions,
            "buffer_overflow": False,
        },
    }
    if native_end is not None:
        metadata["native_end"].update(native_end)
    if finish is not None:
        metadata["finish_timestamp_s"] = finish
    route = {
        "start_station_m": 5,
        "finish_station_m": 10,
        "association_window_m": 12,
        "max_height_error_m": 3,
        "physical_boundaries_approved": False,
        "mean_absolute_offset_target_m": 0.5,
        "std_signed_offset_target_m": 0.5,
        "max_speed_mph": 10,
    }
    road = RoadGeometry(
        [{"middle": [x, 0, 0], "left": [x, 3, 0], "right": [x, -3, 0]} for x in (0, 10, 20)]
    )
    return analyze_physics(path, metadata, route, road)


def test_proposed_target_pass_is_not_official_compliance(tmp_path):
    summary, rows = fixture_record(tmp_path, [-0.4, 0.4])
    assert summary["mean_absolute_lateral_offset_m"] == pytest.approx(0.4)
    assert summary["std_signed_lateral_offset_m"] == pytest.approx(0.4)
    assert summary["std_absolute_lateral_offset_m"] == pytest.approx(0)
    assert summary["all_proposed_test_conditions"]
    assert summary["zero_manual_inputs_verified"]
    assert not summary["physical_trail_boundaries_verified"]
    assert "Await Mentor Approval" in summary["official_centering_compliance"]
    assert rows.shape == (2, 20)


@pytest.mark.parametrize("issue", ["sequence", "fault", "intervention"])
def test_missing_or_failed_authority_evidence_cannot_verify_zero_intervention(tmp_path, issue):
    kwargs = {}
    if issue == "sequence":
        kwargs["sequence"] = [1, 3]
    if issue == "fault":
        kwargs["faults"] = 1
    if issue == "intervention":
        kwargs["interventions"] = 1
    summary, _ = fixture_record(tmp_path, [0.1, 0.1], **kwargs)
    assert not summary["zero_manual_inputs_verified"]
    assert not summary["all_proposed_test_conditions"]


def test_full_vehicle_envelope_can_depart_with_reference_position_inside(tmp_path):
    summary, _ = fixture_record(tmp_path, [2.1, 2.1])
    assert summary["road_departures"] == 1
    assert not summary["zero_authored_corridor_departures"]


@pytest.mark.parametrize("finish", [None, 0.0005])
def test_intervention_after_last_scored_sample_prevents_zero_input_claim(tmp_path, finish):
    summary, _ = fixture_record(
        tmp_path, [0.1, 0.1], native_end={"interventions": 1}, finish=finish
    )
    assert summary["manual_interventions"] == 1
    assert not summary["zero_manual_inputs_verified"]
    assert not summary["all_proposed_test_conditions"]


@pytest.mark.parametrize(
    "native_end",
    [{"interventions": None}, {"interventions": -1}, {"buffer_overflow": True}],
)
def test_incomplete_native_evidence_cannot_certify_clean_recording(tmp_path, native_end):
    summary, _ = fixture_record(tmp_path, [0.1, 0.1], native_end=native_end)
    assert not summary["physics_recording_integrity"]
    assert not summary["zero_manual_inputs_verified"]


def test_no_unapproved_target_tolerance_is_added(tmp_path):
    summary, _ = fixture_record(tmp_path, [0.500001, 0.500001])
    assert not summary["proposed_mean_target"]
    assert summary["proposed_signed_sd_target"]


def test_batch_envelope_transform_preserves_different_vehicle_poses():
    corners = envelope_corners(
        [[0, 0, 0], [10, 20, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [0, 0, 1]],
        [[-2, 2, -1, 1, 0, 2]] * 2,
    )
    assert corners.shape == (2, 8, 3)
    assert np.max(np.linalg.norm(corners[0, :, :2], axis=1)) == pytest.approx(np.sqrt(5))
    assert corners[1].mean(axis=0) == pytest.approx([10, 20, 1])
