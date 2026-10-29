"""Independent geometry and evidence accounting, using synthetic fixtures only."""

import math

import numpy as np
import pytest

from offroad_autonomy.runtime.r3_evaluation import (
    RoadGeometry,
    aggregate_runs,
    event_count,
    footprint_departure,
    summarize_run,
    write_csv,
)


def road():
    return RoadGeometry(
        [{"middle": [x, 0, 0], "left": [x, 3, 0], "right": [x, -3, 0]} for x in (0, 10, 20)]
    )


def bbox(x, y, yaw=0.0, half_width=1.0, half_length=2.0):
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    result = {}
    for i, (dx, dy, z) in enumerate(
        (dx, dy, z)
        for dx in (-half_length, half_length)
        for dy in (-half_width, half_width)
        for z in (0, 1)
    ):
        xy = rotation @ [dx, dy] + [x, y]
        result[str(i)] = [*xy, z]
    return result


def measure(x, y, yaw=0.0):
    return road().measure([x, y, 0], bbox(x, y, yaw), x, 10, 2)


def test_shortest_segment_projection_not_nearest_vertex_and_sign():
    left = measure(5, 1.25)
    right = measure(5, -1.25)
    assert left["centerline_x_m"] == pytest.approx(5)
    assert left["centerline_y_m"] == pytest.approx(0)
    assert left["signed_lateral_offset_m"] == pytest.approx(1.25)
    assert right["signed_lateral_offset_m"] == pytest.approx(-1.25)
    assert left["road_departure"] == 0


def test_width_and_heading_can_depart_while_vehicle_center_is_inside():
    assert measure(5, 2.01)["road_departure"] == 1
    assert measure(5, 1.6)["road_departure"] == 0
    rotated = measure(5, 1.6, math.pi / 4)
    assert rotated["road_departure"] == 1
    assert rotated["vehicle_center_upper_limit_m"] == pytest.approx(3 - 3 / math.sqrt(2))


def test_contact_without_crossing_is_not_a_departure():
    assert measure(5, 2)["road_departure"] == 0


def test_variable_width_curved_road_segment_interpolates_in_world_coordinates():
    geometry = RoadGeometry(
        [
            {"middle": [0, 0, 0], "left": [0, 3, 0], "right": [0, -3, 0]},
            {"middle": [10, 0, 1], "left": [8, 2, 1], "right": [12, -2, 1]},
            {"middle": [20, 10, 2], "left": [18, 12, 2], "right": [22, 8, 2]},
        ]
    )
    result = geometry.project([14, 6, 1.5], 17, 10, 2)
    assert result["segment_index"] == 1
    assert result["centerline_x_m"] == pytest.approx(15)
    assert result["centerline_y_m"] == pytest.approx(5)
    assert result["centerline_z_m"] == pytest.approx(1.5)
    assert result["signed_lateral_offset_m"] == pytest.approx(math.sqrt(2))


def test_station_and_height_gates_reject_irrelevant_segments():
    geometry = road()
    with pytest.raises(ValueError, match="No Relevant"):
        geometry.project([19, 0, 0], 0, 1, 2)
    with pytest.raises(ValueError, match="No Relevant"):
        geometry.project([5, 0, 10], 5, 10, 2)


def test_along_road_overrun_is_not_mislabeled_as_lateral_error():
    with pytest.raises(ValueError, match="Beyond Surveyed"):
        road().project([-1, 0, 0], 0, 10, 2)
    with pytest.raises(ValueError, match="Beyond Surveyed"):
        road().project([21, 0, 0], 20, 10, 2)


def test_reversing_a_to_b_reverses_offset_sign_and_station():
    edges = [{"middle": [x, 0, 0], "left": [x, 3, 0], "right": [x, -3, 0]} for x in (20, 10, 0)]
    result = RoadGeometry(edges).measure([5, 1, 0], bbox(5, 1), 15, 10, 2)
    assert result["signed_lateral_offset_m"] == -1
    assert result["route_station_m"] == 15
    assert result["left_boundary_offset_m"] == 3
    assert result["right_boundary_offset_m"] == -3


def test_footprint_edges_detect_concave_notch_when_all_corners_are_inside():
    polygon = np.array([[0, -3], [20, -3], [20, 3], [6, 3], [5, 0], [4, 3], [0, 3]])
    corners = np.array(list(bbox(5, 0.5, half_length=3, half_width=1).values()))
    assert footprint_departure(corners, polygon)


def test_invalid_geometry_rejected_before_measurement():
    with pytest.raises(ValueError, match="Duplicate"):
        RoadGeometry([{"middle": [0, 0, 0], "left": [0, 3, 0], "right": [0, -3, 0]}] * 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        measure(float("nan"), 0)


def samples(offsets, outside=None, manual=None):
    if outside is None:
        outside = [0] * len(offsets)
    if manual is None:
        manual = ["unknown"] * len(offsets)
    return [
        {
            "timestamp_s": i,
            "signed_lateral_offset_m": offset,
            "speed_mph": 5,
            "road_departure": departure,
            "manual_intervention": intervention,
        }
        for i, (offset, departure, intervention) in enumerate(zip(offsets, outside, manual))
    ]


def test_signed_spread_is_not_absolute_error_spread_and_unknown_is_not_zero():
    summary = summarize_run(samples([-1, 1]), {"run": "run_01", "route_completed": True})
    assert summary["mean_absolute_lateral_offset_m"] == 1
    assert summary["std_signed_lateral_offset_m"] == 1
    assert summary["std_absolute_lateral_offset_m"] == 0
    assert summary["manual_interventions"] == "Unknown"
    assert summary["successful_run"] == "Unverified"
    assert summary["observed_pr3_criteria"] == "Unverified"


def test_departures_are_intervals_and_initial_outside_counts_once():
    assert event_count([True, True, False, True, True, False]) == 2
    summary = summarize_run(
        samples([1, 1, 0, 2, 2, 0], [1, 1, 0, 1, 1, 0], [0] * 6),
        {"run": "run_01", "route_completed": True},
    )
    assert summary["road_departures"] == 2
    assert summary["successful_run"] == "No"


def test_completion_and_speed_are_independent_and_no_tolerance_invents_a_pass():
    rows = samples([0, 0], manual=[0, 0])
    rows[1]["speed_mph"] = 10.001
    summary = summarize_run(rows, {"run": "run_01", "route_completed": True})
    assert summary["speed_compliance"] == "Fail"
    assert summary["successful_run"] == "Yes"
    assert summary["observed_pr3_criteria"] == "Fail"
    assert summarize_run(rows, {"run": "run_01"})["route_completed"] == "No"


def test_pooled_statistics_include_between_run_variation_and_weight_by_samples():
    runs = [samples([-1, -1]), samples([1, 1, 1])]
    summaries = [summarize_run(rows, {"run": f"run_{i}"}) for i, rows in enumerate(runs)]
    aggregate = aggregate_runs(runs, summaries)
    assert aggregate["pooled_std_signed_lateral_offset_m"] == pytest.approx(
        np.std([-1, -1, 1, 1, 1])
    )
    assert aggregate["pooled_std_absolute_lateral_offset_m"] == 0
    assert aggregate["manual_status_unknown_runs"] == 2


def test_invalid_data_and_no_execution_do_not_create_results(tmp_path):
    with pytest.raises(ValueError, match="No Executed"):
        aggregate_runs([], [])
    with pytest.raises(ValueError, match="Do Not Create"):
        write_csv(tmp_path / "run_01.csv", [])
    assert not (tmp_path / "run_01.csv").exists()
    rows = samples([0, 0])
    rows[1]["timestamp_s"] = 0
    with pytest.raises(ValueError, match="Timestamps"):
        summarize_run(rows, {"run": "run_01"})
