"""Independent R3 geometry and statistics in BeamNG world XY metres."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

_EPS = 1e-8


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_status(point: np.ndarray, polygon: np.ndarray) -> int:
    """Return -1 outside, 0 on the boundary, 1 inside a simple polygon."""
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    delta = end - start
    length2 = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.sum((point - start) * delta, axis=1) / length2, 0, 1)
    if np.min(np.linalg.norm(point - (start + fraction[:, None] * delta), axis=1)) <= _EPS:
        return 0
    active = (start[:, 1] > point[1]) != (end[:, 1] > point[1])
    intersections = (
        start[active, 0] + (point[1] - start[active, 1]) * delta[active, 0] / (delta[active, 1])
    )
    if np.count_nonzero(intersections > point[0]) % 2:
        return 1
    return -1


def _proper_intersection(a, b, c, d) -> bool:
    first = _cross(b - a, c - a) * _cross(b - a, d - a)
    second = _cross(d - c, a - c) * _cross(d - c, b - c)
    return bool(first < -_EPS and second < -_EPS)


def footprint_departure(corners: np.ndarray, polygon: np.ndarray) -> bool:
    """Check the complete convex world-XY footprint, including its edges."""
    points = np.unique(np.asarray(corners, dtype=float)[:, :2], axis=0)
    hull = points[ConvexHull(points).vertices]
    if any(_point_status(point, polygon) < 0 for point in hull):
        return True
    # Corners alone miss a concave road edge entering and leaving the footprint.
    if any(_point_status(point, hull) > 0 for point in polygon):
        return True
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        for c, d in zip(polygon, np.roll(polygon, -1, axis=0)):
            if _proper_intersection(a, b, c, d):
                return True
    return False


class RoadGeometry:
    """A single, explicitly selected road ribbon, ordered from A toward B.

    Restrict association by longitudinal station and height to avoid silently
    selecting a neighbouring branch, crossing road, or vertically stacked road.
    """

    def __init__(self, edges: list[dict]) -> None:
        self.center = np.asarray([edge["middle"] for edge in edges], dtype=float)
        left = np.asarray([edge["left"] for edge in edges], dtype=float)
        right = np.asarray([edge["right"] for edge in edges], dtype=float)
        if self.center.ndim != 2 or self.center.shape[1] != 3 or len(self.center) < 2:
            raise ValueError("Road Requires At Least Two XYZ Cross Sections")
        if not all(np.isfinite(array).all() for array in (self.center, left, right)):
            raise ValueError("Road Geometry Must Be Finite")
        self.delta = np.diff(self.center[:, :2], axis=0)
        self.length = np.linalg.norm(self.delta, axis=1)
        if np.any(self.length <= _EPS):
            raise ValueError("Duplicate Or Vertical Centerline Segments Are Unsupported")
        self.station = np.r_[0.0, np.cumsum(self.length)]
        self.left = left
        self.right = right
        polygon = np.vstack((left[:, :2], right[::-1, :2]))
        keep = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > _EPS
        self.polygon = polygon[keep]
        if len(self.polygon) < 4:
            raise ValueError("Road Boundaries Have Zero Area")
        # A self-intersecting ribbon has no unambiguous inside. Fail instead of
        # reporting a favourable boundary count from a broken polygon.
        count = len(self.polygon)
        for i in range(count):
            for j in range(i + 2, count):
                if i == 0 and j == count - 1:
                    continue
                if _proper_intersection(
                    self.polygon[i],
                    self.polygon[(i + 1) % count],
                    self.polygon[j],
                    self.polygon[(j + 1) % count],
                ):
                    raise ValueError("Self-Intersecting Road Ribbon Requires A Surveyed Corridor")

    def project(
        self,
        position: list[float],
        expected_station_m: float,
        association_window_m: float,
        max_height_error_m: float,
    ) -> dict:
        point = np.asarray(position, dtype=float)
        if point.shape != (3,):
            raise ValueError("Ground Truth Requires XYZ Position")
        if not np.isfinite(point).all():
            raise ValueError("Nonfinite Ground Truth")
        fraction = np.clip(
            np.sum((point[:2] - self.center[:-1, :2]) * self.delta, axis=1) / self.length**2,
            0,
            1,
        )
        nearest = self.center[:-1] + fraction[:, None] * np.diff(self.center, axis=0)
        station = self.station[:-1] + fraction * self.length
        eligible = (np.abs(station - expected_station_m) <= association_window_m) & (
            np.abs(nearest[:, 2] - point[2]) <= max_height_error_m
        )
        if not np.any(eligible):
            raise ValueError("No Relevant Road Segment Within Station And Elevation Gates")
        distance = np.linalg.norm(point[:2] - nearest[:, :2], axis=1)
        distance[~eligible] = np.inf
        index = int(np.argmin(distance))
        unit = self.delta[index] / self.length[index]
        along = float(np.dot(point[:2] - nearest[index, :2], unit))
        if (index == 0 and fraction[index] == 0 and along < -_EPS) or (
            index == len(self.length) - 1 and fraction[index] == 1 and along > _EPS
        ):
            raise ValueError(
                "Vehicle Is Beyond Surveyed Centerline Endpoints; Lateral Error Undefined"
            )
        normal = np.array([-unit[1], unit[0]])
        center = nearest[index]
        signed = math.copysign(float(distance[index]), float(_cross(unit, point[:2] - center[:2])))
        limits = []
        for boundary in (self.left, self.right):
            edge = boundary[index] + fraction[index] * (boundary[index + 1] - boundary[index])
            limits.append(float(np.dot(edge[:2] - center[:2], normal)))
        if min(limits) >= 0 or max(limits) <= 0:
            raise ValueError("Centerline Is Not Between The Road Boundaries")
        return {
            "centerline_x_m": float(center[0]),
            "centerline_y_m": float(center[1]),
            "centerline_z_m": float(center[2]),
            "segment_index": index,
            "route_station_m": float(station[index]),
            "signed_lateral_offset_m": signed,
            "absolute_lateral_offset_m": abs(signed),
            "right_boundary_offset_m": min(limits),
            "left_boundary_offset_m": max(limits),
        }

    def measure(
        self,
        position: list[float],
        bbox: dict,
        expected_station_m: float,
        association_window_m: float,
        max_height_error_m: float,
    ) -> dict:
        result = self.project(
            position, expected_station_m, association_window_m, max_height_error_m
        )
        corners = np.asarray(list(bbox.values()), dtype=float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("Ground Truth Requires Eight Finite XYZ BBox Corners")
        index = result["segment_index"]
        unit = self.delta[index] / self.length[index]
        normal = np.array([-unit[1], unit[0]])
        footprint = (corners[:, :2] - np.asarray(position)[:2]) @ normal
        result.update(
            vehicle_center_lower_limit_m=result["right_boundary_offset_m"] - float(footprint.min()),
            vehicle_center_upper_limit_m=result["left_boundary_offset_m"] - float(footprint.max()),
            road_departure=int(footprint_departure(corners, self.polygon)),
        )
        return result


def event_count(values: list[bool]) -> int:
    """A continuous outside interval counts once; starting outside counts once."""
    return sum(current and not previous for previous, current in zip([False] + values, values))


def summarize_run(rows: list[dict], metadata: dict) -> dict:
    if len(rows) < 2:
        raise ValueError("At Least Two Measurements Are Required For A Run Summary")
    times = np.array([float(row["timestamp_s"]) for row in rows])
    signed = np.array([float(row["signed_lateral_offset_m"]) for row in rows])
    speeds = np.array([float(row["speed_mph"]) for row in rows])
    if not all(np.isfinite(array).all() for array in (times, signed, speeds)):
        raise ValueError("Nonfinite Measurements Cannot Be Summarized")
    if np.any(np.diff(times) <= 0) or np.any(speeds < 0):
        raise ValueError("Timestamps Must Increase And Speeds Must Be Nonnegative")
    outside = [bool(int(row["road_departure"])) for row in rows]
    manual = [str(row["manual_intervention"]).strip().lower() for row in rows]
    if any(value not in ("0", "1", "unknown") for value in manual):
        raise ValueError("Manual Status Must Be 0, 1, Or Unknown")
    known = "unknown" not in manual
    interventions: int | str = "Unknown"
    if known:
        interventions = event_count([value == "1" for value in manual])
    completed = metadata.get("route_completed") is True
    departures = event_count(outside)
    speed_pass = bool(np.max(speeds) <= 10.0)
    result = {
        "run": metadata["run"],
        "samples": len(rows),
        "duration_s": float(times[-1] - times[0]),
        "mean_absolute_lateral_offset_m": float(np.abs(signed).mean()),
        "std_signed_lateral_offset_m": float(signed.std(ddof=0)),
        "std_absolute_lateral_offset_m": float(np.abs(signed).std(ddof=0)),
        "mean_signed_lateral_offset_m": float(signed.mean()),
        "maximum_absolute_lateral_offset_m": float(np.abs(signed).max()),
        "road_departures": departures,
        "manual_interventions": interventions,
        "route_completed": "No",
        "speed_compliance": "Fail",
        "maximum_speed_mph": float(speeds.max()),
        "maximum_sample_gap_s": float(np.diff(times).max()),
        "successful_run": "No",
        "observed_pr3_criteria": "Fail",
        "end_reason": metadata.get("end_reason", "Unknown"),
        "provisional": metadata.get("provisional", True),
    }
    if completed:
        result["route_completed"] = "Yes"
    if speed_pass:
        result["speed_compliance"] = "Pass"
    if completed and departures == 0 and interventions == 0:
        result["successful_run"] = "Yes"
        if speed_pass:
            result["observed_pr3_criteria"] = "Pass"
    elif completed and departures == 0 and not known:
        result["successful_run"] = "Unverified"
        if speed_pass:
            result["observed_pr3_criteria"] = "Unverified"
    return result


def aggregate_runs(rows_by_run: list[list[dict]], summaries: list[dict]) -> dict:
    """Pooled population spread includes both within-run and between-run variation."""
    signed = np.array(
        [float(row["signed_lateral_offset_m"]) for rows in rows_by_run for row in rows]
    )
    if not len(signed):
        raise ValueError("No Executed Runs To Aggregate")
    return {
        "executed_runs": len(summaries),
        "successful_runs": sum(row["successful_run"] == "Yes" for row in summaries),
        "completed_runs": sum(row["route_completed"] == "Yes" for row in summaries),
        "speed_compliant_runs": sum(row["speed_compliance"] == "Pass" for row in summaries),
        "manual_status_unknown_runs": sum(
            row["manual_interventions"] == "Unknown" for row in summaries
        ),
        "road_departures": sum(row["road_departures"] for row in summaries),
        "samples": len(signed),
        "pooled_mean_absolute_lateral_offset_m": float(np.abs(signed).mean()),
        "pooled_std_signed_lateral_offset_m": float(signed.std(ddof=0)),
        "pooled_std_absolute_lateral_offset_m": float(np.abs(signed).std(ddof=0)),
        "maximum_absolute_lateral_offset_m": float(np.abs(signed).max()),
        "mean_of_run_mean_absolute_lateral_offset_m": float(
            np.mean([row["mean_absolute_lateral_offset_m"] for row in summaries])
        ),
        "std_of_run_mean_absolute_lateral_offset_m": float(
            np.std([row["mean_absolute_lateral_offset_m"] for row in summaries], ddof=0)
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Do Not Create Measurement CSVs Without Measurements")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
