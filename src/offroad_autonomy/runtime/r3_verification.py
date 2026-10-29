"""Physics-Rate R3 Measurements Against The Frozen Authored Road Ribbon."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry, event_count

_SIGNS = np.array(list(itertools.product((0, 1), repeat=3)))


def envelope_corners(position, forward, up, bounds) -> np.ndarray:
    position = np.atleast_2d(position).astype(float)
    forward = np.atleast_2d(forward).astype(float)
    up = np.atleast_2d(up).astype(float)
    forward /= np.linalg.norm(forward, axis=1)[:, None]
    right = np.cross(forward, up)
    right /= np.linalg.norm(right, axis=1)[:, None]
    up = np.cross(right, forward)
    frame = np.stack((forward, right, up), axis=1)
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 3, 2)
    local = np.take_along_axis(bounds[:, None, :, :], _SIGNS[None, :, :, None], axis=3)[..., 0]
    return position[:, None, :] + np.einsum("nki,nij->nkj", local, frame)


def analyze_physics(
    raw_path: Path, metadata: dict, route: dict, geometry: RoadGeometry
) -> tuple[dict, np.ndarray]:
    import shapely

    data = np.genfromtxt(raw_path, delimiter=",", names=True, dtype=float)
    data = np.atleast_1d(data)
    if len(data) < 2:
        raise ValueError("No Complete Physics Recording")
    finite = all(np.isfinite(data[name]).all() for name in data.dtype.names)
    seq = data["sequence"]
    steps = data["dt_s"]
    timestamps = data["timestamp_s"]
    sampling_ok = bool(
        finite
        and np.array_equal(seq, np.arange(1, len(seq) + 1))
        and np.all(steps > 0)
        and np.all(np.diff(timestamps) > 0)
        and np.allclose(np.diff(timestamps), steps[1:], atol=2e-9, rtol=0)
    )
    # Native counters can change between the final physics sample and stop.
    # Keep authority evidence from the entire recording, including after B.
    manual = int(data["manual_interventions"].max())
    guard_faults = int(data["guard_faults"].max())
    native = metadata.get("native_end", {})
    final_manual = native.get("interventions")
    counters_ok = type(final_manual) is int and final_manual >= manual
    if counters_ok:
        manual = final_manual
    cutoff = metadata.get("finish_timestamp_s", float("inf"))
    data = data[data["timestamp_s"] <= cutoff + 1e-9]
    names = (
        "timestamp_s",
        "vehicle_x_m",
        "vehicle_y_m",
        "vehicle_z_m",
        "centerline_x_m",
        "centerline_y_m",
        "centerline_z_m",
        "signed_lateral_offset_m",
        "absolute_lateral_offset_m",
        "speed_mph",
        "road_departure",
        "manual_interventions",
        "steering",
        "throttle",
        "brake",
        "route_station_m",
        "right_boundary_offset_m",
        "left_boundary_offset_m",
        "vehicle_center_lower_limit_m",
        "vehicle_center_upper_limit_m",
    )
    output = np.full((len(data), len(names)), np.nan)
    road_polygon = shapely.Polygon(geometry.polygon)
    if not shapely.is_valid(road_polygon):
        raise ValueError("Authored Road Polygon Is Not Valid")
    shapely.prepare(road_polygon)
    all_valid = True
    previous_station = route["start_station_m"]
    for begin in range(0, len(data), 2048):
        block = data[begin : begin + 2048]
        p = np.column_stack([block[key] for key in ("x", "y", "z")])
        f = np.column_stack([block[key] for key in ("fx", "fy", "fz")])
        up = np.column_stack([block[key] for key in ("ux", "uy", "uz")])
        bounds = np.column_stack(
            [block[key] for key in ("low_f", "high_f", "low_r", "high_r", "low_u", "high_u")]
        )
        corners = envelope_corners(p, f, up, bounds)
        footprint = shapely.convex_hull(shapely.multipoints(corners[:, :, :2]))
        outside = ~shapely.covers(road_polygon, footprint)
        fractions = np.clip(
            np.einsum("nsi,si->ns", p[:, None, :2] - geometry.center[None, :-1, :2], geometry.delta)
            / geometry.length**2,
            0,
            1,
        )
        closest = geometry.center[None, :-1] + fractions[:, :, None] * np.diff(
            geometry.center, axis=0
        )
        distance2 = np.sum((p[:, None, :2] - closest[:, :, :2]) ** 2, axis=2)
        distance2[np.abs(p[:, None, 2] - closest[:, :, 2]) > route["max_height_error_m"]] = np.inf
        index = np.argmin(distance2, axis=1)
        row = np.arange(len(block))
        q = closest[row, index]
        unit = geometry.delta[index] / geometry.length[index, None]
        normal = np.column_stack((-unit[:, 1], unit[:, 0]))
        delta = p[:, :2] - q[:, :2]
        signed = np.copysign(np.linalg.norm(delta, axis=1), np.sum(delta * normal, axis=1))
        station = geometry.station[index] + fractions[row, index] * geometry.length[index]
        along = np.sum(delta * unit, axis=1)
        valid = np.isfinite(distance2[row, index])
        valid &= ~((index == 0) & (fractions[row, index] == 0) & (along < -1e-8))
        valid &= ~(
            (index == len(geometry.length) - 1) & (fractions[row, index] == 1) & (along > 1e-8)
        )
        valid &= np.abs(np.diff(np.r_[previous_station, station])) <= route["association_window_m"]
        all_valid &= bool(valid.all())
        previous_station = float(station[-1])
        signed[~valid] = np.nan
        limits = []
        for boundary in (geometry.left, geometry.right):
            edge = boundary[index] + fractions[row, index, None] * (
                boundary[index + 1] - boundary[index]
            )
            limits.append(np.sum((edge[:, :2] - q[:, :2]) * normal, axis=1))
        lower = np.minimum(*limits)
        upper = np.maximum(*limits)
        relative = np.einsum("nki,ni->nk", corners[:, :, :2] - p[:, None, :2], normal)
        values = (
            block["timestamp_s"],
            p[:, 0],
            p[:, 1],
            p[:, 2],
            q[:, 0],
            q[:, 1],
            q[:, 2],
            signed,
            np.abs(signed),
            np.sqrt(block["vx"] ** 2 + block["vy"] ** 2 + block["vz"] ** 2) / 0.44704,
            outside.astype(int),
            block["manual_interventions"],
            block["steering"],
            block["throttle"],
            block["brake"],
            station,
            lower,
            upper,
            lower - relative.min(axis=1),
            upper - relative.max(axis=1),
        )
        output[begin : begin + len(block)] = np.column_stack(values)
    np.savetxt(
        raw_path.with_name(raw_path.name.replace(".physics.csv", ".csv")),
        output,
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.9f",
    )
    e = output[:, 7]
    departures = event_count(output[:, 10].astype(bool).tolist())
    health = (
        sampling_ok
        and all_valid
        and counters_ok
        and guard_faults == 0
        and native.get("sequence") == int(seq[-1])
        and native.get("guard_faults") == 0
        and native.get("buffer_overflow") is False
    )
    summary = {
        "run": metadata["run"],
        "samples": len(data),
        "duration_s": float(data["timestamp_s"][-1]),
        "mean_absolute_lateral_offset_m": float(np.mean(np.abs(e))),
        "std_signed_lateral_offset_m": float(np.std(e, ddof=0)),
        "std_absolute_lateral_offset_m": float(np.std(np.abs(e), ddof=0)),
        "maximum_absolute_lateral_offset_m": float(np.max(np.abs(e))),
        "road_departures": departures,
        "manual_interventions": manual,
        "maximum_speed_mph": float(output[:, 9].max()),
        "route_completed": metadata.get("route_completed") is True,
        "physics_recording_integrity": health,
        "ground_truth_association_valid": all_valid,
        "maximum_physics_interval_s": float(steps.max()),
        "centering_targets_status": "Proposed; Mentor Approval Pending",
        "end_reason": metadata["end_reason"],
    }
    summary["proposed_mean_target"] = bool(
        health
        and summary["mean_absolute_lateral_offset_m"] <= route["mean_absolute_offset_target_m"]
    )
    summary["proposed_signed_sd_target"] = bool(
        health and summary["std_signed_lateral_offset_m"] <= route["std_signed_offset_target_m"]
    )
    summary["speed_compliance"] = bool(
        health and summary["maximum_speed_mph"] <= route["max_speed_mph"]
    )
    summary["zero_manual_inputs_verified"] = bool(health and manual == 0)
    summary["zero_authored_corridor_departures"] = bool(health and departures == 0)
    summary["physical_trail_boundaries_verified"] = route["physical_boundaries_approved"]
    summary["all_proposed_test_conditions"] = bool(
        summary["route_completed"]
        and summary["proposed_mean_target"]
        and summary["proposed_signed_sd_target"]
        and summary["speed_compliance"]
        and summary["zero_manual_inputs_verified"]
        and summary["zero_authored_corridor_departures"]
    )
    summary["official_centering_compliance"] = "Not Verified: Targets Await Mentor Approval"
    # JSON null keeps missing ground truth distinguishable from numerical zeros.
    for key, value in list(summary.items()):
        if isinstance(value, float) and not np.isfinite(value):
            summary[key] = None
    raw_path.with_name(raw_path.name.replace(".physics.csv", ".summary.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, output
