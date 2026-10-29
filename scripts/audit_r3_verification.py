#!/usr/bin/env python3
"""Audit Native R3 Evidence And Optionally Create Lossless Raw CSV Archives."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry, footprint_departure
from offroad_autonomy.runtime.r3_verification import envelope_corners


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def lossless_archive(path: Path) -> dict:
    archive = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(archive, "wb", compresslevel=1) as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
    checked = hashlib.sha256()
    with gzip.open(archive, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checked.update(block)
    assert checked.hexdigest() == digest(path)
    return {"file": archive.name, "sha256": digest(archive)}


def numeric_csv(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        names = stream.readline().strip().split(",")
        return np.atleast_1d(
            np.loadtxt(stream, delimiter=",", dtype=[(name, float) for name in names])
        )


def audit(batch: Path, compress: bool, historical: bool = False) -> None:
    experiment = json.loads((batch / "experiment.json").read_text())
    route = experiment["route"]
    snapshot = ROOT / "results/r3/final_verification/approved_route_geometry.json"
    assert digest(snapshot) == experiment["geometry_sha256"]
    geometry = RoadGeometry(json.loads(snapshot.read_text())["road"]["edges"])
    changed_sources = [
        name
        for name, expected in experiment["source_hashes"].items()
        if digest(ROOT / name.replace("\\", "/")) != expected
    ]
    if not historical:
        assert not changed_sources, f"Frozen Sources Changed: {changed_sources}"
    archived_match = False
    if historical and changed_sources:
        for name in changed_sources:
            original = batch / "source_snapshot" / name.replace("\\", "/")
            assert original.exists(), f"Archive Historical Sources Before Auditing: {name}"
            assert digest(original) == experiment["source_hashes"][name]
        archived_match = True
    results = []
    for number in range(1, experiment["planned_attempts"] + 1):
        label = f"run_{number:02d}"
        metadata = json.loads((batch / f"{label}.json").read_text())
        summary = json.loads((batch / f"{label}.summary.json").read_text())
        assert metadata["source_hashes"] == experiment["source_hashes"]
        assert metadata["route"] == route
        raw_path = batch / f"{label}.physics.csv"
        raw = numeric_csv(raw_path)
        measured_path = batch / f"{label}.csv"
        measured = numeric_csv(measured_path)
        assert all(np.isfinite(raw[name]).all() for name in raw.dtype.names)
        assert all(np.isfinite(measured[name]).all() for name in measured.dtype.names)
        assert np.array_equal(raw["sequence"], np.arange(1, len(raw) + 1))
        assert len(raw) == metadata["native_end"]["sequence"]
        assert np.all(raw["dt_s"] > 0)
        assert np.allclose(np.diff(raw["timestamp_s"]), raw["dt_s"][1:], atol=2e-9, rtol=0)
        assert np.all(np.diff(raw["manual_interventions"]) >= 0)
        assert np.all(np.diff(raw["guard_faults"]) >= 0)
        assert int(raw["manual_interventions"][-1]) == metadata["native_end"]["interventions"]
        assert int(raw["guard_faults"][-1]) == metadata["native_end"]["guard_faults"]
        assert not metadata["native_end"]["buffer_overflow"]
        assert len(metadata["native_end"]["events"]) == metadata["native_end"]["interventions"]
        n = len(measured)
        assert np.allclose(measured["timestamp_s"], raw["timestamp_s"][:n], atol=1e-9, rtol=0)
        for source, destination in (
            ("x", "vehicle_x_m"),
            ("y", "vehicle_y_m"),
            ("z", "vehicle_z_m"),
            ("steering", "steering"),
            ("throttle", "throttle"),
            ("brake", "brake"),
        ):
            assert np.allclose(raw[source][:n], measured[destination], atol=1e-8, rtol=0)
        speed = np.sqrt(raw["vx"] ** 2 + raw["vy"] ** 2 + raw["vz"] ** 2) / 0.44704
        assert np.allclose(speed[:n], measured["speed_mph"], atol=1e-7, rtol=0)
        assert np.allclose(
            np.abs(measured["signed_lateral_offset_m"]),
            measured["absolute_lateral_offset_m"],
            atol=1e-9,
            rtol=0,
        )
        position = np.column_stack([raw[key] for key in ("x", "y", "z")])
        distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
        displacement_bound = 0.5 * (speed[1:] + speed[:-1]) * 0.44704 * raw["dt_s"][1:]
        # This is a diagnostic for resets/teleports, not an acceptance allowance.
        jumps = int(np.count_nonzero(distance > displacement_bound + 0.02))
        assert jumps == 0, "Unexplained Pose Discontinuity; Intervention Evidence Requires Review"
        assert metadata["recording_start"]["native"]["nodes"] == 854
        bounds = np.column_stack(
            [raw[key] for key in ("low_f", "high_f", "low_r", "high_r", "low_u", "high_u")]
        )
        calibrated = np.array(metadata["recording_start"]["body_bounds"])
        margin = metadata["recording_start"]["envelope_margin_m"]
        assert np.all(bounds[:, ::2] <= calibrated[::2] - margin + 1e-8)
        assert np.all(bounds[:, 1::2] >= calibrated[1::2] + margin - 1e-8)
        f = np.column_stack([raw[key] for key in ("fx", "fy", "fz")])
        up = np.column_stack([raw[key] for key in ("ux", "uy", "uz")])
        assert np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-7)
        assert np.allclose(np.linalg.norm(up, axis=1), 1, atol=1e-7)
        assert np.all(np.abs(np.sum(f * up, axis=1)) < 1e-7)
        indices = np.unique(
            np.r_[
                np.linspace(0, n - 1, 40, dtype=int),
                np.argmax(measured["absolute_lateral_offset_m"]),
            ]
        )
        for index in indices:
            projection = geometry.project(
                position[index],
                float(measured["route_station_m"][index]),
                route["association_window_m"],
                route["max_height_error_m"],
            )
            assert (
                abs(
                    projection["signed_lateral_offset_m"]
                    - measured["signed_lateral_offset_m"][index]
                )
                < 1e-6
            )
            corners = envelope_corners(position[index], f[index], up[index], bounds[index])[0]
            assert footprint_departure(corners, geometry.polygon) == bool(
                measured["road_departure"][index]
            )
        final_distance = float(np.linalg.norm(position[n - 1, :2] - route["point_b_xyz_m"][:2]))
        if summary["route_completed"]:
            assert final_distance <= route["finish_radius_m"]
            assert measured["route_station_m"][-1] >= route["finish_station_m"]
        raw_hash = digest(raw_path)
        item = {
            "run": label,
            "raw_samples": len(raw),
            "scored_samples": n,
            "sequence_gaps": 0,
            "pose_discontinuities_over_diagnostic_bound": jumps,
            "maximum_damage": float(raw["damage"].max()),
            "finish_distance_m": final_distance,
            "independent_scalar_geometry_checks": len(indices),
            "raw_sha256": raw_hash,
            "measurements_sha256": digest(measured_path),
            "manual_events": int(raw["manual_interventions"][-1]),
            "guard_faults": int(raw["guard_faults"][-1]),
            "integrity": "Pass",
        }
        if compress:
            native_archive = lossless_archive(raw_path)
            derived_archive = lossless_archive(measured_path)
            item["lossless_raw_archive"] = native_archive["file"]
            item["archive_sha256"] = native_archive["sha256"]
            item["lossless_measurements_archive"] = derived_archive["file"]
            item["measurements_archive_sha256"] = derived_archive["sha256"]
        results.append(item)
        print(json.dumps(item), flush=True)
    result = {
        "all_five_integrity_checks_passed": len(results) == 5,
        "all_recorded_integrity_checks_passed": True,
        "geometry_sha256": digest(snapshot),
        "frozen_sources_match": not changed_sources,
        "historical_source_changes": changed_sources,
        "historical_audit": historical,
        "changed_sources_match_exact_archives": archived_match,
        "runs": results,
        "diagnostic_pose_margin_m": 0.02,
        "note": "Diagnostic Margin Is Not An R3 Acceptance Threshold",
    }
    (batch / "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, default=ROOT / "results/r3/final_verification/batch_01"
    )
    parser.add_argument("--compress-raw", action="store_true")
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Audit Preserved Development Data Without Claiming Current Sources Match Old Hashes",
    )
    args = parser.parse_args()
    audit(args.batch, args.compress_raw, args.historical)


if __name__ == "__main__":
    main()
