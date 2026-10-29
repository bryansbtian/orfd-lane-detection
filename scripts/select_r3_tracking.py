#!/usr/bin/env python3
"""Freeze The R3 Development Selection Before Independent Verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from r3_verify import ROOT, _frozen_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/r3_tracking_visible_origin.yaml")
    parser.add_argument("--route", type=Path, default=ROOT / "configs/r3_approved_bends.yaml")
    parser.add_argument(
        "--development",
        type=Path,
        default=ROOT / "results/r3/tracking_improvement/development/08_visible_origin",
    )
    args = parser.parse_args()
    directory = ROOT / "results/r3/tracking_improvement"
    baseline = json.loads(
        (ROOT / "results/r3/final_verification/batch_01/aggregate.json").read_text()
    )
    selected = json.loads((args.development / "run_01.summary.json").read_text())
    assert selected["route_completed"] and selected["physics_recording_integrity"]
    assert selected["road_departures"] == selected["manual_interventions"] == 0
    assert selected["speed_compliance"]
    assert (
        selected["mean_absolute_lateral_offset_m"]
        < baseline["pooled_mean_absolute_lateral_offset_m"]
    )
    assert selected["std_signed_lateral_offset_m"] < baseline["pooled_std_signed_lateral_offset_m"]
    evidence = []
    for path in sorted((directory / "development").glob("*/run_??.summary.json")):
        evidence.append({"attempt": path.parent.name, **json.loads(path.read_text())})
    synthetic = json.loads((directory / "adaptive_generalization.json").read_text())["scenarios"]
    checks = [row for row in synthetic if row["config"] == args.config]
    assert len(checks) == 12 and all(row["completed"] and row["edge_frames"] == 0 for row in checks)
    frozen = _frozen_hashes(args)
    record = {
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "selected_config": args.config,
        "selected_development": str(args.development.relative_to(ROOT)),
        "behavior_change": "Adaptive preview begins at max(latency-predicted position, first visible path point), then uses the existing curvature-adaptive lookahead and clamps to the observed path end. Original Stanley gains, 0.30 m soft deadzone, 0.45 steering EMA, and speed settings are retained. Experimental edge-filter/weighting modes are disabled.",
        "rationale": [
            "Lowest Mean Error Among The Safe Completed Development Candidates",
            "Improved Signed SD And Maximum Error Relative To Original Baseline",
            "Retains Curvature Adaptation Instead Of Imposing A Fixed 6 m Preview",
            "Lower Command Variation And No Near-Lock Commands In The Selection Trial",
            "All 12 Independent Synthetic Scenarios Completed With Zero Center-Departure Frames; Footprint Safety Is Not Certified By Those Synthetic Cases",
            "Tighter Deadzone Worsened Signed SD; Extra EMA And Curve-Speed Tuning Added Little Benefit",
            "Strict Visible-Edge And Weighted-Fit Candidates Rejected; All Failures Retained",
        ],
        "centering_limits": "Proposed; Mentor Approval Pending; Neither Limit Met In Selection Trial",
        "source_hashes": frozen,
        "development_results": evidence,
        "synthetic_evidence": "results/r3/tracking_improvement/adaptive_generalization.json",
        "synthetic_selected_results": checks,
        "planned_verification_attempts": 5,
        "verification_directory": "results/r3/tracking_improvement/verification_01",
        "verification_diagnostics": False,
        "verification_protocol": "Native Physics Recording Only, Matching Original Baseline; No Further Tuning Or Selective Discarding",
        "native_reference_point": "Unchanged BeamNG Vehicle Reference Position; No Datum Shift Or Lateral Bias",
    }
    target = directory / "selection.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    archive = directory / "selected_source"
    for name, expected in frozen.items():
        source = ROOT / name.replace("\\", "/")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        if source.suffix == ".pt":
            continue
        copy = archive / source.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copy)
    print(f"Frozen Selection Written: {target}; {len(evidence)} Development Attempts Retained")


if __name__ == "__main__":
    main()
