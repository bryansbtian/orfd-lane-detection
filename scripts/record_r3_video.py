#!/usr/bin/env python3
"""Record Fresh R3 Demonstrations With The Existing Protected Test Runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import r3_verify
from r3_verify import ROOT

from offroad_autonomy.simulation.presentation_capture_client import PresentationCaptureClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use A New Presentation Directory; Existing Recordings Are Preserved")
    args.out.mkdir(parents=True)
    files = [
        Path(__file__),
        ROOT / "src/offroad_autonomy/simulation/presentation_capture_client.py",
    ]
    manifest = {
        "purpose": "Presentation Demonstrations; Not Additional Verification Trials",
        "recording_sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "baseline_evidence": "results/r3/final_verification/batch_01",
        "improved_evidence": "results/r3/tracking_improvement/verification_01",
        "camera_overhead": "An Additional Camera And Encoder Can Affect Runtime; Do Not Pool These Runs With Verification",
    }
    (args.out / "presentation.json").write_text(json.dumps(manifest, indent=2))
    for label, config in (
        ("baseline", "configs/r3_approved_bends_driving.yaml"),
        ("improved", "configs/r3_tracking_visible_origin.yaml"),
    ):
        directory = args.out / label

        # This process-local client factory adds capture only. The frozen
        # controller, pipeline, test runner, geometry and guard stay untouched.
        def client_factory(cfg, target=directory):
            return PresentationCaptureClient(cfg, target)

        r3_verify.R3VerificationClient = client_factory
        run_args = SimpleNamespace(
            route=ROOT / "configs/r3_approved_bends.yaml",
            config=str(ROOT / config),
            qualification=ROOT
            / "results/r3/final_verification/qualification/attempt_03/qualification.json",
            out=directory,
            runs=1,
            phase="development",
            diagnostics=False,
            snapshots=False,
        )
        r3_verify.run_batch(run_args)
        capture = json.loads((directory / "capture.json").read_text())
        summary = json.loads((directory / "run_01.summary.json").read_text())
        if capture["encoder_failed"] or capture["capture_error"]:
            raise RuntimeError(f"Recording Failed: {capture}")
        if not summary["route_completed"]:
            raise RuntimeError(f"Demonstration Did Not Complete; Attempt Preserved: {summary}")
        print(f"Recorded {label}: {directory / 'cameras.mp4'}", flush=True)


if __name__ == "__main__":
    main()
