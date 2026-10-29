#!/usr/bin/env python3
"""Qualify And Execute The Approved R3 Dirt-Road Bends Test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.r3_evaluation import RoadGeometry, write_csv
from offroad_autonomy.runtime.r3_verification import analyze_physics, envelope_corners
from offroad_autonomy.simulation.r3_verification_client import R3VerificationClient
from offroad_autonomy.utils.config import load_config

logger = logging.getLogger("offroad_autonomy.r3_verify")


def probe(args) -> None:
    client = R3VerificationClient(load_config(args.config))
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        client.connect()
        environment = client.configure_environment("sunny_noon", 0.0)
        data = client.probe_physics_geometry()
        data["environment"] = environment
        data["recorded_utc"] = datetime.now(timezone.utc).isoformat()
        (args.out / "physics_geometry_probe.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        print(json.dumps(data["checks"], indent=2), flush=True)
    finally:
        client.disconnect()


def qualify(args) -> None:
    import cv2

    route = yaml.safe_load(args.route.read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    client = R3VerificationClient(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "tests": [],
        "scope": "Native Driving Input APIs And Physics Envelope",
    }
    active = False
    try:
        client.connect()
        client.configure_environment(route["weather_preset"], route["time_of_day"])
        live = client.selected_road(route["road_persistent_id"])
        frozen = json.loads((args.route.parent / route["geometry_file"]).read_text())["road"]
        if live["edges"] != frozen["edges"]:
            raise ValueError("Live Road Geometry Differs From The Frozen Approved Road")
        result["live_geometry_matches_snapshot"] = True
        frame = client.capture_frame()
        if frame is not None:
            cv2.imwrite(str(args.out / "point_a_scene.png"), frame.image)
        result["recording_start"] = client.start_physics_recording(
            args.out / "deliberate_interventions.physics.csv", route["envelope_margin_m"]
        )
        active = True
        time.sleep(1)
        before = client.native_status()
        if before["sequence"] < cfg.r3_startup_min_samples or before["guard_faults"] != 0:
            raise RuntimeError(f"Native Physics Or Guard Startup Failed: {before}")
        cases = [
            ("Keyboard Steering", "input.kbdSteer(false, 1, FILTER_KBD)"),
            ("Gamepad Steering", "input.event('steering', 0.75, FILTER_PAD)"),
            ("Gamepad Pedals", "input.padAccelerateBrake(0.8, FILTER_PAD)"),
            ("Direct Brake Input", "input.event('brake', 1, FILTER_DIRECT)"),
            ("Clutch Input", "input.event('clutch', 1, FILTER_KBD)"),
            ("Parking Brake Toggle", "input.toggleEvent('parkingbrake')"),
            ("Gear Selection", "drivetrain.shiftToGear(-1)"),
        ]
        for name, chunk in cases:
            previous = client.native_status()
            after = client.inject_validation_input(chunk)
            passed = after["interventions"] > previous["interventions"]
            result["tests"].append(
                {
                    "name": name,
                    "injection": chunk,
                    "detected": passed,
                    "interventions_after": after["interventions"],
                }
            )
            if not passed:
                raise RuntimeError(f"Deliberate Intervention Not Detected: {name}")
        previous = client.native_status()
        client.inject_external_control()
        time.sleep(0.1)
        after = client.native_status()
        result["tests"].append(
            {
                "name": "External BeamNGpy Control Request",
                "detected": after["interventions"] > previous["interventions"],
            }
        )
        if not result["tests"][-1]["detected"]:
            raise RuntimeError("External BeamNGpy Inputs Not Detected")
        if (
            after["guard_faults"] != 0
            or after["latest"]["speed_mph"] > cfg.r3_stationary_speed_limit_mph
        ):
            raise RuntimeError("Rejected Inputs Changed Guard Authority Or Moved Parked Vehicle")
        result["all_injected_inputs_blocked"] = True
        result["intervention_test_final"] = client.stop_physics_recording()
        active = False
        result["clean_recording_start"] = client.start_physics_recording(
            args.out / "clean_physics_interval.physics.csv", route["envelope_margin_m"]
        )
        active = True
        time.sleep(0.5)
        client.native_command({"steering": 0.25, "throttle": 0.12, "brake": 0, "parkingbrake": 0})
        time.sleep(2)
        client.park()
        sample = client.probe_physics_geometry()
        initial = result["clean_recording_start"]
        forward = np.asarray(sample["forward"])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, sample["up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        box = np.asarray(list(sample["ge_bbox"].values()))
        local = (box - sample["position"]) @ np.array([forward, right, up]).T
        bounds = np.asarray(initial["body_bounds"]).reshape(3, 2)
        excess = max(
            float(np.max(bounds[:, 0] - local.min(axis=0))),
            float(np.max(local.max(axis=0) - bounds[:, 1])),
            0,
        )
        result["moving_pose_calibration_excess_m"] = excess
        result["moving_pose_calibration_within_margin"] = excess <= route["envelope_margin_m"]
        result["clean_recording_final"] = client.stop_physics_recording()
        active = False
        clean = result["clean_recording_final"]
        result["passed"] = (
            result["moving_pose_calibration_within_margin"]
            and clean["interventions"] == 0
            and clean["guard_faults"] == 0
            and clean["sequence"] >= cfg.r3_clean_min_samples
        )
        if not result["passed"]:
            raise RuntimeError("Clean Physics Recording Or Moving Envelope Qualification Failed")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Qualification Failed")
    finally:
        if active:
            client.park()
            result["aborted_recording_status"] = client.stop_physics_recording()
        client.disconnect()
        (args.out / "qualification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "error": result.get("error")}), flush=True)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")


def _frozen_hashes(args) -> dict:
    files = [
        args.route,
        Path(args.config),
        ROOT / "configs/default.yaml",
        ROOT / "src/offroad_autonomy/pipeline.py",
        ROOT / "models/weights.pt",
        ROOT / "src/offroad_autonomy/simulation/beamng_client.py",
        ROOT / "src/offroad_autonomy/simulation/odometry.py",
        ROOT / "src/offroad_autonomy/simulation/r3_verification_client.py",
        ROOT / "src/offroad_autonomy/simulation/r3_vehicle_verifier.lua",
        Path(__file__),
        ROOT / "src/offroad_autonomy/runtime/r3_diagnostics.py",
    ]
    for directory in ("control", "planning", "perception", "postprocessing", "preprocessing"):
        files.extend((ROOT / "src/offroad_autonomy" / directory).glob("*.py"))
    files.extend((ROOT / "src/offroad_autonomy/runtime").glob("r3_*.py"))
    files.extend(
        [ROOT / "src/offroad_autonomy/types.py", ROOT / "src/offroad_autonomy/utils/config.py"]
    )
    parent = Path(args.config).resolve()
    while True:
        files.append(parent)
        data = yaml.safe_load(parent.read_text(encoding="utf-8"))
        if not data.get("extends"):
            break
        parent = (parent.parent / data["extends"]).resolve()
    return {
        str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


def run_batch(args) -> None:
    import torch

    from offroad_autonomy.main import _StuckDetector
    from offroad_autonomy.pipeline import AutonomyPipeline
    from offroad_autonomy.types import VehicleState

    route = yaml.safe_load(args.route.read_text(encoding="utf-8"))
    snapshot = json.loads((args.route.parent / route["geometry_file"]).read_text(encoding="utf-8"))
    geometry = RoadGeometry(snapshot["road"]["edges"])
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    lua_path = ROOT / "src/offroad_autonomy/simulation/r3_vehicle_verifier.lua"
    if not qualification.get("passed"):
        raise ValueError("A Passed Deliberate-Intervention And Physics Qualification Is Required")
    if (
        qualification["clean_recording_start"]["source_sha256"]
        != hashlib.sha256(lua_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Observer Changed Since Qualification; Requalify Before Scored Runs")
    if args.out.exists() and any(args.out.glob("run_*")):
        raise FileExistsError(
            "Existing Verification Attempts Must Be Preserved; Use A New Batch Directory"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    frozen = _frozen_hashes(args)
    from dataclasses import asdict

    manifest = {
        "route": route,
        "source_hashes": frozen,
        "geometry_sha256": hashlib.sha256(
            (args.route.parent / route["geometry_file"]).read_bytes()
        ).hexdigest(),
        "qualification_file": str(args.qualification),
        "qualification_sha256": hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
        "resolved_driving_config": asdict(cfg),
        "software_seeds": route["seed"],
        "simulator_rng_seed": "Not Controlled",
        "centering_limits": "Proposed; Mentor Approval Pending",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("beamngpy", "numpy", "torch", "ultralytics", "matplotlib", "shapely")
        },
        "planned_attempts": args.runs,
        "scoring": "All Native Physics Steps; Population SD",
        "experiment_phase": args.phase,
        "control_diagnostics": args.diagnostics,
    }
    _write_json(args.out / "experiment.json", manifest)
    summaries = []
    for number in range(1, args.runs + 1):
        if _frozen_hashes(args) != frozen:
            raise RuntimeError("Frozen Driving Or Observer Files Changed; Batch Cannot Continue")
        if number > 1:
            time.sleep(route["restart_delay_s"])
        random.seed(route["seed"])
        np.random.seed(route["seed"])
        torch.manual_seed(route["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(route["seed"])
        label = f"run_{number:02d}"
        client = R3VerificationClient(cfg)
        active = False
        diagnostics = None
        metadata = {
            "run": label,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "route_completed": False,
            "end_reason": "Startup Failed",
            "controls_issued": 0,
            "route": route,
            "native_end": {},
            "source_hashes": frozen,
        }
        raw_path = args.out / f"{label}.physics.csv"
        print(f"Starting Frozen Verification Attempt {number}/{args.runs}", flush=True)
        try:
            client.connect()
            metadata["environment"] = client.configure_environment(
                route["weather_preset"], route["time_of_day"]
            )
            live = client.selected_road(route["road_persistent_id"])
            if live["edges"] != snapshot["road"]["edges"]:
                raise ValueError("Live Road Geometry Does Not Match The Frozen Snapshot")
            metadata["session"] = client.session_info()
            metadata["beamng_version"] = metadata["session"]["beamng_version"]
            client.park()
            pipeline = AutonomyPipeline(cfg)
            for _ in range(route["warmup_frames"]):
                capture = client.capture_frame()
                if capture is None:
                    raise RuntimeError("No Camera Frame For Parked Inference Warmup")
                pipeline.step_result(capture, VehicleState())
            pipeline.reset()
            if args.diagnostics:
                from offroad_autonomy.runtime.r3_diagnostics import TrackingDiagnostics

                diagnostics = TrackingDiagnostics(
                    args.out / f"{label}.control.jsonl",
                    pipeline.view.camera,
                    cfg.r3_snapshot_interval_s,
                    args.snapshots,
                )
            client.reset_odometry()
            metadata["recording_start"] = client.start_physics_recording(
                raw_path, route["envelope_margin_m"]
            )
            active = True
            client.native_command({"steering": 0, "throttle": 0, "brake": 0, "parkingbrake": 0})
            detector = _StuckDetector(cfg.safety_min_road_fraction, cfg.safety_no_road_time_s)
            started = last_frame = time.perf_counter()
            station = route["start_station_m"]
            metadata["end_reason"] = "Run Time Limit"
            last_progress_log = started
            while time.perf_counter() - started < route["max_run_wall_s"]:
                status = client.native_status()
                latest = status.get("latest")
                if latest is None:
                    time.sleep(0.005)
                    continue
                if "initial_position" not in metadata:
                    metadata["initial_position"] = latest["position"]
                    segment = geometry.project(
                        latest["position"],
                        station,
                        route["association_window_m"],
                        route["max_height_error_m"],
                    )["segment_index"]
                    forward = np.asarray(latest["direction"][:2])
                    cosine = np.dot(forward, geometry.delta[segment]) / (
                        np.linalg.norm(forward) * geometry.length[segment]
                    )
                    metadata["initial_forward_route_cosine"] = float(cosine)
                    if (
                        cosine < 0.95
                        or np.linalg.norm(
                            np.array(latest["position"][:2]) - route["point_a_xyz_m"][:2]
                        )
                        > route["start_radius_m"]
                    ):
                        raise RuntimeError(
                            "Initial Position Or Heading Does Not Match The Frozen A Gate"
                        )
                if status["interventions"] or status["guard_faults"] or status["buffer_overflow"]:
                    metadata["end_reason"] = (
                        "Input Intervention Or Authority/Telemetry Integrity Failure"
                    )
                    break
                corners = envelope_corners(
                    latest["position"], latest["direction"], latest["up"], latest["bounds"]
                )[0]
                measured = geometry.measure(
                    latest["position"],
                    {str(i): p for i, p in enumerate(corners)},
                    station,
                    route["association_window_m"],
                    route["max_height_error_m"],
                )
                station = measured["route_station_m"]
                if measured["road_departure"]:
                    metadata["end_reason"] = (
                        "Automatic Stop: Vehicle Envelope Crossed Authored Road Boundary"
                    )
                    break
                if latest["speed_mph"] > route["max_speed_mph"]:
                    metadata["end_reason"] = "Automatic Stop: Speed Exceeded 10 mph"
                    break
                if (
                    station >= route["finish_station_m"]
                    and np.linalg.norm(
                        np.array(latest["position"][:2]) - route["point_b_xyz_m"][:2]
                    )
                    <= route["finish_radius_m"]
                ):
                    metadata["route_completed"] = True
                    metadata["finish_timestamp_s"] = latest["timestamp_s"]
                    metadata["end_reason"] = "Point B Reached"
                    break
                if latest["timestamp_s"] >= route["max_run_sim_s"]:
                    break
                capture = client.capture_frame()
                if capture is None or not capture.is_new:
                    if time.perf_counter() - last_frame > cfg.safety_no_road_time_s:
                        metadata["end_reason"] = "Camera Watchdog Safe Stop"
                        break
                    time.sleep(0.005)
                    continue
                last_frame = time.perf_counter()
                state = client.get_vehicle_state()
                if not state.valid:
                    metadata["end_reason"] = "Invalid Wheel/Gyro State Safe Stop"
                    break
                result = pipeline.step_result(capture, state)
                stop, reason = detector.update(
                    time.perf_counter(),
                    result.stabilized.road_fraction,
                    state.speed_mps,
                    result.command.throttle,
                )
                if stop:
                    metadata["end_reason"] = f"Existing Automatic Safe Stop: {reason}"
                    break
                client.send_controls(result.command)
                if diagnostics is not None:
                    after = client.native_status()["latest"]
                    diagnostics.record(result, state, latest, after)
                metadata["controls_issued"] += 1
                if time.perf_counter() - last_progress_log >= 15:
                    print(
                        f"{label}: {latest['timestamp_s']:.1f} s, progress {station - route['start_station_m']:.1f}/{route['route_length_m']:.1f} m, {latest['speed_mph']:.2f} mph",
                        flush=True,
                    )
                    last_progress_log = time.perf_counter()
        except Exception as exc:
            logger.exception("Verification Attempt Failed")
            metadata["end_reason"] = f"Aborted: {type(exc).__name__}: {exc}"
        finally:
            if diagnostics is not None:
                diagnostics.close()
            try:
                client.park()
                if active:
                    metadata["native_end"] = client.stop_physics_recording()
            finally:
                client.disconnect()
                metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
                _write_json(args.out / f"{label}.json", metadata)
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            summary, _ = analyze_physics(raw_path, metadata, route, geometry)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
            write_csv(args.out / "summary.csv", summaries)
        else:
            _write_json(
                args.out / f"{label}.unmeasured.json",
                {
                    "run": label,
                    "status": "No Physics Measurements",
                    "reason": metadata["end_reason"],
                },
            )
    analyze_batch(args)


def analyze_batch(args) -> None:
    from offroad_autonomy.runtime.r3_verification_plots import make_figures

    route = yaml.safe_load(args.route.read_text(encoding="utf-8"))
    summaries = [
        json.loads(path.read_text()) for path in sorted(args.out.glob("run_*.summary.json"))
    ]
    if not summaries:
        print("No Measured Verification Attempts; No Numerical Results Or Figures Created")
        return
    write_csv(args.out / "summary.csv", summaries)
    make_figures(args.out, route, summaries, args.route.parent / route["geometry_file"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["probe", "qualify", "run", "analyze"])
    parser.add_argument("--config", default=str(ROOT / "configs/r3_approved_bends_driving.yaml"))
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/r3/final_verification/qualification"
    )
    parser.add_argument("--route", type=Path, default=ROOT / "configs/r3_approved_bends.yaml")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--phase", choices=["development", "verification"], default="verification")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument(
        "--qualification",
        type=Path,
        default=ROOT / "results/r3/final_verification/qualification/attempt_03/qualification.json",
    )
    args = parser.parse_args()
    if args.command == "probe":
        probe(args)
    elif args.command == "qualify":
        qualify(args)
    elif args.command == "run":
        run_batch(args)
    elif args.command == "analyze":
        analyze_batch(args)


if __name__ == "__main__":
    main()
