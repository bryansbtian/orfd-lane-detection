#!/usr/bin/env python3
"""Survey, Record, And Analyse Preliminary BeamNG R3 Experiments."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.r3_evaluation import (
    RoadGeometry,
    aggregate_runs,
    summarize_run,
    write_csv,
)
from offroad_autonomy.simulation.r3_client import R3BeamNGClient
from offroad_autonomy.utils.config import load_config


def select_geometry(survey_path: Path, route_path: Path) -> Path:
    route = yaml.safe_load(route_path.read_text(encoding="utf-8"))
    data = json.loads(survey_path.read_text(encoding="utf-8"))
    if data["map"] != route["map"]:
        raise ValueError("Survey And Route Maps Differ")
    matches = [
        (road_id, road)
        for road_id, road in data["roads"].items()
        if road.get("persistentId") == route["road_persistent_id"]
    ]
    if len(matches) != 1:
        raise ValueError("Selected Persistent Road ID Must Match Exactly One Surveyed Road")
    road_id, road = matches[0]
    RoadGeometry(road["edges"])
    snapshot = {
        "map": data["map"],
        "source": "BeamNG Road Survey; Evaluation Only",
        "source_survey_utc": data["recorded_utc"],
        "road_id_at_survey": road_id,
        "road": road,
    }
    output = (route_path.parent / route["geometry_file"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Existing snapshots bind recorded runs to their original road geometry.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2, allow_nan=False)
    return output


def survey(config: str, output: Path) -> None:
    cfg = load_config(config)
    client = R3BeamNGClient(cfg)
    try:
        client.connect()
        data = client.road_survey()
        data.update(
            recorded_utc=datetime.now(timezone.utc).isoformat(),
            map=cfg.beamng_map,
            vehicle=cfg.beamng_vehicle,
            spawn=client._resolve_spawn(cfg),
            controller=cfg.control_mode,
            planner=cfg.planner_mode,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Survey Written: {output}")
    finally:
        client.disconnect()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def load_route(path: Path) -> tuple[dict, RoadGeometry]:
    route = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = (
        "name",
        "map",
        "vehicle",
        "spawn_position",
        "spawn_rotation",
        "geometry_file",
        "road_persistent_id",
        "point_a_xyz_m",
        "point_b_xyz_m",
        "start_station_m",
        "finish_station_m",
        "finish_radius_m",
        "start_radius_m",
        "association_window_m",
        "max_height_error_m",
        "max_run_sim_s",
        "max_run_wall_s",
        "seed",
        "weather_preset",
        "time_of_day",
        "boundary_status",
        "provisional",
        "restart_delay_s",
    )
    if not isinstance(route, dict) or any(route.get(key) is None for key in required):
        raise ValueError("Fill In The Route Template Using Surveyed Coordinates Before Running")
    geometry_path = (path.parent / route["geometry_file"]).resolve()
    snapshot = json.loads(geometry_path.read_text(encoding="utf-8"))
    if snapshot["road"]["persistentId"] != route["road_persistent_id"]:
        raise ValueError("Route And Geometry Persistent IDs Differ")
    edges = snapshot["road"]["edges"]
    if route.get("reverse", False):
        edges = list(reversed(edges))
    geometry = RoadGeometry(edges)
    if not 0 <= route["start_station_m"] < route["finish_station_m"] <= geometry.station[-1]:
        raise ValueError("Start And Finish Stations Must Increase Within The Surveyed Road")
    for key in (
        "finish_radius_m",
        "start_radius_m",
        "association_window_m",
        "max_height_error_m",
        "max_run_sim_s",
        "max_run_wall_s",
        "restart_delay_s",
    ):
        if not np.isfinite(route[key]) or route[key] <= 0:
            raise ValueError(f"Route {key} Must Be Positive And Finite")
    for key, count in (
        ("spawn_position", 3),
        ("spawn_rotation", 4),
        ("point_a_xyz_m", 3),
        ("point_b_xyz_m", 3),
    ):
        values = np.asarray(route[key], dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"Route {key} Has Invalid Coordinates")
    if not 0 <= route["time_of_day"] <= 1:
        raise ValueError("Time Of Day Must Be In [0, 1]")
    route["geometry_sha256"] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    return route, geometry


def _measurement(sample: dict, geometry: RoadGeometry, station: float, route: dict) -> dict:
    result = geometry.measure(
        sample["position"],
        sample["bbox"],
        station,
        route["association_window_m"],
        route["max_height_error_m"],
    )
    row = {
        "timestamp_s": sample["timestamp_s"],
        "graphics_sequence": sample["sequence"],
        "sample_interval_s": sample["dt_s"],
        "vehicle_x_m": sample["position"][0],
        "vehicle_y_m": sample["position"][1],
        "vehicle_z_m": sample["position"][2],
    }
    row.update(result)
    row.update(
        speed_mph=sample["speed_mph"],
        steering=sample["steering"],
        throttle=sample["throttle"],
        brake=sample["brake"],
        manual_intervention=sample["manual_intervention"],
        bbox_world_xyz_json=json.dumps(sample["bbox"], separators=(",", ":")),
    )
    return row


def record_runs(args) -> None:
    # Torch and the model are only needed by live driving, not analysis or survey.
    import torch

    from offroad_autonomy.main import _StuckDetector, resolve_weights
    from offroad_autonomy.pipeline import AutonomyPipeline

    route, geometry = load_route(args.route)
    cfg = load_config(args.config)
    if cfg.beamng_map != route["map"] or cfg.beamng_vehicle != route["vehicle"]:
        raise ValueError("Driving Config Map And Vehicle Must Match The Route")
    spawn = cfg.map_spawns[cfg.beamng_map]["spawns"][cfg.beamng_spawn_index]
    if not np.allclose(spawn["pos"], route["spawn_position"], atol=1e-6, rtol=0) or not np.allclose(
        spawn["rot"], route["spawn_rotation"], atol=1e-6, rtol=0
    ):
        raise ValueError("Driving Config Spawn Must Match The Route")
    weights = resolve_weights(cfg.model_weights)
    cfg.model_weights = str(weights)
    versions = {"python": platform.python_version()}
    for package in ("beamngpy", "numpy", "matplotlib", "torch", "ultralytics", "opencv-python"):
        versions[package] = importlib.metadata.version(package)
    provenance = {
        "versions": versions,
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "resolved_config": asdict(cfg),
        "route": route,
        "sampling": "Every Positive-dt GE onUpdate Graphics Step; Not Physics Substeps",
        "simulator_seed": "Not Controlled; No Global Seed API Used",
        "manual_detection": "Unknown For Native Simulator Inputs; Runner Has No Manual Mode",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    # Never replace existing evidence, including a partial run after a crash.
    numbers = range(args.start_run, args.start_run + args.runs)
    for number in numbers:
        if list(args.out.glob(f"run_{number:02d}.*")):
            raise FileExistsError(f"Run {number:02d} Already Exists; Choose A New Output Directory")
    _json(args.out / "experiment.json", provenance)
    for number in numbers:
        if number > args.start_run:
            # The launcher can retain its single-instance lock briefly after the
            # engine exits. Starting immediately can fail before a scenario loads.
            time.sleep(route["restart_delay_s"])
        label = f"run_{number:02d}"
        seed = int(route["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        client = R3BeamNGClient(cfg)
        metadata = {
            "run": label,
            "provisional": route["provisional"],
            "route_completed": False,
            "seed": seed,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "end_reason": "Startup Failed",
            "controls_issued": 0,
            "recorder_overflow": False,
            "route": route,
        }
        samples: list[dict] = []
        started = False
        recording = False
        failure = False
        print(f"Starting {label}", flush=True)
        try:
            if args.diagnose_startup:
                faulthandler.dump_traceback_later(60, repeat=True)
            client.connect()
            pipe = AutonomyPipeline(cfg)
            metadata["environment"] = client.configure_environment(
                route["weather_preset"], route["time_of_day"]
            )
            actual_road = client.selected_road(route["road_persistent_id"])
            expected = json.loads(
                (args.route.parent / route["geometry_file"]).read_text(encoding="utf-8")
            )["road"]
            if actual_road["edges"] != expected["edges"]:
                raise ValueError("Live Road Edges Differ From The Frozen Geometry Snapshot")
            metadata["session"] = client.session_info()
            metadata["recorder_start"] = client.start_recording()
            faulthandler.cancel_dump_traceback_later()
            recording = True
            client.release_park()
            started = True
            metadata["end_reason"] = "Time Limit"
            detector = _StuckDetector(cfg.safety_min_road_fraction, cfg.safety_no_road_time_s)
            begin = last_frame = time.perf_counter()
            station = route["start_station_m"]
            first_position = None
            with (args.out / f"{label}.raw.jsonl").open("x", encoding="utf-8") as raw:
                while time.perf_counter() - begin < route["max_run_wall_s"]:
                    packet = client.drain_recording()
                    if packet["overflow"] or not packet["active"]:
                        metadata["recorder_overflow"] = bool(packet["overflow"])
                        raise RuntimeError("Native Recorder Overflow Or Vehicle Lost")
                    batch = packet["samples"]
                    if batch:
                        for sample in batch:
                            if sample["sequence"] != len(samples) + 1:
                                raise RuntimeError("Native Recorder Sequence Gap")
                            raw.write(json.dumps(sample, allow_nan=False) + "\n")
                            samples.append(sample)
                        raw.flush()
                        for sample in batch:
                            measured = geometry.project(
                                sample["position"],
                                station,
                                route["association_window_m"],
                                route["max_height_error_m"],
                            )
                            station = measured["route_station_m"]
                            if first_position is None:
                                first_position = np.array(sample["position"])
                                segment = measured["segment_index"]
                                forward = np.array(sample["direction"][:2])
                                alignment = float(
                                    np.dot(forward, geometry.delta[segment])
                                    / (np.linalg.norm(forward) * geometry.length[segment])
                                )
                                metadata["initial_forward_route_cosine"] = alignment
                                if alignment <= 0:
                                    raise RuntimeError("Spawn Faces Away From The A To B Route")
                                if (
                                    np.linalg.norm(first_position[:2] - route["point_a_xyz_m"][:2])
                                    > route["start_radius_m"]
                                ):
                                    raise RuntimeError(
                                        "Initial Position Is Outside The Point A Gate"
                                    )
                            distance = np.linalg.norm(
                                np.array(sample["position"][:2]) - route["point_b_xyz_m"][:2]
                            )
                            if (
                                station >= route["finish_station_m"]
                                and distance <= route["finish_radius_m"]
                            ):
                                metadata["route_completed"] = True
                                metadata["finish_timestamp_s"] = sample["timestamp_s"]
                                metadata["end_reason"] = "Point B Reached"
                                break
                        if (
                            metadata["route_completed"]
                            or samples[-1]["timestamp_s"] >= route["max_run_sim_s"]
                        ):
                            break
                    capture = client.capture_frame()
                    if capture is None or not pipe.has_input(capture) or not capture.is_new:
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
                    result = pipe.step_result(capture, state)
                    stop, reason = detector.update(
                        time.perf_counter(),
                        result.stabilized.road_fraction,
                        state.speed_mps,
                        result.command.throttle,
                    )
                    if stop:
                        metadata["end_reason"] = f"Automatic Safe Stop: {reason}"
                        break
                    client.send_controls(result.command)
                    metadata["controls_issued"] += 1
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001
            # Preserve partial evidence for simulator/driver exceptions with
            # undocumented types, and for an operator abort.
            metadata["end_reason"] = f"Aborted: {type(exc).__name__}: {exc}"
            metadata["error_traceback"] = traceback.format_exc()
            failure = True
            print(metadata["end_reason"], flush=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
            try:
                client.park()
                if recording:
                    # Retain observations up to the stop, including a failing control
                    # iteration. Exclude post-finish samples from the scored interval.
                    packet = client.drain_recording()
                    with (args.out / f"{label}.raw.jsonl").open("a", encoding="utf-8") as raw:
                        for sample in packet["samples"]:
                            raw.write(json.dumps(sample, allow_nan=False) + "\n")
                            samples.append(sample)
                    metadata["recorder_overflow"] |= bool(packet["overflow"])
                    client.stop_recording()
            finally:
                client.disconnect()
                metadata["simulation_started"] = started
                metadata["raw_samples"] = len(samples)
                metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
                _json(args.out / f"{label}.json", metadata)
        if samples:
            rows = []
            station = route["start_station_m"]
            for sample in samples:
                if sample["timestamp_s"] > metadata.get("finish_timestamp_s", float("inf")):
                    break
                try:
                    row = _measurement(sample, geometry, station, route)
                except ValueError as exc:
                    metadata["measurement_error"] = str(exc)
                    metadata["route_completed"] = False
                    _json(args.out / f"{label}.json", metadata)
                    break
                station = row["route_station_m"]
                rows.append(row)
            if rows:
                write_csv(args.out / f"{label}.csv", rows)
            if len(rows) >= 2:
                print(json.dumps(summarize_run(rows, metadata)), flush=True)
        if failure and not samples:
            break
    analyze(args.out)


def analyze(directory: Path) -> None:
    summaries = []
    all_rows = []
    for path in sorted(directory.glob("run_[0-9][0-9].csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        summaries.append(summarize_run(rows, metadata))
        all_rows.append(rows)
    if not summaries:
        print("No Executed Runs With Measurements; No Summary Or Figures Created")
        return
    write_csv(directory / "summary.csv", summaries)
    _json(directory / "aggregate.json", aggregate_runs(all_rows, summaries))
    from offroad_autonomy.runtime.r3_plots import plot_results

    plot_results(directory, all_rows, summaries)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("survey", help="Export Actual Map Road Edges Without Driving")
    scan.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    scan.add_argument("--out", type=Path, default=ROOT / "results/r3/road_survey.json")
    selection = commands.add_parser("select-road", help="Freeze Route Geometry From A Saved Survey")
    selection.add_argument("--survey", type=Path, required=True)
    selection.add_argument("--route", type=Path, required=True)
    run = commands.add_parser("run", help="Run The Unmodified Autonomy Pipeline")
    run.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    run.add_argument("--route", type=Path, required=True)
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--start-run", type=int, default=1)
    run.add_argument("--out", type=Path, default=ROOT / "results/r3")
    run.add_argument("--diagnose-startup", action="store_true")
    analysis = commands.add_parser("analyze", help="Regenerate Summary And Figures From Run CSVs")
    analysis.add_argument("--out", type=Path, default=ROOT / "results/r3")
    args = parser.parse_args()
    if args.command == "survey":
        survey(args.config, args.out)
    elif args.command == "select-road":
        output = select_geometry(args.survey, args.route)
        print(f"Route Geometry Written: {output}")
    elif args.command == "run":
        if args.runs < 1 or args.start_run < 1:
            parser.error("--runs And --start-run Must Be Positive")
        record_runs(args)
    elif args.command == "analyze":
        analyze(args.out)


if __name__ == "__main__":
    main()
