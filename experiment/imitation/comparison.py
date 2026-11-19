"""Matched, procedural BeamNG trials; report unavailable infrastructure instead of invented scores."""

from __future__ import annotations

import copy
import json
import math
import socket
from pathlib import Path

import numpy as np
import yaml

from offroad_autonomy.types import PipelineConfig


def route_geometry(case: dict):
    """The reference road is independent of whichever planner/controller is evaluated."""
    radius = float(case["radius_m"])
    straight = float(case["straight_m"])
    spacing = float(case["node_spacing_m"])
    points = []
    x, y, yaw = 0.0, 0.0, 0.0
    for length, curvature in (
        (straight, 0),
        (math.pi * radius / 2, 1 / radius),
        (math.pi * radius / 2, -1 / radius),
        (straight, 0),
    ):
        for _ in range(max(1, int(length / spacing))):
            points.append((x, y))
            yaw += curvature * spacing
            x += spacing * math.cos(yaw)
            y += spacing * math.sin(yaw)
    return np.asarray(points)


def summarize_trial(rows: list[dict], case: dict, config: PipelineConfig) -> dict:
    if not rows:
        return {"status": "No Frames", "completed": False, "safe_stop_compliance": None}
    errors = np.array([row["center_error_m"] for row in rows])
    offroad = errors + config.vehicle_half_width_m > case["width_m"] / 2
    departures = int(np.count_nonzero(offroad & ~np.r_[False, offroad[:-1]]))
    steering = np.array([row["steering"] for row in rows])
    oscillation = 0.0
    if len(steering) > 1:
        oscillation = float(abs(np.diff(steering)).mean())
    return {
        "status": "Measured",
        "completed": bool(rows[-1]["remaining_m"] <= case["completion_margin_m"]),
        "trail_departures": departures,
        "offroad_frames": int(offroad.sum()),
        "mean_center_error_m": float(errors.mean()),
        "max_center_error_m": float(errors.max()),
        "mean_steering_change": oscillation,
        "mean_control_ms": float(np.mean([row["control_ms"] for row in rows])),
        "safe_stop_compliance": all(
            not row["stop_required"]
            or row["throttle"] == 0
            and (row["brake"] > 0 or row["parkingbrake"] > 0)
            for row in rows
        ),
        "frames": len(rows),
        "elapsed_simulation_s": rows[-1]["simulation_time_s"] - rows[0]["simulation_time_s"],
    }


def run_trial(config, case, controller, output):
    from experiment.imitation.evaluation_client import EvaluationClient
    from experiment.pipeline import ExperimentPipeline
    from offroad_autonomy.main import _StuckDetector

    cfg = copy.deepcopy(config)
    cfg.control_mode = controller
    cfg.beamng_map = case["map"]
    cfg.ui_headless = True
    origin = np.asarray(case["origin_m"], dtype=float)
    centerline = route_geometry(case)
    world = centerline + origin[:2]
    yaw = math.pi / 2 + float(case["start_yaw_rad"])
    spawn = origin.copy()
    spawn[1] += case["start_lateral_m"]
    cfg.map_spawns[case["map"]] = [
        {"pos": spawn.tolist(), "rot": [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]}
    ]
    cfg.beamng_spawn_index = 0
    route = {
        "material": case["material"],
        "width_m": case["width_m"],
        "nodes": [(float(x), float(y), float(origin[2]), case["width_m"]) for x, y in world],
    }
    client = EvaluationClient(cfg, route, cfg.imitation.comparison_physics_hz)
    rows = []
    detector = _StuckDetector(cfg.safety_min_road_fraction, cfg.safety_no_road_time_s)
    try:
        client.connect()
        client.begin_evaluation(cfg.imitation.comparison_physics_hz)
        simulation_time = 0.0

        def clock():
            return simulation_time

        pipeline = ExperimentPipeline(cfg, clock=clock)
        count = int(
            cfg.imitation.evaluation_seconds
            * cfg.imitation.comparison_physics_hz
            / cfg.imitation.comparison_steps
        )
        for _ in range(count):
            client.step_evaluation(cfg.imitation.comparison_steps)
            truth = client.evaluation_state()
            simulation_time = float(truth["time"])
            capture = client.capture_frame()
            state = client.get_vehicle_state()
            if capture is None or not capture.is_new or not state.valid:
                client.park()
                continue
            result = pipeline.step_result(capture, state)
            command = result.command
            stopped, _ = detector.update(
                truth["time"], result.stabilized.road_fraction, state.speed_mps, command.throttle
            )
            stop_required = stopped or result.plan.speed_scale <= 0
            if stopped:
                command.throttle = 0
                command.parkingbrake = 1
            client.send_controls(command)
            distances = np.linalg.norm(world - np.asarray(truth["pos"][:2]), axis=1)
            nearest = int(np.argmin(distances))
            remaining = float(np.linalg.norm(np.diff(world[nearest:], axis=0), axis=1).sum())
            rows.append(
                {
                    "simulation_time_s": truth["time"],
                    "center_error_m": float(distances[nearest]),
                    "remaining_m": remaining,
                    "steering": command.steering,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "parkingbrake": command.parkingbrake,
                    "control_ms": result.timings_ms["control"],
                    "stop_required": stop_required,
                }
            )
            if stopped or remaining <= case["completion_margin_m"]:
                break
    finally:
        client.disconnect()
    output.write_text(json.dumps(rows, indent=2) + "\n")
    return summarize_trial(rows, case, cfg)


def compare_suite(config: PipelineConfig, suite_path: Path) -> dict:
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    output = Path(config.imitation.run_dir) / "closed_loop"
    output.mkdir(parents=True, exist_ok=True)
    report = {"simulation_only": True, "source": str(suite_path), "trials": [], "status": "Not Run"}
    try:
        with socket.create_connection(
            (config.beamng_host, config.beamng_port), timeout=config.imitation.connection_timeout_s
        ):
            pass
    except OSError as exc:
        report.update(
            status="Unavailable", reason=f"BeamNG Endpoint Unavailable: {type(exc).__name__}"
        )
    else:
        for case in suite["cases"]:
            for controller in ("stanley", "imitation", "rl"):
                trial = {"case": case["name"], "controller": controller}
                try:
                    trial.update(
                        run_trial(
                            config, case, controller, output / f"{case['name']}_{controller}.json"
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    trial.update(status="Unavailable", reason=str(exc))
                report["trials"].append(trial)
        report["status"] = "Attempted"
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
