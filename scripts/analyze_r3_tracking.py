#!/usr/bin/env python3
"""Locate R3 Tracking Excursions And Separate Path Error From Vehicle Response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.runtime.r3_evaluation import RoadGeometry, write_csv
from offroad_autonomy.utils.config import load_config


def numeric(path):
    with path.open(encoding="utf-8") as stream:
        names = stream.readline().strip().split(",")
        return np.atleast_1d(np.loadtxt(stream, delimiter=",", dtype=[(x, float) for x in names]))


def road_curvature(geometry, stations):
    middle = 0.5 * (geometry.station[1:] + geometry.station[:-1])
    heading = np.unwrap(np.arctan2(geometry.delta[:, 1], geometry.delta[:, 0]))
    return (
        np.interp(stations + 2.0, middle, heading) - np.interp(stations - 2.0, middle, heading)
    ) / 4.0


def analyze(batch, out, config):
    out.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(
        (ROOT / "results/r3/final_verification/approved_route_geometry.json").read_text()
    )
    geometry = RoadGeometry(snapshot["road"]["edges"])
    cfg = load_config(config)
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    peaks = []
    for file in sorted(batch.glob("run_??.csv")):
        data = numeric(file)
        i = int(np.argmax(data["absolute_lateral_offset_m"]))
        peak = {"run": file.stem}
        for key in (
            "timestamp_s",
            "vehicle_x_m",
            "vehicle_y_m",
            "route_station_m",
            "signed_lateral_offset_m",
            "speed_mph",
            "steering",
            "throttle",
            "brake",
        ):
            peak[key] = float(data[key][i])
        peak["road_curvature_left_per_m"] = float(
            road_curvature(geometry, data["route_station_m"][i])
        )
        peaks.append(peak)
    if peaks:
        write_csv(out / "peak_errors.csv", peaks)
        print(json.dumps(peaks, indent=2), flush=True)
    for file in sorted(batch.glob("run_??.control.jsonl")):
        frames = [json.loads(line) for line in file.read_text().splitlines()]
        metadata = json.loads(file.with_name(file.name.split(".")[0] + ".json").read_text())
        calibration = metadata["recording_start"]["calibration"]
        mount = np.array([-cfg.camera.pos[1], -cfg.camera.pos[0], cfg.camera.pos[2]])
        mount_source = "Nominal Configuration; Actual Mount Was Not Recorded"
        if "camera_world_position" in calibration:
            direction = np.asarray(calibration["forward"])
            direction /= np.linalg.norm(direction)
            right_axis = np.cross(direction, calibration["up"])
            right_axis /= np.linalg.norm(right_axis)
            up_axis = np.cross(right_axis, direction)
            mount = (
                np.asarray(calibration["camera_world_position"]) - calibration["position"]
            ) @ np.array([direction, right_axis, up_axis]).T
            mount_source = "Measured Parked Sensor Position; Pose Alignment Remains Approximate"
        rows = []
        station = 129.40945544141377
        for frame in frames:
            pose = frame["before"]
            debug = frame["command"]["debug"]
            if debug is None:
                continue
            projected = geometry.project(pose["position"], station, 12, 3)
            station = projected["route_station_m"]
            f = np.asarray(frame["path_forward_m"])
            r = np.asarray(frame["path_right_m"])
            if len(f) < 2:
                continue
            order = np.argsort(f)
            f, r = f[order], r[order]
            unit = np.asarray(pose["direction"])
            right = np.cross(unit, pose["up"])
            right /= np.linalg.norm(right)
            up = np.cross(right, unit)
            row = {
                "timestamp_s": pose["timestamp_s"],
                "station_m": station,
                "signed_offset_m": projected["signed_lateral_offset_m"],
                "speed_mph": pose["speed_mph"],
                "steering": frame["command"]["steering"],
                "desired_steering": debug["desired_steering"],
                "confidence": frame["confidence"],
                "pursuit_branch": float(cfg.control_latency_s * frame["speed_mps"] < f.min()),
                "road_curvature_right_per_m": -float(road_curvature(geometry, station + 4)),
                "planned_curvature_right_per_m": debug["curvature"],
                "control_cte_m": debug["cross_track_m"],
                "heading_error_rad": debug["heading_error_rad"],
                "lookahead_m": debug["lookahead_m"],
                "path_start_m": float(f.min()),
                "target_speed_mps": debug["target_speed_mps"],
                "frame_age_ms": frame["timings_ms"].get("frame_age", 0),
                "pipeline_ms": frame["timings_ms"]["perception_planning"],
                "command_interval_sim_s": frame["after"]["timestamp_s"] - pose["timestamp_s"],
                "speed_reason": debug["speed_reason"],
                "plan_status": frame["plan_status"],
            }
            target = debug.get("lookahead_px")
            row["target_forward_m"] = None
            if target is not None:
                tf, _, tk = camera.image_to_ground([target])
                if tk[0]:
                    row["target_forward_m"] = float(tf[0])
            for distance in (4.0, 6.0):
                error = None
                if f.min() <= distance <= f.max():
                    lateral = np.interp(distance, f, r)
                    point = (
                        np.asarray(pose["position"])
                        + unit * (distance + mount[0])
                        + right * (lateral + mount[1])
                        + up * (mount[2] - cfg.camera.pos[2])
                    )
                    error = geometry.project(point, station + distance, 15, 6)[
                        "signed_lateral_offset_m"
                    ]
                row[f"approx_path_error_{int(distance)}m"] = error
            rows.append(row)
        if not rows:
            continue
        write_csv(out / f"{file.stem}.csv", rows)
        summary = {
            "frames": len(rows),
            "total_logged_frames": len(frames),
            "scope": "Geometric Diagnostics Use Frames With A Usable Path; Gate Counts Include Every Logged Frame",
            "camera_mount_forward_right_up_m": mount.tolist(),
            "approximate_path_transform": mount_source,
            "pursuit_branch_fraction": np.mean([r["pursuit_branch"] for r in rows]),
            "plan_status_counts": {
                key: sum(r["plan_status"] == key for r in frames)
                for key in {r["plan_status"] for r in frames}
            },
        }
        for key in (
            "confidence",
            "frame_age_ms",
            "pipeline_ms",
            "command_interval_sim_s",
            "path_start_m",
        ):
            values = np.array([r[key] for r in rows])
            summary[key] = {
                "mean": float(values.mean()),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
            }
        for key in ("approx_path_error_4m", "approx_path_error_6m"):
            values = np.array([r[key] for r in rows if r[key] is not None])
            summary[key] = {
                "mean_signed_m": float(values.mean()),
                "mean_absolute_m": float(np.abs(values).mean()),
            }
        absolute_error = np.abs([r["signed_offset_m"] for r in rows])
        summary["descriptive_correlations_with_absolute_vehicle_error"] = {}
        for key in (
            "speed_mph",
            "confidence",
            "steering",
            "road_curvature_right_per_m",
            "planned_curvature_right_per_m",
        ):
            values = np.array([r[key] for r in rows])
            if key not in ("speed_mph", "confidence"):
                values = np.abs(values)
            summary["descriptive_correlations_with_absolute_vehicle_error"][key] = float(
                np.corrcoef(absolute_error, values)[0, 1]
            )
        summary["correlation_scope"] = (
            "Autocorrelated Control Frames; Association Does Not Establish Causation"
        )
        times = np.array([r["timestamp_s"] for r in rows])
        summary["control_period_sim_s"] = {
            "median": float(np.median(np.diff(times))),
            "p95": float(np.percentile(np.diff(times), 95)),
        }
        summary["fraction_command_near_lock"] = float(
            np.mean(np.abs([r["steering"] for r in rows]) >= 0.95)
        )
        command_delta = np.abs(np.diff([r["steering"] for r in rows]))
        summary["steering_command_total_variation_per_s"] = float(
            command_delta.sum() / (times[-1] - times[0])
        )
        summary["steering_command_rate_p95_per_s"] = float(
            np.percentile(command_delta / np.diff(times), 95)
        )
        worst = max(rows, key=lambda r: abs(r["signed_offset_m"]))
        summary["largest_offset_frame"] = worst
        (out / f"{file.stem}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, layout="constrained")
        t = [r["timestamp_s"] for r in rows]
        for key, label in (
            ("signed_offset_m", "Vehicle Offset"),
            ("approx_path_error_4m", "Projected Path At 4 m"),
        ):
            axes[0].plot(t, [r[key] for r in rows], label=label)
        axes[0].set_ylabel("Offset (m)")
        for key, label in (
            ("road_curvature_right_per_m", "Authored Road"),
            ("planned_curvature_right_per_m", "Perceived Path"),
        ):
            axes[1].plot(t, [r[key] for r in rows], label=label)
        axes[1].set_ylabel("Curvature (1/m)")
        for key, label in (
            ("steering", "Applied Command"),
            ("desired_steering", "Desired Command"),
        ):
            axes[2].plot(t, [r[key] for r in rows], label=label)
        axes[2].set_ylabel("Steering [-1,1]")
        axes[3].plot(t, [r["speed_mph"] for r in rows], label="Actual Speed")
        axes[3].plot(t, [r["target_speed_mps"] / 0.44704 for r in rows], label="Target Speed")
        axes[3].set_ylabel("Speed (mph)")
        axes[4].plot(t, [r["confidence"] for r in rows], label="Segmentation Confidence")
        axes[4].set_ylabel("Confidence")
        axes[4].set_xlabel("Simulation Time (s)")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(loc="upper right")
        fig.suptitle("R3 Development Diagnostics; Path Errors Use Approximate Pose Alignment")
        fig.savefig(out / f"{file.stem}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, default=ROOT / "results/r3/final_verification/batch_01"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/r3/tracking_improvement/baseline_analysis"
    )
    parser.add_argument("--config", default=str(ROOT / "configs/r3_approved_bends_driving.yaml"))
    args = parser.parse_args()
    analyze(args.batch, args.out, args.config)
