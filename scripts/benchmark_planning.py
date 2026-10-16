"""Deterministic mask stress test, not a real-world trajectory benchmark.

Analytic corridors use measured-style camera parameters. The planner only
receives a predicted mask and confidence, never the reference centerline.
Run against a saved source tree via PYTHONPATH to compare the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult


def _mask(camera, right, half_width):
    forward = np.linspace(1, 20, len(right))
    left = camera.ground_to_image(forward, right - half_width)
    edge = camera.ground_to_image(forward, right + half_width)[::-1]
    mask = np.zeros((camera.height, camera.width), np.uint8)
    cv2.fillPoly(mask, [np.rint(np.vstack([left, edge])).astype(np.int32)], 1)
    return mask.astype(bool)


def _polyline_score(points, reference):
    if len(points) < 2 or not np.isfinite(points).all():
        return None, None
    samples = []
    for index in range(1, len(points)):
        start, end = points[index - 1], points[index]
        count = max(2, int(np.ceil(np.max(np.abs(end - start)) * 2)) + 1)
        samples.append(np.linspace(start, end, count))
    pixels = np.rint(np.concatenate(samples)).astype(int)
    h, w = reference.shape
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    x, y = np.clip(pixels[:, 0], 0, w - 1), np.clip(pixels[:, 1], 0, h - 1)
    coverage = float((inside & reference[y, x]).mean())
    distances = cv2.distanceTransform(np.pad(reference.astype(np.uint8), 1), cv2.DIST_L2, 5)
    clearance = float(distances[y + 1, x + 1].min())
    return coverage, clearance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--half-width-m", type=float, default=0.0)
    args = parser.parse_args()
    config = PipelineConfig(enable_ema=False, enable_morphology=False, baseline_temporal_blend=0.4)
    config.path_half_width_m = args.half_width_m
    camera = CameraModel(config.camera, 720, 465)
    forward = np.linspace(1, 20, 160)
    timings, rows, jitter = [], [], []
    manifest = hashlib.sha256()
    for scenario in ("straight", "curve", "narrow", "hole", "s_bend", "weak_corridor", "dropout"):
        planner = CenterlinePlanner(config, camera=camera)
        previous_target = None
        for frame in range(30):
            center = np.full_like(forward, 0.03 * np.sin(frame * 0.7))
            half_width = 2.0
            if scenario == "curve":
                center += 0.025 * (forward - 2) ** 2
            if scenario == "narrow":
                half_width = 0.5
            if scenario == "s_bend":
                center += 2.5 * np.sin((forward - 2) * 0.55)
                half_width = 0.8
            mask = _mask(camera, center, half_width)
            if scenario == "hole":
                mask[270:280, 345:375] = False
            if scenario == "dropout" and frame >= 24:
                mask[:] = False
            confidence = np.full(mask.shape, 0.9, np.float32)
            scores = [0.9]
            if scenario == "weak_corridor":
                confidence[:] = 0.1
                mask[235:250, 550:650] = True
                confidence[235:250, 550:650] = 0.99
                scores = [0.99]
            confidence[~mask] = 0
            manifest.update(mask.tobytes())
            manifest.update(confidence.tobytes())
            raw = PerceptionResult(mask=mask, confidences=scores)
            # Also works against the baseline type, which predates the map.
            raw.confidence_map = confidence
            stabilized = StabilizedResult(mask=mask, raw_result=raw)
            start = time.perf_counter()
            plan = planner.plan(stabilized)
            elapsed = (time.perf_counter() - start) * 1000
            timings.append(elapsed)
            valid = not plan.fallback_active
            fraction, clearance = _polyline_score(plan.centerline, mask)
            target = None
            if (
                valid
                and len(plan.centerline) >= 2
                and plan.centerline[0, 1] <= 300 <= plan.centerline[-1, 1]
            ):
                target = float(np.interp(300, plan.centerline[:, 1], plan.centerline[:, 0]))
            if target is not None and previous_target is not None:
                jitter.append(abs(target - previous_target))
            previous_target = target
            rows.append(
                {
                    "scenario": scenario,
                    "frame": frame,
                    "valid": valid,
                    "on_mask_fraction": fraction,
                    "clearance_px": clearance,
                    "reason": plan.fallback_reason,
                }
            )
    valid_rows = [row for row in rows if row["valid"]]
    report = {
        "kind": "synthetic_mask_stress_only",
        "configured_half_width_m": args.half_width_m,
        "input_sha256": manifest.hexdigest(),
        "frames": len(rows),
        "valid_plans": len(valid_rows),
        "valid_paths_leaving_mask": sum(row["on_mask_fraction"] < 1 for row in valid_rows),
        "weak_corridor_valid_plans": sum(
            row["valid"] for row in rows if row["scenario"] == "weak_corridor"
        ),
        "planning_mean_ms": float(np.mean(timings)),
        "planning_p95_ms": float(np.percentile(timings, 95)),
        "mean_target_change_px": float(np.mean(jitter)),
        "minimum_valid_clearance_px": min(row["clearance_px"] for row in valid_rows),
        "scenarios": {
            name: {
                "valid": sum(row["valid"] for row in rows if row["scenario"] == name),
                "frames": 30,
            }
            for name in sorted({row["scenario"] for row in rows})
        },
        "results": rows,
        "notes": "Synthetic projected masks isolate planner behavior. These results do not prove physical clearance, road accuracy, or trajectory stability on recorded RGB. Narrow means a visible centerline, not vehicle footprint clearance.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
