"""Evaluate identical recorded RGB sequences with manual label images.

No annotation is ever passed to a perception or planning stage. ORFD's
0=background, 128=void, 255=road mapping is the default, and is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.runtime.rgb_evaluation import MaskMetrics, manual_path_metrics
from offroad_autonomy.types import CameraFrame, ControlCommand, PathPlan, PipelineStepResult
from offroad_autonomy.utils.config import load_config


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p95": None}
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-suffix", default="_fillcolor")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--working-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--weights")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override CUDA FP16 inference (--no-half uses FP32)",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--morphology", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument(
        "--render-ui",
        action="store_true",
        help="Measure off-screen rendering separately; opens no window",
    )
    parser.add_argument("--road-value", type=int, default=255)
    parser.add_argument("--void-value", type=int, default=128)
    parser.add_argument("--timestamp-scale", type=float, default=0.001)
    parser.add_argument("--sequence-gap-s", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--planning",
        action="store_true",
        help="Only with calibration for this RGB camera in --config",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.weights:
        config.model_weights = args.weights
    if args.threshold is not None:
        if not 0 <= args.threshold <= 1:
            parser.error("--threshold must be in [0, 1]")
        config.confidence_threshold = args.threshold
    config.enable_clahe = args.clahe
    if args.half is not None:
        config.perception_half = args.half
    config.enable_morphology = args.morphology
    config.enable_ema = args.ema
    if args.input_size is not None:
        config.perception_input_size = args.input_size
    if args.working_size is not None:
        if min(args.working_size) <= 0:
            parser.error("--working-size dimensions must be positive")
        config.preprocess_width, config.preprocess_height = args.working_size
    paths = sorted(args.images.glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        parser.error("No PNG RGB frames found")
    preprocessor = ImagePreprocessor(config)
    segmenter = RoadSegmenter(config)
    stabilizer = TemporalStabilizer(config)
    planner = None
    valid_roi = None
    if args.planning:
        view = PerceptionView(config)
        valid_roi = view.valid_roi
        planner = CenterlinePlanner(config, camera=view.camera)
    dashboard = None
    if args.render_ui:
        from offroad_autonomy.visualization.dashboard import AutonomyDashboard

        dashboard = AutonomyDashboard(sensor=config.camera.sensor)
    raw_metrics, stabilized_metrics = MaskMetrics(), MaskMetrics()
    stage_samples = {
        key: []
        for key in (
            "decode",
            "preprocess",
            "segmentation",
            "postprocess",
            "planning",
            "rgb_to_path",
        )
    }
    valid_plans = 0
    render_samples = []
    target_changes = []
    previous_target = None
    manifest = hashlib.sha256()
    previous_time = None
    rows = []
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Unreadable RGB frame: {paths[0]}")
    packet = preprocessor.process(first)
    for _ in range(args.warmup):
        segmenter.predict(packet, valid_roi)
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for index, path in enumerate(paths):
        label_path = args.labels / f"{path.stem}{args.label_suffix}.png"
        encoded, encoded_label = path.read_bytes(), label_path.read_bytes()
        manifest.update(path.name.encode())
        manifest.update(encoded)
        manifest.update(encoded_label)
        label = cv2.imdecode(np.frombuffer(encoded_label, np.uint8), cv2.IMREAD_UNCHANGED)
        if label is None or label.ndim != 2:
            raise ValueError(f"Expected an indexed grayscale manual annotation: {label_path}")
        unexpected = set(np.unique(label).tolist()) - {0, args.void_value, args.road_value}
        if unexpected:
            raise ValueError(f"Unmapped annotation values {unexpected} in {label_path}")
        timestamp = float(path.stem) * args.timestamp_scale
        if previous_time is None or not 0 < timestamp - previous_time <= args.sequence_gap_s:
            stabilizer.reset()
            raw_metrics.reset_sequence()
            stabilized_metrics.reset_sequence()
            previous_target = None
            if planner is not None:
                planner.reset()
        previous_time = timestamp
        start = time.perf_counter()
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable RGB frame: {path}")
        decoded = time.perf_counter()
        packet = preprocessor.process(image)
        preprocessed = time.perf_counter()
        perception = segmenter.predict(packet, valid_roi)
        segmented = time.perf_counter()
        stabilized = stabilizer.stabilize(perception)
        postprocessed = time.perf_counter()
        plan = None
        if planner is not None:
            plan = planner.plan(stabilized)
            if not plan.fallback_active:
                valid_plans += 1
        planned = time.perf_counter()
        stages = (
            decoded - start,
            preprocessed - decoded,
            segmented - preprocessed,
            postprocessed - segmented,
            planned - postprocessed,
            planned - start,
        )
        for key, elapsed in zip(stage_samples, stages):
            stage_samples[key].append(elapsed * 1000)
        # Score on the native label grid so downsampling cannot erase errors
        # in the reference annotations. No camera-specific hood is hidden here.
        truth = label == args.road_value
        valid = label != args.void_value
        for metrics, mask in (
            (raw_metrics, perception.mask),
            (stabilized_metrics, stabilized.mask),
        ):
            pred = cv2.resize(
                mask.astype(np.uint8),
                (label.shape[1], label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            metrics.add(pred, truth, valid)
        row = {"frame": path.name, "timestamp_s": timestamp, "latency_ms": stages[-1] * 1000}
        if plan is not None:
            row["plan_valid"] = not plan.fallback_active
            row["reason"] = plan.fallback_reason
            row.update(
                manual_path_metrics(plan.centerline, truth, valid, (packet.width, packet.height))
            )
            target = None
            if not plan.fallback_active and len(plan.centerline) >= 2:
                forward, right, good = planner._camera.image_to_ground(plan.centerline)
                if good.all() and forward.min() <= 5 <= forward.max():
                    order = np.argsort(forward)
                    target = float(np.interp(5, forward[order], right[order]))
            if target is not None and previous_target is not None:
                target_changes.append(abs(target - previous_target))
            previous_target = target
        if dashboard is not None:
            from offroad_autonomy.visualization.dashboard import DashboardTelemetry

            display_plan = plan
            if display_plan is None:
                display_plan = PathPlan(centerline=np.empty((0, 2), np.float32))
            result = PipelineStepResult(
                packet,
                perception,
                stabilized,
                display_plan,
                ControlCommand(),
                capture=CameraFrame(image),
            )
            telemetry = DashboardTelemetry(
                0, 0, 0, 0, 0, stabilized.stability_score, False, 0, stages[-1] * 1000
            )
            render_start = time.perf_counter()
            dashboard.render(result, telemetry, plan=plan, valid_roi=valid_roi)
            render_samples.append((time.perf_counter() - render_start) * 1000)
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"Evaluated {index + 1}/{len(paths)} frames", flush=True)
    elapsed, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
    gpu_memory = None
    gpu_name = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
        gpu_name = torch.cuda.get_device_name(0)
    report = {
        "config": asdict(config),
        "input_manifest_sha256": manifest.hexdigest(),
        "weights_sha256": hashlib.sha256(Path(config.model_weights).read_bytes()).hexdigest(),
        "rgb_only": True,
        "annotations_used_for_inference": False,
        "raw": raw_metrics.report(),
        "stabilized": stabilized_metrics.report(),
        "stages_ms": {key: _summary(values) for key, values in stage_samples.items()},
        "processing_fps_excluding_io_and_metrics": 1000 / np.mean(stage_samples["rgb_to_path"]),
        "offline_wall_fps_including_io_and_metrics": len(paths) / elapsed,
        "process_cpu_core_equivalents": cpu / elapsed,
        "gpu": gpu_name,
        "gpu_peak_allocated_mib": gpu_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warmup_frames_excluded": args.warmup,
        "planning_enabled": args.planning,
        "valid_plans": valid_plans,
        "unregistered_5m_target_change_m": _summary(target_changes),
        "offscreen_ui_render_ms": _summary(render_samples),
        "frames": rows,
        "notes": "tIoU is unwarped consecutive road-mask IoU; stability is label-relative flip agreement in active road pixels. Offline processing FPS is not live capture FPS or UI FPS. ORFD calibration is not inferred from BeamNG.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "raw",
                    "stabilized",
                    "stages_ms",
                    "processing_fps_excluding_io_and_metrics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
