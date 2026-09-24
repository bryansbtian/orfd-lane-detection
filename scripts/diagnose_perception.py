#!/usr/bin/env python3
"""Stage-by-stage perception diagnostics.

Two sub-commands, so the (slow) simulator is only needed once::

    # 1. Launch BeamNG, park at each spawn, save the raw perception frames.
    python scripts/diagnose_perception.py capture --frames 4

    # 2. Replay the saved frames through every stage and dump what each saw.
    python scripts/diagnose_perception.py analyze

``analyze`` writes, per frame, into ``output/diagnostics/<frame>/``:

    0_raw.png          the BGR frame exactly as BeamNG delivered it
    1_preprocessed.png what the model is actually fed (resize + CLAHE)
    2_raw_mask.png     every instance the model returned at a near-zero
                       threshold, labelled with its score
    3_binary_mask.png  the mask after the configured confidence threshold
    4_stabilized.png   after EMA + morphology (what the planner receives)
    5_trajectory.png   the planned path over the frame, fallback in red

and prints a table of the numbers that decide which stage is failing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.utils.config import load_config

logger = logging.getLogger("diagnose")


def capture(args: argparse.Namespace) -> None:
    from offroad_autonomy.simulation.beamng_client import BeamNGClient

    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spawns = config.map_spawns.get(config.beamng_map, {}).get("spawns", [None])
    indices = [args.spawn]
    if args.spawn < 0:
        indices = list(range(len(spawns)))

    client = BeamNGClient(replace(config, beamng_spawn_index=indices[0]))
    try:
        client.connect()
        for index in indices:
            spawn = spawns[index]
            if spawn is not None and index != indices[0]:
                client._vehicle.teleport(tuple(spawn["pos"]), tuple(spawn["rot"]), reset=True)
                time.sleep(3.0)
            client.park()
            time.sleep(1.0)
            cfg = config
            for shot in range(args.frames):
                pair = client.capture_pair()
                if pair is None or pair.left is None:
                    logger.warning("spawn %d shot %d: no frame", index, shot)
                    continue
                stem = f"{cfg.beamng_map}_s{index}_f{shot}"
                cv2.imwrite(str(out / f"{stem}_left.png"), pair.left)
                if pair.right is not None:
                    cv2.imwrite(str(out / f"{stem}_right.png"), pair.right)
                display = client.capture_display()
                if display is not None:
                    cv2.imwrite(str(out / f"{stem}_display.png"), display)
                logger.info("saved %s", stem)
                time.sleep(0.5)
    finally:
        client.disconnect()


def _tint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    if mask.any():
        out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    for i, line in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
    return out


def _raw_instances(segmenter, frame, conf: float):
    """Every instance the model returns at ``conf``, bypassing the pipeline."""
    results = segmenter._model.predict(
        frame.preprocessed,
        conf=conf,
        imgsz=segmenter._imgsz,
        verbose=False,
        retina_masks=True,
    )
    if not results or results[0].masks is None:
        return [], [], []
    result = results[0]
    masks = [m.cpu().numpy().astype(bool) for m in result.masks.data]
    confs = [float(c) for c in result.boxes.conf.cpu()]
    names = [result.names[int(c)] for c in result.boxes.cls.cpu()]
    return masks, confs, names


def analyze(args: argparse.Namespace) -> None:
    from offroad_autonomy.perception.perception_view import PerceptionView
    from offroad_autonomy.perception.road_segmenter import RoadSegmenter
    from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
    from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
    from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
    from offroad_autonomy.types import VehicleState

    config = load_config(args.config)
    if args.conf is not None:
        config = replace(config, confidence_threshold=args.conf)
    if args.no_clahe:
        config = replace(config, enable_clahe=False)
    if args.weights:
        config = replace(config, model_weights=args.weights)

    src = Path(args.src)
    frames = sorted(src.glob(f"*_{args.side}.png"))
    if not frames:
        sys.exit(f"No *_{args.side}.png frames in {src} - run `capture` first")

    view = PerceptionView(config)
    preprocessor = ImagePreprocessor(config, target_size=view.size)
    segmenter = RoadSegmenter(config)
    out_root = Path(args.out)
    rows = []

    for path in frames:
        raw = cv2.imread(str(path))
        stem = path.stem
        out = out_root / stem
        out.mkdir(parents=True, exist_ok=True)

        # Fresh temporal state per frame: each is judged on its own.
        stabilizer = TemporalStabilizer(config)
        planner = CenterlinePlanner(config)

        frame = preprocessor.process(raw)
        masks, confs, names = _raw_instances(segmenter, frame, conf=0.01)
        perception = segmenter.predict(frame, view.valid_roi)
        stabilized = stabilizer.stabilize(perception)
        plan = planner.plan(stabilized, vehicle_state=VehicleState())

        img = frame.preprocessed
        h, w = img.shape[:2]

        cv2.imwrite(str(out / "0_raw.png"), raw)
        cv2.imwrite(str(out / "1_preprocessed.png"), img)

        raw_vis = img.copy()
        palette = [(0, 200, 255), (255, 120, 0), (0, 255, 120), (200, 0, 255), (255, 255, 0)]
        order = np.argsort(confs)[::-1]
        for rank, i in enumerate(order[:8]):
            raw_vis = _tint(raw_vis, masks[i], palette[rank % len(palette)], 0.35)
        raw_lines = [f"{names[i]} {confs[i]:.3f} area {masks[i].mean():.1%}" for i in order[:8]]
        cv2.imwrite(
            str(out / "2_raw_mask.png"),
            _label(raw_vis, ["raw instances @conf 0.01:"] + (raw_lines or ["(none)"])),
        )

        cv2.imwrite(
            str(out / "3_binary_mask.png"),
            _label(
                _tint(img, perception.mask, (0, 255, 0)),
                [
                    f"conf>={config.confidence_threshold:.2f}: {perception.num_detections} det",
                    f"road {perception.road_fraction:.1%}",
                ],
            ),
        )
        planner_mask = plan.planner_mask
        if planner_mask is None:
            planner_mask = np.zeros((h, w), bool)
        stab_vis = _tint(img, stabilized.mask, (120, 120, 120), 0.5)
        stab_vis = _tint(stab_vis, planner_mask, (0, 255, 0), 0.6)
        cv2.line(stab_vis, (0, plan.roi_top), (w - 1, plan.roi_top), (255, 255, 0), 1)
        cv2.imwrite(
            str(out / "4_stabilized.png"),
            _label(
                stab_vis,
                [
                    f"stabilized road {stabilized.road_fraction:.1%}",
                    "green = component the planner uses, line = ROI top",
                ],
            ),
        )

        traj = _tint(img, planner_mask, (0, 255, 0), 0.3)
        color = (255, 200, 0)
        status = "perceived path"
        if plan.fallback_active:
            color = (0, 0, 255)
            status = f"GATE: {plan.fallback_reason}"
        if len(plan.centerline) >= 2:
            pts = np.round(plan.centerline).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(traj, [pts], False, color, 3, cv2.LINE_AA)
        cv2.line(traj, (w // 2, h - 1), (w // 2, h - 40), (255, 255, 255), 1)
        cv2.imwrite(str(out / "5_trajectory.png"), _label(traj, [status]))

        # Where is the ground? Share of rows below the horizon that the
        # mask covers, and how bright/saturated the image is.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green = ((hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)).mean()
        rows.append(
            {
                "frame": stem,
                "raw_shape": list(raw.shape),
                "model_input": [w, h],
                "mean_bgr": [round(float(v), 1) for v in img.reshape(-1, 3).mean(0)],
                "green_fraction": round(float(green), 3),
                "top_raw": [
                    (names[i], round(confs[i], 3), round(float(masks[i].mean()), 3))
                    for i in order[:5]
                ],
                "max_raw_conf": round(max(confs, default=0.0), 3),
                "n_above_threshold": perception.num_detections,
                "raw_road_fraction": round(perception.road_fraction, 4),
                "stabilized_fraction": round(stabilized.road_fraction, 4),
                "gate": _gate(plan),
                "path_points": int(len(plan.centerline)),
                "bottom_x": _bottom_x(plan),
            }
        )

    print(json.dumps(rows, indent=1))
    (out_root / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"\nPer-stage images in {out_root}")


def _gate(plan) -> str:
    if plan.fallback_active:
        return plan.fallback_reason
    return "ok"


def _bottom_x(plan) -> float | None:
    if len(plan.centerline) == 0:
        return None
    return round(float(plan.centerline[-1, 0]), 1)


def full(args: argparse.Namespace) -> None:
    """Whole pipeline, stereo inline, on saved left/right pairs.

    Reports what the *new* planner inputs (fused traversability, metric
    clearance) do to a frame whose appearance mask is already good.
    """
    from offroad_autonomy.pipeline import AutonomyPipeline
    from offroad_autonomy.types import StereoFramePair, VehicleState

    config = replace(load_config(args.config), stereo_async=False)
    pipeline = AutonomyPipeline(config, start_worker=False)
    src = Path(args.src)
    out_root = Path(args.out)
    rows = []
    for left_path in sorted(src.glob("*_left.png")):
        right_path = left_path.with_name(left_path.name.replace("_left", "_right"))
        left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        pipeline.reset()
        result = None
        # Warm up: the first step has no depth yet (the mask feeds the ROI).
        for i in range(3):
            pair = StereoFramePair(
                left=left,
                right=right,
                timestamp=time.perf_counter(),
                frame_id=i + 1,
                synchronized=True,
                is_new=True,
            )
            result = pipeline.step_result(pair, VehicleState(speed_mps=4.0))
        # Same stabilised mask, planned without any terrain.
        pipeline.planner.reset()
        stab = replace(result.stabilized, traversability=None)
        plain = pipeline.planner.plan(stab, VehicleState(speed_mps=4.0), terrain=None)

        terrain = result.terrain
        rgb = result.perception.rgb_mask
        if rgb is None:
            rgb = result.perception.mask
        vetoed = float((rgb & ~result.perception.mask).sum()) / max(int(rgb.sum()), 1)
        row = {
            "frame": left_path.stem,
            "rgb_road": round(float(rgb.mean()), 3),
            "fused_road": round(result.perception.road_fraction, 3),
            "vetoed_by_depth": round(vetoed, 3),
            "depth_coverage": None,
            "valid_disp": None,
            "obstacle_frac_in_mask": None,
            "min_clearance_m": None,
            "ground_slope_deg": None,
            "gate": _gate(result.plan),
            "plan_no_depth_gate": _gate(plain),
            "bottom_x": _bottom_x(result.plan),
            "throttle": round(result.command.throttle, 3),
            "brake": round(result.command.brake, 3),
            "steer": round(result.command.steering, 3),
        }
        if result.depth is not None:
            row["depth_coverage"] = round(result.depth.coverage, 3)
            row["valid_disp"] = round(result.depth.valid_disparity_fraction, 3)
        if terrain is not None:
            obstacle_px = float((terrain.obstacle_mask & rgb).sum())
            row["obstacle_frac_in_mask"] = round(obstacle_px / max(int(rgb.sum()), 1), 3)
            row["min_clearance_m"] = round(float(terrain.min_forward_clearance_m), 2)
            row["ground_slope_deg"] = round(terrain.ground_slope_deg, 1)
        rows.append(row)

        out = out_root / left_path.stem
        out.mkdir(parents=True, exist_ok=True)
        img = result.frame.preprocessed
        vis = _tint(img, result.perception.mask, (0, 255, 0), 0.3)
        vis = _tint(vis, rgb & ~result.perception.mask, (0, 0, 255), 0.6)
        for plan, color in ((result.plan, (255, 200, 0)), (plain, (255, 0, 255))):
            if len(plan.centerline) >= 2:
                pts = np.round(plan.centerline).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, color, 3, cv2.LINE_AA)
        cv2.imwrite(
            str(out / "6_full_pipeline.png"),
            _label(
                vis,
                [
                    "green=fused mask  red=vetoed by depth",
                    "cyan=plan w/ depth  magenta=plan w/o depth",
                ],
            ),
        )
    pipeline.close()
    print(json.dumps(rows, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="save raw frames from BeamNG")
    cap.add_argument("--out", default=str(ROOT / "output/diagnostics/frames"))
    cap.add_argument("--frames", type=int, default=3)
    cap.add_argument("--spawn", type=int, default=-1, help="-1 = every spawn of the map")

    ana = sub.add_parser("analyze", help="run every stage on saved frames")
    ana.add_argument("--src", default=str(ROOT / "output/diagnostics/frames"))
    ana.add_argument("--out", default=str(ROOT / "output/diagnostics"))
    ana.add_argument("--side", default="left")
    ana.add_argument("--conf", type=float, default=None)
    ana.add_argument("--weights", default="")
    ana.add_argument("--no-clahe", action="store_true")

    fl = sub.add_parser("full", help="whole pipeline incl. inline stereo on saved pairs")
    fl.add_argument("--src", default=str(ROOT / "output/diagnostics/frames"))
    fl.add_argument("--out", default=str(ROOT / "output/diagnostics"))

    args = parser.parse_args()
    {"capture": capture, "analyze": analyze, "full": full}[args.cmd](args)


if __name__ == "__main__":
    main()
