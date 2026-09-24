#!/usr/bin/env python3
"""Headless closed-loop BeamNG run that logs every control frame.

Drives the default config with no dashboard window and writes one CSV row
per frame (speed, position, steering desired/final, cross-track, heading
error, lookahead, throttle/brake, road fraction, confidence, gate state), so
tracking can be judged from numbers rather than by eye::

    python scripts/closed_loop_log.py --seconds 120 --out output/diagnostics/run.csv \
        --snap-from 40 --summary

``--snap-from T`` also saves the pipeline debug view (raw | mask | planner
input | trajectory + control) every other frame after T seconds, next to the
CSV. Note: there is no stuck / no-road safe stop here - that lives in main.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.utils.config import load_config

_DEBUG_FIELDS = (
    ("desired", "desired_steering", 4),
    ("cte", "cross_track_m", 3),
    ("heading", "heading_error_rad", 4),
    ("lookahead", "lookahead_m", 2),
    ("kappa", "curvature", 4),
    ("target", "target_speed_mps", 2),
    ("reason", "speed_reason", None),
    ("sat", "saturation", 3),
    ("edge_l", "left_boundary_m", 3),
    ("edge_r", "right_boundary_m", 3),
    ("edge_risk", "edge_risk", 3),
    ("clearance", "edge_clearance_m", 3),
    ("lat_acc", "lateral_accel_mps2", 3),
    ("rejoin", "rejoin_m", 2),
)


def _debug_fields(debug) -> dict:
    """Blank columns for manual frames keep every CSV row the same width."""
    fields = {}
    for column, attribute, digits in _DEBUG_FIELDS:
        value = ""
        if debug is not None:
            value = getattr(debug, attribute)
            if digits is not None:
                value = round(value, digits)
        fields[column] = value
    return fields


def summarize(rows: list[dict]) -> None:
    def col(key):
        return np.array([float(r[key]) for r in rows if r[key] != ""])

    t, s, hd, cte, v = col("t"), col("steer"), col("heading"), col("cte"), col("speed")
    x, y = col("x"), col("y")
    significant = s[np.abs(s) > 0.03]
    print(
        f"{len(rows)} frames, {len(rows) / max(t[-1], 1e-6):.1f} FPS, "
        f"{np.sum(np.hypot(np.diff(x), np.diff(y))):.0f} m, mean speed {v.mean():.2f} m/s"
    )
    print(
        f"|steer| mean {np.abs(s).mean():.3f} max {np.abs(s).max():.2f}, "
        f"max change/frame {np.abs(np.diff(s)).max():.3f}, "
        f"sign flips {(np.diff(np.sign(significant)) != 0).sum()}"
    )
    print(
        f"|cross-track| mean {np.abs(cte).mean():.2f} m, "
        f"|heading err| p95 {np.percentile(np.abs(hd), 95):.3f} rad, "
        f"|cross-track| p95 {np.percentile(np.abs(cte), 95):.2f} m, "
        f"gate rejects {sum(r['gate'] != 'ok' for r in rows)}"
    )
    clear = np.array(
        [float(r["clearance"]) for r in rows if r.get("clearance") not in ("", "nan", None)]
    )
    clear = clear[np.isfinite(clear)]
    if len(clear):
        print(
            f"edge clearance: median {np.median(clear):.2f} m, frames < 0.3 m {int((clear < 0.3).sum())}, "
            f"frames outside trail (< 0) {int((clear < 0).sum())}"
        )
    sat = col("sat")
    if len(sat):
        print(
            f"saturation p95 {np.percentile(sat, 95):.0%}, frames at lock (|steer|>0.95) "
            f"{int((np.abs(s) > 0.95).sum())}, max speed while |steer|>0.8: "
            f"{v[1:][np.abs(s[:-1]) > 0.8].max(initial=0.0):.2f} m/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--out", default=str(ROOT / "output/diagnostics/closed_loop.csv"))
    parser.add_argument("--snap-from", type=float, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dash = snap_dir = None
    if args.snap_from is not None:
        import cv2

        from offroad_autonomy.visualization import AutonomyDashboard

        dash = AutonomyDashboard(
            width=1600, height=900, colors=cfg.dashboard_colors, sensor=cfg.left_camera.sensor
        )
        snap_dir = out.with_suffix("")
        snap_dir.mkdir(parents=True, exist_ok=True)

    client = BeamNGClient(cfg)
    pipe = AutonomyPipeline(cfg)
    rows: list[dict] = []
    try:
        client.connect()
        client.release_park()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            pair = client.capture_pair()
            if pair is None or not pipe.has_input(pair):
                time.sleep(0.005)
                continue
            state = client.get_vehicle_state()
            res = pipe.step_result(pair, state)
            client.send_controls(res.command)
            elapsed = time.perf_counter() - t0
            if dash is not None and elapsed >= args.snap_from and len(rows) % 2 == 0:
                cv2.imwrite(str(snap_dir / f"{len(rows):04d}.jpg"), dash._render_pipeline_view(res))
            gate = "ok"
            if res.plan.fallback_active:
                gate = res.plan.fallback_reason
            row = {
                "t": round(elapsed, 3),
                "speed": round(state.speed_mps, 3),
                "x": round(state.position[0], 2),
                "y": round(state.position[1], 2),
                "steer": round(res.command.steering, 4),
            }
            row.update(_debug_fields(res.command.debug))
            row.update(
                throttle=round(res.command.throttle, 3),
                brake=round(res.command.brake, 3),
                road=round(res.stabilized.road_fraction, 3),
                conf=round(max(res.perception.confidences, default=0.0), 3),
                gate=gate,
            )
            rows.append(row)
    finally:
        pipe.close()
        client.disconnect()
        if rows:
            with open(out, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {len(rows)} rows to {out}")
            if args.summary:
                summarize(rows)


if __name__ == "__main__":
    main()
