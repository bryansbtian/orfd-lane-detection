#!/usr/bin/env python3
"""Offline closed-loop check of the controller on a synthetic S-bend.

A kinematic bicycle drives an S-bend at the real loop rate (~4 FPS) with
two frames of capture-to-actuation delay and BeamNG's measured steering
slew (~0.9/s). Every frame the visible centreline (1-14 m
ahead) is fitted with a quadratic in the vehicle frame - as the baseline
planner does - projected into the segmentation camera, and handed to the
real ``StanleyController``. No BeamNG, no perception: this isolates the
controller, so gains can be tuned in seconds and regressions caught in CI.

    python scripts/sbend_sim.py --radius 9 --fps 4
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState
from offroad_autonomy.utils.config import load_config


def s_bend(radius: float, straight: float = 30.0, step: float = 0.1) -> np.ndarray:
    """World centreline: straight, 90 deg left, 90 deg right, straight. (x fwd, y left)."""
    pts = [np.array([x, 0.0]) for x in np.arange(0.0, straight, step)]
    pos, heading = np.array([straight, 0.0]), 0.0
    for turn in (+1, -1):
        for _ in range(int(0.5 * math.pi * radius / step)):
            heading += turn * step / radius
            pos = pos + step * np.array([math.cos(heading), math.sin(heading)])
            pts.append(pos.copy())
    for _ in range(int(straight / step)):
        pos = pos + step * np.array([math.cos(heading), math.sin(heading)])
        pts.append(pos.copy())
    return np.array(pts)


@dataclass
class SimResult:
    max_abs_steer: float
    max_lateral_error: float
    frames_at_lock: int
    min_speed: float
    max_speed: float
    completed: bool
    log: list
    #: Centring, after the first ``_WARMUP`` frames (the misaligned start).
    mean_abs_error: float = 0.0
    p95_error: float = 0.0
    #: Frames where the vehicle body was within 0.3 m of the trail edge.
    edge_frames: int = 0
    #: Mean |change in steering| per frame - smoothness.
    steer_jerk: float = 0.0


_WARMUP = 20
_HALF_WIDTH = 0.95


def _finish(log, max_err, lock_frames, completed, trail_width) -> SimResult:
    errs = np.array([e["err"] for e in log[_WARMUP:]] or [0.0])
    steer = np.array([e["steer"] for e in log])
    edge = trail_width / 2.0 - _HALF_WIDTH - 0.3
    steer_jerk = 0.0
    if len(steer) > 1:
        steer_jerk = float(np.abs(np.diff(steer)).mean())
    return SimResult(
        max(abs(e["steer"]) for e in log),
        max_err,
        lock_frames,
        min(e["v"] for e in log),
        max(e["v"] for e in log),
        completed,
        log,
        mean_abs_error=float(errs.mean()),
        p95_error=float(np.percentile(errs, 95)),
        edge_frames=int((errs > edge).sum()),
        steer_jerk=steer_jerk,
    )


def trail_mask(
    camera: CameraModel,
    road: np.ndarray,
    normals: np.ndarray,
    width: float,
    cx: float,
    cy: float,
    yaw: float,
    roi_top: int,
    rng: np.random.Generator | None = None,
    holes: int = 40,
) -> np.ndarray:
    """The planner's road component: the trail polygon, ROI band only."""
    h, w = camera.height, camera.width
    mask = np.zeros((h, w), np.uint8)
    edges = []
    for side in (+1.0, -1.0):  # left (+normal), right
        pts = road + side * 0.5 * width * normals
        d = pts - np.array([cx, cy])
        f = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        r = d[:, 0] * math.sin(yaw) - d[:, 1] * math.cos(yaw)
        keep = (f > 0.3) & (f < 18.0)
        if keep.sum() < 2:
            return mask.astype(bool)
        edges.append(camera.ground_to_image(f[keep], r[keep]))
    poly = np.vstack([edges[0], edges[1][::-1]])
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    mask[:roi_top] = 0
    # Real masks are full of small holes (depth vetoes, specks); an edge
    # estimator that reads a hole as the trail edge fails in BeamNG.
    if rng is not None:
        for _ in range(holes):
            y = int(rng.integers(roi_top, h))
            x = int(rng.integers(0, w))
            hw = int(rng.integers(3, 14))
            mask[y : y + 3, x : x + hw] = 0
    return mask.astype(bool)


def simulate(
    config: PipelineConfig,
    radius: float = 9.0,
    fps: float = 4.0,
    start_speed: float = 0.0,
    seed: int = 0,
    noise_m: float = 0.05,
    visible_m: float = 14.0,
    fit_degree: int = 2,
    start_yaw: float = 0.3,
    slope_noise: float = 0.04,
    curv_noise: float = 0.01,
    delay_frames: int = 2,
    steer_rate_per_s: float = 0.9,
    trail_width: float = 6.0,
) -> SimResult:
    """Closed loop on an S-bend.

    Defaults match what BeamNG showed: a standing start 0.3 rad off the road
    direction, the Hopper's ~4 m/s^2 of launch at 0.45 throttle, and a path
    whose fitted slope and curvature jitter from frame to frame (the fit is
    redone on a new mask every frame), not just per-point noise.
    """
    rng = np.random.default_rng(seed)
    camera = CameraModel(
        config.segmentation_camera, config.preprocess_width, config.preprocess_height
    )
    controller = StanleyController(config, camera=camera)
    road = s_bend(radius)
    tangent = np.gradient(road, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # to the left
    roi_top = int(round(camera.height * (1.0 - config.planner_roi_height)))
    dt = 1.0 / fps
    L = config.wheelbase_m
    max_wheel = math.radians(config.max_wheel_angle_deg)
    cam_ahead = 1.4  # camera ahead of the rear-axle reference point (m)

    x, y, yaw, v = 0.0, 0.0, start_yaw, start_speed
    # Commands reach the wheel `delay_frames` late (capture -> actuation), then
    # BeamNG slews steering_input at ~0.9/s (measured on the Hopper: 0 -> 0.6
    # in ~0.7 s, with <0.1 s of pure delay).
    pending = [0.0] * max(1, delay_frames)
    wheel_cmd = 0.0
    log, max_err, lock_frames = [], 0.0, 0
    for _ in range(int(120 * fps)):
        # Visible road in the camera frame: forward f, right r.
        cx, cy = x + cam_ahead * math.cos(yaw), y + cam_ahead * math.sin(yaw)
        d = road - np.array([cx, cy])
        f = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
        r = d[:, 0] * math.sin(yaw) - d[:, 1] * math.cos(yaw)
        nearest = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))
        err_now = float(np.hypot(d[nearest, 0], d[nearest, 1]))
        max_err = max(max_err, err_now)
        if nearest >= len(road) - 5:
            return _finish(log, max_err, lock_frames, True, trail_width)
        ahead = np.arange(nearest, len(road))
        vis = ahead[(f[ahead] > 1.0) & (f[ahead] < visible_m)]
        if len(vis) > 2:
            vis = vis[: np.argmax(np.diff(f[vis]) < 0) or len(vis)]
        # The baseline planner's continuity rule ends the path where the road
        # turns past 45 deg to the car (|dr/df| > 1), so do the same here.
        if len(vis) > 2:
            steep = np.abs(np.diff(r[vis])) > np.abs(np.diff(f[vis])) + 1e-6
            if steep.any():
                vis = vis[: np.argmax(steep) + 1]
        plan = PathPlan(centerline=np.empty((0, 2), np.float32))
        if len(vis) >= 6:
            coeffs = np.polyfit(f[vis], r[vis] + rng.normal(0, noise_m, len(vis)), fit_degree)
            coeffs[-2] += rng.normal(0, slope_noise)  # heading jitter
            coeffs[-3] += rng.normal(0, curv_noise) / 2.0  # curvature jitter
            fs = np.linspace(f[vis].min(), f[vis].max(), 24)
            plan = PathPlan(centerline=camera.ground_to_image(fs, np.polyval(coeffs, fs))[::-1])
            if trail_width > 0:
                plan.planner_mask = trail_mask(
                    camera, road, normals, trail_width, cx, cy, yaw, roi_top, rng
                )
                plan.roi_top = roi_top

        cmd = controller.compute(plan, VehicleState(speed_mps=v))
        # Bicycle model with the PREVIOUS command (latency), + = right.
        step = steer_rate_per_s * dt
        wheel_cmd += float(np.clip(pending[0] - wheel_cmd, -step, step))
        wheel = wheel_cmd * max_wheel
        yaw -= v * math.tan(wheel) / L * dt
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        # Hopper in BeamNG: 0 -> 5.7 m/s in ~1.4 s at 0.45 throttle.
        v = max(0.0, v + (9.0 * cmd.throttle - 8.0 * cmd.brake - 0.3) * dt)
        pending = pending[1:] + [cmd.steering]
        lock_frames += abs(cmd.steering) > 0.95
        dbg = cmd.debug
        log.append(
            {
                "steer": cmd.steering,
                "v": v,
                "err": err_now,
                "kappa": dbg.curvature,
                "cte": dbg.cross_track_m,
                "des": dbg.desired_steering,
                "ff": dbg.feedforward_steering,
                "psi": dbg.heading_error_rad,
                "rejoin": dbg.rejoin_m,
                "path_end": dbg.path_end_m,
                "target": dbg.target_speed_mps,
                "la": dbg.lookahead_m,
                "sat": dbg.saturation,
                "reason": dbg.speed_reason,
            }
        )
    return _finish(log, max_err, lock_frames, False, trail_width)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--radius", type=float, default=9.0)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    res = simulate(config, radius=args.radius, fps=args.fps)
    print(
        f"radius {args.radius} m @ {args.fps} FPS: completed={res.completed} "
        f"max|steer|={res.max_abs_steer:.2f} lock frames={res.frames_at_lock} "
        f"max off-centre={res.max_lateral_error:.2f} m speed {res.min_speed:.1f}-{res.max_speed:.1f} m/s"
    )
    print(
        f"  centring: mean {res.mean_abs_error:.2f} m  p95 {res.p95_error:.2f} m  "
        f"edge frames {res.edge_frames}  steer jerk {res.steer_jerk:.3f}/frame"
    )
    if args.verbose:
        for i, e in enumerate(res.log[::2]):
            print(
                f"{2 * i:4d} v={e['v']:.2f} tgt={e['target']:.2f} ({e['reason']:10s}) "
                f"steer={e['steer']:+.2f} kappa={e['kappa']:+.3f} la={e['la']:.1f} sat={e['sat']:.2f} err={e['err']:.2f}"
            )
    # A non-zero exit lets CI reject a tuning change that leaves the trail.
    if not res.completed or res.edge_frames > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
