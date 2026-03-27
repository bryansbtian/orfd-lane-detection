"""Application entry point for the offroad_autonomy pipeline.

Loads configuration, initialises the BeamNG client and the autonomy
pipeline, then runs the main perception-planning-control loop until
interrupted.

Drive modes (toggle via keyboard while the dashboard window is focused):
    P  — Path-planning (autonomous via Stanley + Kalman)
    M  — Manual (no controls sent, drive with keyboard/wheel)
    B  — BeamNG AI (simulator's built-in AI driver)
    Q  — Quit
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import ControlCommand, PipelineConfig, VehicleState
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.logger import setup_logger
from offroad_autonomy.visualization import (
    AutonomyDashboard,
    DashboardTelemetry,
    DashboardWindow,
)

logger = logging.getLogger("offroad_autonomy.main")

_shutdown = False
_WINDOW_TITLE = "Off-Road Autonomy Dashboard"

DRIVE_MANUAL = "MANUAL"
DRIVE_AUTO = "AUTO"
DRIVE_BEAMNG_AI = "BEAMNG_AI"


def _signal_handler(signum, frame) -> None:
    del signum, frame
    global _shutdown
    _shutdown = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="offroad_autonomy - autonomous off-road driving in BeamNG",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _mean_confidence(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_dashboard_telemetry(
    speed_mps: float,
    steering: float,
    throttle: float,
    brake: float,
    perception_confidences: list[float],
    stability_score: float,
    kalman_active: bool,
    fps: float,
    latency_ms: float,
) -> DashboardTelemetry:
    return DashboardTelemetry(
        speed_mps=speed_mps,
        steering=steering,
        throttle=throttle,
        brake=brake,
        perception_confidence=_mean_confidence(perception_confidences),
        stability_score=stability_score,
        kalman_active=kalman_active,
        fps=fps,
        latency_ms=latency_ms,
    )


# ── CSV logger ───────────────────────────────────────────────────────────────

LOG_FIELDS = [
    "frame", "time_s", "lateral_offset_m", "heading_error_deg",
    "speed_mps", "steering", "throttle", "brake", "drive_mode",
]


class RunLogger:
    """Lightweight CSV logger — writes essentials each frame."""

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_dir / f"run_{stamp}.csv"
        self._fp = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=LOG_FIELDS)
        self._writer.writeheader()
        self._fp.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _shutdown
    _shutdown = False
    args = build_parser().parse_args()

    level = getattr(logging, args.log_level)
    setup_logger(level=level)

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Configuration file not found: {config_path}")

    config = load_config(config_path)
    logger.info("Configuration loaded from %s", config_path)

    signal.signal(signal.SIGINT, _signal_handler)

    client = BeamNGClient(config)
    pipeline = AutonomyPipeline(config)
    dashboard = AutonomyDashboard(
        width=1600,
        height=900,
        colors=config.dashboard_colors,
    )
    dashboard_window = None
    run_log = RunLogger(Path("simulator_log"))

    frame_count = 0
    t_start = time.perf_counter()
    fps_ema = 0.0
    drive_mode = DRIVE_MANUAL

    print("=" * 60)
    print("  Off-Road Autonomy Pipeline")
    print("=" * 60)
    print(f"  Mode: {drive_mode}")
    print("  P=PathPlan  M=Manual  B=BeamNG AI  Q=Quit")
    print("=" * 60)

    try:
        client.connect()
        dashboard_window = DashboardWindow(
            _WINDOW_TITLE,
            dashboard.width,
            dashboard.height,
        )
        logger.info("Entering main loop — default mode: MANUAL")

        while not _shutdown:
            frame = client.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            state = client.get_vehicle_state()
            t_loop = time.perf_counter()

            # Always run perception + planning (for dashboard display)
            result = pipeline.step_result(frame, state)

            # Only send controls in AUTO mode; manual = hands off
            if drive_mode == DRIVE_AUTO:
                client.send_controls(result.command)

            loop_latency_ms = (time.perf_counter() - t_loop) * 1000.0
            fps_now = 1000.0 / max(loop_latency_ms, 1e-6)
            fps_ema = fps_now if fps_ema <= 0.0 else (0.18 * fps_now + 0.82 * fps_ema)

            # Compute lateral offset from plan centerline
            lateral_offset_m = 0.0
            heading_error_deg = 0.0
            if len(result.plan.centerline) > 0:
                cx_bottom = result.plan.centerline[-1, 0]
                frame_w = result.frame.width
                # Normalize to [-1, 1] range (rough estimate, not ground truth)
                lateral_offset_m = (cx_bottom - frame_w / 2.0) / (frame_w / 2.0)
            heading_error_deg = result.plan.heading_rad * 57.2958  # rad to deg

            # Log
            elapsed = time.perf_counter() - t_start
            cmd = result.command if drive_mode == DRIVE_AUTO else ControlCommand()
            run_log.log({
                "frame": frame_count,
                "time_s": round(elapsed, 3),
                "lateral_offset_m": round(lateral_offset_m, 4),
                "heading_error_deg": round(heading_error_deg, 2),
                "speed_mps": round(state.speed_mps, 3),
                "steering": round(cmd.steering, 4),
                "throttle": round(cmd.throttle, 3),
                "brake": round(cmd.brake, 3),
                "drive_mode": drive_mode,
            })

            telemetry = _build_dashboard_telemetry(
                speed_mps=state.speed_mps,
                steering=result.command.steering,
                throttle=result.command.throttle,
                brake=result.command.brake,
                perception_confidences=result.perception.confidences,
                stability_score=result.stabilized.stability_score,
                kalman_active=result.plan.kalman_active,
                fps=fps_ema,
                latency_ms=loop_latency_ms,
            )
            dashboard_frame = dashboard.render(
                result.frame.raw,
                result.stabilized.mask,
                result.plan,
                telemetry,
            )

            key = -1
            if dashboard_window is not None:
                key = dashboard_window.show(dashboard_frame)
                if key == -2:
                    _shutdown = True
                    continue

            # Key handling
            if key == ord("p") or key == ord("P"):
                drive_mode = DRIVE_AUTO
                client.set_beamng_ai(False)
                logger.info("Switched to AUTO (path planning)")
                print(f"[MODE] {drive_mode}")
            elif key == ord("m") or key == ord("M"):
                drive_mode = DRIVE_MANUAL
                client.set_beamng_ai(False)
                logger.info("Switched to MANUAL")
                print(f"[MODE] {drive_mode}")
            elif key == ord("b") or key == ord("B"):
                drive_mode = DRIVE_BEAMNG_AI
                client.set_beamng_ai(True)
                logger.info("Switched to BEAMNG AI")
                print(f"[MODE] {drive_mode}")

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                logger.info("Processed %d frames (%.1f FPS avg)", frame_count, fps)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if dashboard_window is not None:
            dashboard_window.close()
        run_log.close()
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info("Log saved: %s", run_log.path)
        logger.info("Session complete - %d frames in %.1f s", frame_count, elapsed)


if __name__ == "__main__":
    main()
