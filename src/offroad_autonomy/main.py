"""Application entry point for the offroad_autonomy pipeline.

Loads configuration, initialises the BeamNG client and the autonomy
pipeline, then runs the main perception-planning-control loop until
interrupted.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.simulation.beamng_client import BeamNGClient
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
_MPH_PER_MPS = 2.2369362920544


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
        speed_mph=speed_mps * _MPH_PER_MPS,
        steering=steering,
        throttle=throttle,
        brake=brake,
        perception_confidence=_mean_confidence(perception_confidences),
        stability_score=stability_score,
        kalman_active=kalman_active,
        fps=fps,
        latency_ms=latency_ms,
    )


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

    frame_count = 0
    t_start = time.perf_counter()
    fps_ema = 0.0

    try:
        client.connect()
        dashboard_window = DashboardWindow(
            _WINDOW_TITLE,
            dashboard.width,
            dashboard.height,
        )
        logger.info("Entering main loop - press Ctrl+C to stop")

        while not _shutdown:
            frame = client.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            state = client.get_vehicle_state()
            t_loop = time.perf_counter()
            result = pipeline.step_result(frame, state)
            client.send_controls(result.command)

            loop_latency_ms = (time.perf_counter() - t_loop) * 1000.0
            fps_now = 1000.0 / max(loop_latency_ms, 1e-6)
            fps_ema = fps_now if fps_ema <= 0.0 else (0.18 * fps_now + 0.82 * fps_ema)

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
            if dashboard_window is not None and not dashboard_window.show(dashboard_frame):
                _shutdown = True
                continue

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
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info("Session complete - %d frames in %.1f s", frame_count, elapsed)


if __name__ == "__main__":
    main()
