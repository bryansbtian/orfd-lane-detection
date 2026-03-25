"""Application entry point for the offroad_autonomy pipeline.

Loads configuration, initialises the BeamNG client and the autonomy
pipeline, then runs the main perception→planning→control loop until
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

logger = logging.getLogger("offroad_autonomy.main")

_shutdown = False


def _signal_handler(signum, frame) -> None:
    global _shutdown
    _shutdown = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="offroad_autonomy — autonomous off-road driving in BeamNG",
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


def main() -> None:
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

    try:
        client.connect()
        logger.info("Entering main loop — press Ctrl+C to stop")

        frame_count = 0
        t_start = time.perf_counter()

        while not _shutdown:
            frame = client.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            state = client.get_vehicle_state()
            command = pipeline.step(frame, state)
            client.send_controls(command)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(
                    "Processed %d frames (%.1f FPS avg)", frame_count, fps,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info(
            "Session complete — %d frames in %.1f s", frame_count, elapsed,
        )


if __name__ == "__main__":
    main()
