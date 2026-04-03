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

import cv2
import numpy as np

from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import ControlCommand, PathPlan, PerceptionResult, PipelineStepResult, StabilizedResult
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


_NO_ROAD_CONFIDENCE_THRESHOLD = 0.05
_NO_ROAD_TIME_S = 2.0
_STUCK_SPEED_MPS = 0.4
_STUCK_THROTTLE_MIN = 0.15
_STUCK_TIME_S = 3.0


class _StuckDetector:
    """Detects stuck or off-road conditions from perception and motion data."""

    def __init__(self) -> None:
        self._no_road_since: float | None = None
        self._stuck_since: float | None = None

    def reset(self) -> None:
        self._no_road_since = None
        self._stuck_since = None

    def update(
        self,
        now: float,
        perception_confidence: float,
        speed_mps: float,
        throttle: float,
    ) -> tuple[bool, str]:
        """Return (should_trigger, reason). Resets internal timers on trigger."""
        if perception_confidence < _NO_ROAD_CONFIDENCE_THRESHOLD:
            if self._no_road_since is None:
                self._no_road_since = now
            elif now - self._no_road_since >= _NO_ROAD_TIME_S:
                self.reset()
                return True, "no road detected"
        else:
            self._no_road_since = None

        if speed_mps < _STUCK_SPEED_MPS and throttle > _STUCK_THROTTLE_MIN:
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since >= _STUCK_TIME_S:
                self.reset()
                return True, "vehicle stuck"
        else:
            self._stuck_since = None

        return False, ""


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
    autopilot_active = True
    _SAFE_STOP_COMMAND = ControlCommand(steering=0.0, throttle=0.0, brake=0.0, parkingbrake=1.0)
    _last_mask: np.ndarray | None = None
    stuck_detector = _StuckDetector()

    try:
        client.connect()
        if config.debug_mode != "normal":
            ann_map = client.get_annotation_map()
            if ann_map:
                pipeline.set_annotation_map(ann_map)
        dashboard_window = DashboardWindow(
            _WINDOW_TITLE,
            dashboard.width,
            dashboard.height,
        )
        logger.info("Entering main loop - press Ctrl+C to stop")
        logger.info("Press E in dashboard to trigger safe stop; press P to resume autopilot")

        while not _shutdown:
            frame = client.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            state = client.get_vehicle_state()
            t_loop = time.perf_counter()

            if autopilot_active:
                annotation_frame = None
                road_mask_gt = None
                if config.debug_mode == "gt_centerline":
                    command = ControlCommand(steering=0.0, throttle=0.2, brake=0.0)
                    gt_plan = client.get_road_gt_plan(state)
                    blank = np.zeros((config.preprocess_height, config.preprocess_width), dtype=bool)
                    perception = PerceptionResult(mask=blank, confidences=[1.0], num_detections=1)
                    stabilized = StabilizedResult(mask=blank, stability_score=1.0)
                    if gt_plan is not None:
                        command = pipeline.controller.compute(gt_plan, state)
                    # if gt_plan is None (transient poll failure) reuse last command
                    result = PipelineStepResult(
                        frame=pipeline.preprocessor.process(frame),
                        perception=perception,
                        stabilized=stabilized,
                        plan=gt_plan or PathPlan(centerline=np.empty((0, 2))),
                        command=command,
                    )
                elif config.debug_mode == "gt_mask":
                    road_mask_gt = client.get_road_mask_image(
                        state, img_w=config.preprocess_width, img_h=config.preprocess_height,
                    )
                    if road_mask_gt is None:
                        annotation_frame = client.capture_annotation_frame()
                    result = pipeline.step_result(
                        frame, state,
                        annotation_frame=annotation_frame,
                        road_mask_gt=road_mask_gt,
                    )
                else:
                    result = pipeline.step_result(frame, state)
                command = result.command
                steering = command.steering
                throttle = command.throttle
                brake = command.brake
                perception_confidences = result.perception.confidences
                stability_score = result.stabilized.stability_score
                kalman_active = result.plan.kalman_active
                plan = result.plan
                mask = result.stabilized.mask
                raw_frame = result.frame.raw
                _last_mask = mask

                triggered, reason = stuck_detector.update(
                    now=t_loop,
                    perception_confidence=_mean_confidence(perception_confidences),
                    speed_mps=state.speed_mps,
                    throttle=throttle,
                )
                if triggered:
                    autopilot_active = False
                    client.park()
                    logger.warning("Safe stop triggered automatically: %s", reason)
            else:
                held = dashboard_window.held_keys if dashboard_window is not None else set()
                if held:
                    steer_val = (-0.4 if "a" in held else 0.0) + (0.4 if "d" in held else 0.0)
                    if "space" in held:
                        command = ControlCommand(steering=steer_val, throttle=0.0, brake=1.0, parkingbrake=1.0)
                    elif "w" in held:
                        command = ControlCommand(steering=steer_val, throttle=0.25, brake=0.0, parkingbrake=0.0)
                    elif "s" in held:
                        command = ControlCommand(steering=steer_val, throttle=0.0, brake=0.35, parkingbrake=0.0)
                    else:
                        command = ControlCommand(steering=steer_val, throttle=0.0, brake=0.0, parkingbrake=0.0)
                else:
                    command = _SAFE_STOP_COMMAND
                steering = command.steering
                throttle = command.throttle
                brake = command.brake
                perception_confidences = []
                stability_score = 0.0
                kalman_active = False
                plan = None
                mask = _last_mask if _last_mask is not None else np.zeros(
                    (frame.shape[0], frame.shape[1]), dtype=np.uint8
                )
                raw_frame = frame

            client.send_controls(command)

            loop_latency_ms = (time.perf_counter() - t_loop) * 1000.0
            fps_now = 1000.0 / max(loop_latency_ms, 1e-6)
            fps_ema = fps_now if fps_ema <= 0.0 else (0.18 * fps_now + 0.82 * fps_ema)

            telemetry = _build_dashboard_telemetry(
                speed_mps=state.speed_mps,
                steering=steering,
                throttle=throttle,
                brake=brake,
                perception_confidences=perception_confidences,
                stability_score=stability_score,
                kalman_active=kalman_active,
                fps=fps_ema,
                latency_ms=loop_latency_ms,
            )
            telemetry.autopilot_active = autopilot_active
            dashboard_frame = dashboard.render(raw_frame, mask, plan, telemetry)

            if dashboard_window is not None and not dashboard_window.show(dashboard_frame):
                _shutdown = True
                continue

            key = dashboard_window.last_key if dashboard_window is not None else -1
            if key == ord("e") or key == ord("E"):
                if autopilot_active:
                    autopilot_active = False
                    client.park()
                    logger.info("Safe stop triggered - autopilot disabled, manual control required")
            elif key == ord("p") or key == ord("P"):
                if not autopilot_active:
                    client.release_park()
                    autopilot_active = True
                    stuck_detector.reset()
                    logger.info("Autopilot resumed")

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
