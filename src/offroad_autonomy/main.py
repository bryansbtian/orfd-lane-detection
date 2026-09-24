"""Entry point: connect to BeamNG and run the autonomy loop until stopped.

Stereo depth and the dashboard each run on their own thread, so this loop
never waits for either.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.ego_mask import EgoMask
from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.runtime.benchmark import BenchmarkRecorder
from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker
from offroad_autonomy.runtime.timing import DISPLAY_STAGES, MAIN_STAGES, STEREO_STAGES
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import (
    DEBUG_VIEWS,
    ControlCommand,
    PathPlan,
    PipelineConfig,
    PipelineStepResult,
    VehicleState,
)
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.logger import setup_logger
from offroad_autonomy.visualization import (
    AutonomyDashboard,
    DashboardTelemetry,
    DashboardWindow,
    GroundProjector,
)

logger = logging.getLogger("offroad_autonomy.main")

_shutdown = False
_WINDOW_TITLE = "Off-Road Autonomy Dashboard"
_MPH_PER_MPS = 2.2369362920544

_STUCK_SPEED_MPS = 0.4
_STUCK_THROTTLE_MIN = 0.15
_STUCK_TIME_S = 3.0

_MANUAL_STEER = 0.4
_MANUAL_THROTTLE = 0.25
_MANUAL_BRAKE = 0.35

_DEPTH_STATES = {"": "LIVE", "stale": "STALE", "warming up": "WARMING UP", "off": "OFF"}

_SAFE_STOP_COMMAND = ControlCommand(steering=0.0, throttle=0.0, brake=0.0, parkingbrake=1.0)


def _signal_handler(signum, frame) -> None:
    del signum, frame
    global _shutdown
    _shutdown = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="offroad_autonomy - autonomous off-road driving in BeamNG",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the dashboard window (overrides ui.headless).",
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds and write a benchmark report.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=float,
        default=5.0,
        help="Seconds excluded from the report while models and caches warm up.",
    )
    parser.add_argument(
        "--benchmark-out",
        default="",
        help="Benchmark JSON path (default: output/benchmarks/<label>.json).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Name for the benchmark run (default: the config file name).",
    )
    return parser


class _StuckDetector:
    """Safe-stop trigger.

    The no-road test watches the share of *valid* pixels that came back
    traversable rather than a mean detection score, because a score drops as
    soon as the segmenter spends detections on bodywork and would stop the
    vehicle on a perfectly clear trail.
    """

    def __init__(
        self,
        min_road_fraction: float = 0.015,
        no_road_time_s: float = 2.0,
    ) -> None:
        self._min_road_fraction = float(min_road_fraction)
        self._no_road_time_s = float(no_road_time_s)
        self._no_road_since: float | None = None
        self._stuck_since: float | None = None

    def reset(self) -> None:
        self._no_road_since = None
        self._stuck_since = None

    def update(
        self,
        now: float,
        road_fraction: float,
        speed_mps: float,
        throttle: float,
    ) -> tuple[bool, str]:
        if road_fraction < self._min_road_fraction:
            if self._no_road_since is None:
                self._no_road_since = now
            elif now - self._no_road_since >= self._no_road_time_s:
                self.reset()
                return True, (f"no traversable road ({road_fraction:.1%} of valid pixels)")
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
    state: VehicleState,
    command: ControlCommand,
    result: PipelineStepResult,
    plan: PathPlan | None,
    pipeline: AutonomyPipeline,
    autopilot_active: bool,
    timing_overlay: bool,
    display: DisplayWorker,
) -> DashboardTelemetry:
    main_stats = pipeline.stats
    stereo_stats = pipeline.stereo_stats
    primary = main_stats.stage("primary_loop")
    kalman = bool(plan is not None and plan.kalman_active)

    depth_state = _DEPTH_STATES.get(result.depth_fallback, result.depth_fallback.upper())
    # Read the gate off the pipeline's own plan, not ``plan``: that one is
    # withheld under safe stop, which would hide the gate exactly when the
    # stack has given up.
    fallback = []
    if not autopilot_active:
        fallback.append("SAFE STOP")
    if result.plan.fallback_active:
        if result.plan.speed_scale > 0.0:
            fallback.append("GATE HOLD")
        else:
            fallback.append("GATE STOP")
    if kalman:
        fallback.append("KALMAN")
    if depth_state in ("STALE", "WARMING UP"):
        fallback.append("NO DEPTH")
    fallback_state = "NONE"
    if fallback:
        fallback_state = " + ".join(fallback)

    sync_ok = True
    if result.frames is not None:
        sync_ok = result.frames.synchronized

    telemetry = DashboardTelemetry(
        speed_mph=state.speed_mps * _MPH_PER_MPS,
        steering=command.steering,
        throttle=command.throttle,
        brake=command.brake,
        perception_confidence=_mean_confidence(result.perception.confidences),
        stability_score=result.stabilized.stability_score,
        kalman_active=kalman,
        fps=main_stats.fps(),
        latency_ms=primary.mean_ms,
        latency_p95_ms=primary.p95_ms,
        autopilot_active=autopilot_active,
        road_fraction=result.stabilized.road_fraction,
        ego_coverage=pipeline.ego_coverage,
        segmentation_mode=pipeline.view.mode,
        fallback_state=fallback_state,
        depth_active=result.depth is not None,
        depth_state=depth_state,
        depth_age_ms=result.depth_age_s * 1000.0,
        sync_ok=sync_ok,
        dashboard_fps=display.fps(),
    )

    depth = result.depth
    if depth is not None:
        telemetry.depth_coverage = depth.coverage
        telemetry.valid_disparity_fraction = depth.valid_disparity_fraction
        telemetry.median_forward_depth_m = depth.median_forward_depth_m
        telemetry.min_corridor_depth_m = depth.min_corridor_depth_m
    if result.terrain is not None:
        telemetry.min_clearance_m = result.terrain.min_forward_clearance_m
    if stereo_stats is not None:
        total = stereo_stats.stage("stereo_total")
        telemetry.stereo_fps = stereo_stats.fps()
        telemetry.stereo_latency_ms = total.mean_ms
        telemetry.stereo_latency_p95_ms = total.p95_ms

    if timing_overlay:
        # One heading per thread, so a slow dashboard can never be mistaken
        # for a slow vehicle.
        lines = ["AUTONOMY LOOP"] + main_stats.format_lines(MAIN_STAGES)
        if stereo_stats is not None:
            lines += ["", "STEREO WORKER"] + stereo_stats.format_lines(STEREO_STAGES)
        lines += ["", "DASHBOARD"] + display.stats.format_lines(DISPLAY_STAGES)
        telemetry.timing_lines = lines
    return telemetry


def _log_runtime(pipeline: AutonomyPipeline, display: DisplayWorker | None = None) -> None:
    primary = pipeline.stats.stage("primary_loop")
    message = (
        f"autonomy {pipeline.stats.fps():.1f} FPS  primary {primary.mean_ms:.1f}/"
        f"{primary.p95_ms:.1f} ms (mean/p95)"
    )
    stereo = pipeline.stereo_stats
    worker = pipeline.stereo_worker
    if stereo is not None and worker is not None:
        total = stereo.stage("stereo_total")
        message += (
            f" | stereo {stereo.fps():.1f} Hz  {total.mean_ms:.1f}/{total.p95_ms:.1f} ms"
            f"  (done {worker.completed}, superseded {worker.dropped}, failed {worker.failed})"
        )
    if display is not None:
        total = display.stats.stage("dashboard_total")
        message += (
            f" | dashboard {display.fps():.1f} Hz  {total.mean_ms:.1f} ms"
            f"  (drawn {display.rendered}, superseded {display.dropped})"
        )
    logger.info(message)
    for line in pipeline.stats.format_lines(MAIN_STAGES):
        logger.debug("  %s", line)
    if stereo is not None:
        for line in stereo.format_lines(STEREO_STAGES):
            logger.debug("  %s", line)
    if display is not None:
        for line in display.stats.format_lines(DISPLAY_STAGES):
            logger.debug("  %s", line)


def _manual_command(held: set[str]) -> ControlCommand:
    if not held:
        return _SAFE_STOP_COMMAND
    steer = 0.0
    if "a" in held:
        steer -= _MANUAL_STEER
    if "d" in held:
        steer += _MANUAL_STEER
    if "space" in held:
        return ControlCommand(steering=steer, throttle=0.0, brake=1.0, parkingbrake=1.0)
    if "w" in held:
        return ControlCommand(
            steering=steer, throttle=_MANUAL_THROTTLE, brake=0.0, parkingbrake=0.0
        )
    if "s" in held:
        return ControlCommand(steering=steer, throttle=0.0, brake=_MANUAL_BRAKE, parkingbrake=0.0)
    return ControlCommand(steering=steer, throttle=0.0, brake=0.0, parkingbrake=0.0)


def _build_projector(
    config: PipelineConfig,
    pipeline: AutonomyPipeline,
) -> tuple[GroundProjector | None, np.ndarray | None]:
    """Homography and overlay clip for drawing on the display camera.

    The display camera is never passed to ``AutonomyPipeline``,
    ``PerceptionView`` or ``build_camera_models``, so no dashboard setting can
    put it on the compute path.
    """
    if not config.display_rig.enabled:
        return None, None

    spec = config.display_camera
    target = CameraModel(spec, spec.width, spec.height)
    try:
        projector = GroundProjector(pipeline.view.camera, target)
    except ValueError as exc:
        # A display camera aimed at the sky is cosmetic, not a reason to
        # refuse to drive.
        logger.error("Display reprojection unavailable: %s", exc)
        return None, None

    clip = EgoMask(spec.ego_mask).excluded((target.height, target.width))
    return projector, clip


def _open_window(width: int, height: int) -> DashboardWindow | None:
    """A missing display must not stop the vehicle, so a failed window
    degrades to headless instead of aborting the run."""
    try:
        return DashboardWindow(_WINDOW_TITLE, width, height)
    except RuntimeError as exc:
        logger.warning("No dashboard window (%s) - running headless", exc)
        return None


def _handle_key(
    key: int,
    autopilot_active: bool,
    client: BeamNGClient,
    stuck_detector: _StuckDetector,
) -> bool:
    if key in (ord("e"), ord("E")) and autopilot_active:
        client.park()
        logger.info("Safe stop triggered - autopilot disabled, manual control required")
        return False
    if key in (ord("p"), ord("P")) and not autopilot_active:
        client.release_park()
        stuck_detector.reset()
        logger.info("Autopilot resumed")
        return True
    return autopilot_active


def main() -> None:
    global _shutdown
    _shutdown = False
    args = build_parser().parse_args()

    setup_logger(level=getattr(logging, args.log_level))

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Configuration file not found: {config_path}")

    config = load_config(config_path)
    if args.headless:
        config.ui_headless = True
    logger.info("Configuration loaded from %s", config_path)

    # SIGTERM is what `docker stop` sends; without a handler the process dies
    # mid-loop and the vehicle is never parked.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    client = BeamNGClient(config)
    pipeline = AutonomyPipeline(config)
    dashboard_window: DashboardWindow | None = None
    display: DisplayWorker | None = None
    stats = pipeline.stats

    debug_view = config.ui_debug_view
    timing_overlay = config.ui_timing_overlay
    render_every = max(1, config.ui_render_every_n)

    recorder: BenchmarkRecorder | None = None
    if args.benchmark_seconds > 0 or args.benchmark_out:
        recorder = BenchmarkRecorder(
            label=args.label or config_path.stem,
            config_path=str(config_path),
            valid_roi=pipeline.valid_roi,
            warmup_s=args.benchmark_warmup,
        )

    frame_count = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    autopilot_active = True
    stuck_detector = _StuckDetector(
        min_road_fraction=config.safety_min_road_fraction,
        no_road_time_s=config.safety_no_road_time_s,
    )

    try:
        client.connect()

        if not config.ui_headless:
            projector, display_clip = _build_projector(config, pipeline)
            dashboard = AutonomyDashboard(
                width=1600,
                height=900,
                colors=config.dashboard_colors,
                sensor=config.left_camera.sensor,
                projector=projector,
                display_clip=display_clip,
            )
            dashboard_window = _open_window(dashboard.width, dashboard.height)

        if dashboard_window is not None:
            window = dashboard_window

            def _render(state: DisplayState, frame) -> np.ndarray:
                return dashboard.render(
                    state.result,
                    state.telemetry,
                    plan=state.plan,
                    valid_roi=state.valid_roi,
                    debug_view=state.debug_view,
                    timing_overlay=state.timing_overlay,
                    stitched=state.stitched,
                    depth_roi=state.depth_roi,
                    display_frame=frame,
                )

            capture = None
            if config.display_rig.enabled:
                capture = client.capture_display
            # Tkinter's event loop belongs to the thread that built the root
            # window, so only the OpenCV backend can be driven off-thread.
            display = DisplayWorker(
                render=_render,
                show=window.show,
                read_key=lambda: window.last_key,
                capture=capture,
                rate_hz=config.ui_display_rate_hz,
                asynchronous=config.ui_display_async and window.backend == "opencv",
            )
            if config.ui_display_async and not display.asynchronous:
                logger.warning(
                    "Dashboard is running inline: the '%s' window backend cannot be "
                    "driven from another thread, so rendering is part of the loop period",
                    window.backend,
                )
            display.start()
            logger.info(
                "Keys: E safe stop, P resume, 0-9 debug view (%s), T timing overlay",
                " ".join(f"{i}={name}" for i, name in enumerate(DEBUG_VIEWS)),
            )
        else:
            logger.info("Headless: no dashboard; manual control is unavailable")

        logger.info(
            "Perception on the '%s' camera; ego-vehicle exclusion %.1f%% of that view",
            pipeline.view.mode,
            pipeline.ego_coverage * 100.0,
        )
        logger.info("Entering main loop - Ctrl+C or SIGTERM to stop")
        t_start = time.perf_counter()

        while not _shutdown:
            t_iter = time.perf_counter()
            with stats.time("capture"):
                pair = client.capture_pair()
            if pair is None or not pipeline.has_input(pair):
                time.sleep(0.005)
                continue

            with stats.time("vehicle_state"):
                state = client.get_vehicle_state()

            # Perception keeps running under safe stop so the operator can see
            # the road come back before handing control over again.
            result = pipeline.step_result(pair, state)
            plan = result.plan

            if autopilot_active:
                command = result.command
                triggered, reason = stuck_detector.update(
                    now=t_iter,
                    road_fraction=result.stabilized.road_fraction,
                    speed_mps=state.speed_mps,
                    throttle=command.throttle,
                )
                if triggered:
                    autopilot_active = False
                    client.park()
                    logger.warning("Safe stop triggered automatically: %s", reason)
            else:
                held: set[str] = set()
                if dashboard_window is not None:
                    held = dashboard_window.held_keys
                command = _manual_command(held)
                # Withholding the plan keeps the dashboard from suggesting the
                # stack is steering while it is not.
                plan = None

            with stats.time("actuation"):
                client.send_controls(command)
            primary_ms = (time.perf_counter() - t_iter) * 1000.0
            stats.record("primary_loop", primary_ms)

            # The command has already gone out and primary_loop is recorded,
            # so nothing below counts as autonomy latency.
            if display is not None and frame_count % render_every == 0:
                telemetry = _build_dashboard_telemetry(
                    state,
                    command,
                    result,
                    plan,
                    pipeline,
                    autopilot_active,
                    timing_overlay,
                    display,
                )
                stitched = None
                if debug_view == "stitched":
                    stitched = pipeline.view.stitched(pair)
                depth_roi = None
                if debug_view == "roi" and pipeline.depth_estimator is not None:
                    depth_roi = pipeline.depth_estimator.roi_preview(
                        result.stabilized.mask, pipeline.valid_roi
                    )
                display.publish(
                    DisplayState(
                        result=result,
                        telemetry=telemetry,
                        plan=plan,
                        valid_roi=pipeline.valid_roi,
                        debug_view=debug_view,
                        timing_overlay=timing_overlay,
                        stitched=stitched,
                        depth_roi=depth_roi,
                    )
                )

            if display is not None:
                if display.closed:
                    _shutdown = True
                for key in display.drain_keys():
                    if key in (ord("t"), ord("T")):
                        timing_overlay = not timing_overlay
                    elif ord("0") <= key <= ord("9") and key - ord("0") < len(DEBUG_VIEWS):
                        debug_view = DEBUG_VIEWS[key - ord("0")]
                        logger.info("Debug view: %s", debug_view)
                    else:
                        autopilot_active = _handle_key(
                            key, autopilot_active, client, stuck_detector
                        )

            full_ms = (time.perf_counter() - t_iter) * 1000.0
            stats.record("full_loop", full_ms)
            stats.tick()
            frame_count += 1

            if recorder is not None:
                valid_disparity = None
                depth_coverage = None
                if result.depth is not None:
                    valid_disparity = result.depth.valid_disparity_fraction
                    depth_coverage = result.depth.coverage
                recorder.add(
                    now=time.perf_counter(),
                    primary_ms=primary_ms,
                    full_ms=full_ms,
                    confidence=_mean_confidence(result.perception.confidences),
                    mask=result.stabilized.mask,
                    plan=result.plan,
                    valid_disparity=valid_disparity,
                    depth_coverage=depth_coverage,
                    depth_used=result.depth is not None,
                )

            now = time.perf_counter()
            if now - t_last_log >= config.runtime_log_interval_s:
                _log_runtime(pipeline, display)
                t_last_log = now
            if args.benchmark_seconds > 0 and now - t_start >= args.benchmark_seconds:
                logger.info("Benchmark duration reached")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        pipeline.close()
        # The dashboard thread reads the display camera and draws into the
        # window, so it stops before either is torn down.
        if display is not None:
            display.stop()
        if dashboard_window is not None:
            dashboard_window.close()
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info("Session complete - %d frames in %.1f s", frame_count, elapsed)
        _log_runtime(pipeline, display)
        if recorder is not None:
            report = recorder.report(stats, pipeline.stereo_stats, pipeline.stereo_worker)
            out = args.benchmark_out or f"output/benchmarks/{recorder.label}.json"
            path = recorder.save(out, report)
            logger.info("Benchmark report written to %s", path)
            logger.info(
                "  %.1f FPS  primary %s/%s ms  stereo %s Hz  valid disparity %s%%  "
                "coverage %s%%  jitter %s  departures %d",
                report["main_fps"] or 0.0,
                report["primary_latency_mean_ms"],
                report["primary_latency_p95_ms"],
                report["stereo"].get("fps"),
                report["valid_disparity_pct_mean"],
                report["depth_coverage_pct_mean"],
                report["path_jitter_pct"],
                report["lane_departures"],
            )


if __name__ == "__main__":
    main()
