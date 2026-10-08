"""Entry point: connect to BeamNG and run the autonomy loop until stopped.

The dashboard runs on its own thread, so this loop never waits for it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.runtime.benchmark import BenchmarkRecorder
from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker
from offroad_autonomy.runtime.timing import DISPLAY_STAGES, MAIN_STAGES
from offroad_autonomy.runtime.video_recorder import VideoRecorder, default_video_path
from offroad_autonomy.simulation.beamng_client import BeamNGClient, BeamNGConnectionError
from offroad_autonomy.types import (
    DEBUG_VIEW_KEYS,
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
)
from offroad_autonomy.visualization.presentation import (
    PRESENTATION_HEIGHT,
    PRESENTATION_WIDTH,
    PresentationRenderer,
)

logger = logging.getLogger("offroad_autonomy.main")

_shutdown = False
_WINDOW_TITLE = "Off-Road Autonomy Dashboard"
_DASHBOARD_WIDTH = 1600
_DASHBOARD_HEIGHT = 900
# Apart from output/videos, so presentations are not lost among the many
# dashboard recordings made while tuning.
_PRESENTATION_DIR = "output/presentations"
_MPH_PER_MPS = 2.2369362920544

_STUCK_SPEED_MPS = 0.4
_STUCK_THROTTLE_MIN = 0.15
_STUCK_TIME_S = 3.0

_MANUAL_STEER = 0.4
_MANUAL_THROTTLE = 0.25
_MANUAL_BRAKE = 0.35

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
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Record the dashboard to an H.264 mp4 (needs ffmpeg on PATH). "
            "With --headless the dashboard is drawn off screen for the recording only."
        ),
    )
    parser.add_argument(
        "--record-video-out",
        default="",
        help="Video path (default: output/videos/<label>_<timestamp>.mp4).",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Record a 1920x1080 presentation video (orbit camera, dashcam overlay, key stats) "
            "to its own mp4 (needs ffmpeg on PATH). Combine with --record-video to also "
            "record the dashboard, to a separate file."
        ),
    )
    parser.add_argument(
        "--presentation-view",
        action="store_true",
        help=(
            "Show the orbit camera, dashcam overlay and stats in the live window. "
            "Add --presentation to also record this view."
        ),
    )
    parser.add_argument(
        "--presentation-out",
        default="",
        help=f"Presentation path (default: {_PRESENTATION_DIR}/<label>_<timestamp>.mp4).",
    )
    return parser


def _open_video(path: str | Path, width: int, height: int, config: PipelineConfig) -> VideoRecorder:
    try:
        return VideoRecorder(
            path,
            width,
            height,
            fps=config.recording_fps,
            crf=config.recording_crf,
            preset=config.recording_preset,
            queue_frames=config.recording_queue_frames,
        )
    except (RuntimeError, OSError) as exc:
        sys.exit(f"Cannot record video: {exc}")


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
    primary = main_stats.stage("primary_loop")
    kalman = bool(plan is not None and plan.kalman_active)

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
    fallback_state = "NONE"
    if fallback:
        fallback_state = " + ".join(fallback)

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
        fallback_reason=result.plan.fallback_reason,
        dashboard_fps=display.fps(),
    )

    if timing_overlay:
        # One heading per thread, so a slow dashboard can never be mistaken
        # for a slow vehicle.
        lines = ["AUTONOMY LOOP"] + main_stats.format_lines(MAIN_STAGES)
        lines += ["", "DASHBOARD"] + display.stats.format_lines(DISPLAY_STAGES)
        telemetry.timing_lines = lines
    return telemetry


def _log_runtime(
    pipeline: AutonomyPipeline,
    display: DisplayWorker | None = None,
    presentation: DisplayWorker | None = None,
) -> None:
    primary = pipeline.stats.stage("primary_loop")
    message = (
        f"autonomy {pipeline.stats.fps():.1f} FPS  primary {primary.mean_ms:.1f}/"
        f"{primary.p95_ms:.1f} ms (mean/p95)"
    )
    if display is not None:
        total = display.stats.stage("dashboard_total")
        message += (
            f" | dashboard {display.fps():.1f} Hz  {total.mean_ms:.1f} ms"
            f"  (drawn {display.rendered}, superseded {display.dropped})"
        )
    if presentation is not None:
        render = presentation.stats.stage("dashboard_render")
        message += (
            f" | presentation {presentation.fps():.1f} Hz  render {render.mean_ms:.1f}/"
            f"{render.p95_ms:.1f} ms (drawn {presentation.rendered}, "
            f"superseded {presentation.dropped})"
        )
    logger.info(message)
    for line in pipeline.stats.format_lines(MAIN_STAGES):
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


def video_paths(args: argparse.Namespace, label: str) -> tuple[Path, Path]:
    """Where the dashboard and the presentation recordings go.

    Two encoders writing one file would corrupt it, so a run that asks for
    both at the same path is stopped before anything is recorded.
    """
    dashboard = Path(args.record_video_out or default_video_path(label))
    presentation = Path(args.presentation_out or default_video_path(label, _PRESENTATION_DIR))
    if args.record_video and args.presentation and dashboard.resolve() == presentation.resolve():
        sys.exit("--record-video-out and --presentation-out must name different files")
    return dashboard, presentation


def _log_video(video: VideoRecorder) -> None:
    if video.failed:
        logger.error("Video recording failed; %s may be incomplete", video.path)
        return
    logger.info(
        "Video saved to %s: %d frames (%d repeated to keep real time, %d dropped by a busy encoder)",
        video.path,
        video.frames_written,
        video.frames_repeated,
        video.frames_dropped,
    )


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

    show_presentation = args.presentation_view and not config.ui_headless
    use_presentation = args.presentation or show_presentation
    client = BeamNGClient(config, orbit=use_presentation)
    pipeline = AutonomyPipeline(config)
    dashboard_window: DashboardWindow | None = None
    display: DisplayWorker | None = None
    presentation_display: DisplayWorker | None = None
    video: VideoRecorder | None = None
    presentation_video: VideoRecorder | None = None
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

    # Opened before connecting so a missing ffmpeg stops the run before a map
    # is loaded and a vehicle spawned for nothing.
    label = args.label or config_path.stem
    video_path, presentation_path = video_paths(args, label)
    if args.record_video:
        video = _open_video(video_path, _DASHBOARD_WIDTH, _DASHBOARD_HEIGHT, config)
        logger.info("Recording the dashboard to %s", video.path)
    if args.presentation:
        presentation_video = _open_video(
            presentation_path, PRESENTATION_WIDTH, PRESENTATION_HEIGHT, config
        )
        logger.info("Recording the presentation video to %s", presentation_video.path)

    try:
        client.connect()

        record_dashboard = video is not None
        record = None
        if record_dashboard:
            record = video.write

        show_dashboard = not config.ui_headless and not show_presentation
        if show_dashboard or record_dashboard:
            dashboard = AutonomyDashboard(
                width=_DASHBOARD_WIDTH,
                height=_DASHBOARD_HEIGHT,
                colors=config.dashboard_colors,
                sensor=config.camera.sensor,
                thresholds=config.dashboard_thresholds,
            )
            if show_dashboard:
                dashboard_window = _open_window(dashboard.width, dashboard.height)

            def _render(state: DisplayState) -> np.ndarray:
                return dashboard.render(
                    state.result,
                    state.telemetry,
                    plan=state.plan,
                    valid_roi=state.valid_roi,
                    debug_view=state.debug_view,
                    timing_overlay=state.timing_overlay,
                )

        if dashboard_window is not None:
            window = dashboard_window
            # Tkinter's event loop belongs to the thread that built the root
            # window, so only the OpenCV backend can be driven off-thread.
            display = DisplayWorker(
                render=_render,
                show=window.show,
                read_key=lambda: window.last_key,
                record=record,
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
                "Keys: E safe stop, P resume, 0/1/6/9 debug view (%s), T timing overlay",
                " ".join(f"{i}={name}" for i, name in DEBUG_VIEW_KEYS.items()),
            )
        elif record_dashboard:
            # No window to keep on its own thread, so the off-screen dashboard
            # always renders asynchronously, whatever ui.display_async says.
            display = DisplayWorker(
                render=_render,
                show=lambda canvas: True,
                read_key=lambda: -1,
                record=record,
                rate_hz=config.ui_display_rate_hz,
                asynchronous=True,
            )
            display.start()
            logger.info("No window: dashboard drawn off screen for the recording only")
        elif not show_presentation:
            logger.info("Headless: no dashboard; manual control is unavailable")

        if use_presentation:
            presentation = PresentationRenderer(
                colors=config.dashboard_colors,
                sensor=config.camera.sensor,
                thresholds=config.dashboard_thresholds,
            )

            def _render_presentation(state: DisplayState) -> np.ndarray:
                return presentation.render(
                    state.result,
                    state.telemetry,
                    state.orbit,
                    state.plan,
                    state.valid_roi,
                )

            presentation_window = None
            if show_presentation:
                presentation_window = _open_window(PRESENTATION_WIDTH, PRESENTATION_HEIGHT)
                dashboard_window = presentation_window
            # Record-only rendering runs off-thread. A live Tkinter window
            # must render on the main thread, just like the operator dashboard.
            presentation_display = DisplayWorker(
                render=_render_presentation,
                show=presentation_window.show if presentation_window else lambda canvas: True,
                read_key=lambda: presentation_window.last_key if presentation_window else -1,
                record=presentation_video.write if presentation_video else None,
                rate_hz=config.ui_display_rate_hz,
                asynchronous=(
                    not presentation_window
                    or (config.ui_display_async and presentation_window.backend == "opencv")
                ),
            )
            presentation_display.start()
            if presentation_window:
                logger.info("Live presentation view: E safe stop, P resume, Q quit")
            elif presentation_video:
                logger.info("Presentation video drawn off screen for the recording only")

        logger.info(
            "Perception on the '%s' camera; ego-vehicle exclusion %.1f%% of that view",
            pipeline.view.mode,
            pipeline.ego_coverage * 100.0,
        )
        logger.info("Entering main loop - Ctrl+C or SIGTERM to stop")
        t_start = time.perf_counter()
        orbit_image: np.ndarray | None = None

        while not _shutdown:
            t_iter = time.perf_counter()
            with stats.time("capture"):
                capture = client.capture_frame()
            if capture is None or not pipeline.has_input(capture):
                time.sleep(0.005)
                continue

            # Polled here, not on the dashboard thread, because the socket
            # transport is not thread safe. A missed read keeps the last
            # image so the video does not flash the placeholder.
            if use_presentation:
                with stats.time("orbit_capture"):
                    latest_orbit = client.capture_orbit()
                if latest_orbit is not None:
                    orbit_image = latest_orbit

            with stats.time("vehicle_state"):
                state = client.get_vehicle_state()

            # Perception keeps running under safe stop so the operator can see
            # the road come back before handing control over again.
            result = pipeline.step_result(capture, state)
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
                    command = _SAFE_STOP_COMMAND
                    plan = None
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
            workers = [worker for worker in (display, presentation_display) if worker is not None]
            if workers and frame_count % render_every == 0:
                telemetry = _build_dashboard_telemetry(
                    state,
                    command,
                    result,
                    plan,
                    pipeline,
                    autopilot_active,
                    timing_overlay,
                    workers[0],
                )
                # One snapshot for both: each worker only reads it.
                snapshot = DisplayState(
                    result=result,
                    telemetry=telemetry,
                    plan=plan,
                    valid_roi=pipeline.valid_roi,
                    debug_view=debug_view,
                    timing_overlay=timing_overlay,
                    orbit=orbit_image,
                )
                for worker in workers:
                    worker.publish(snapshot)

            active_display = presentation_display if show_presentation else display
            if active_display is not None:
                if active_display.closed:
                    _shutdown = True
                for key in active_display.drain_keys():
                    if key in (ord("t"), ord("T")):
                        timing_overlay = not timing_overlay
                    elif ord("0") <= key <= ord("9") and key - ord("0") in DEBUG_VIEW_KEYS:
                        debug_view = DEBUG_VIEW_KEYS[key - ord("0")]
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
                recorder.add(
                    now=time.perf_counter(),
                    primary_ms=primary_ms,
                    full_ms=full_ms,
                    confidence=_mean_confidence(result.perception.confidences),
                    mask=result.stabilized.mask,
                    plan=result.plan,
                )

            now = time.perf_counter()
            if now - t_last_log >= config.runtime_log_interval_s:
                _log_runtime(pipeline, display, presentation_display)
                t_last_log = now
            if args.benchmark_seconds > 0 and now - t_start >= args.benchmark_seconds:
                logger.info("Benchmark duration reached")
                break

    except BeamNGConnectionError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop drawing before the window is torn down and the video is closed.
        if display is not None:
            display.stop()
        if presentation_display is not None:
            presentation_display.stop()
        for recording in (video, presentation_video):
            if recording is not None:
                recording.close()
                _log_video(recording)
        if dashboard_window is not None:
            dashboard_window.close()
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info("Session ended - %d frames in %.1f s", frame_count, elapsed)
        _log_runtime(pipeline, display, presentation_display)
        if recorder is not None:
            report = recorder.report(stats)
            out = args.benchmark_out or f"output/benchmarks/{recorder.label}.json"
            path = recorder.save(out, report)
            logger.info("Benchmark report written to %s", path)
            logger.info(
                "  %.1f FPS  primary %s/%s ms  jitter %s  departures %d",
                report["main_fps"] or 0.0,
                report["primary_latency_mean_ms"],
                report["primary_latency_p95_ms"],
                report["path_jitter_pct"],
                report["lane_departures"],
            )


if __name__ == "__main__":
    main()
