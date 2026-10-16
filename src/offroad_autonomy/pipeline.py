"""Stage orchestration: preprocess, segment, stabilise, plan and control."""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from offroad_autonomy.control.factory import build_controller
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import (
    CameraFrame,
    ControlCommand,
    PathPlan,
    PipelineConfig,
    PipelineStepResult,
    PlanningStepResult,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.pipeline")


class RGBPlanningPipeline:
    """RGB and calibration to a trajectory, with no actuation or simulator."""

    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self.stats = RuntimeStats()

        self.view = PerceptionView(config)
        self.valid_roi = self.view.valid_roi
        self.preprocessor = ImagePreprocessor(config, target_size=self.view.size)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config, camera=self.view.camera)
        self._max_frame_age_ms = config.max_frame_age_ms

    @property
    def ego_coverage(self) -> float:
        return self.view.ego_coverage

    def has_input(self, frames: CameraFrame | np.ndarray) -> bool:
        capture = self._as_capture(frames)
        return capture.image is not None

    def plan_result(
        self,
        frames: CameraFrame | np.ndarray,
        vehicle_state: VehicleState | None = None,
    ) -> PlanningStepResult:
        capture = self._as_capture(frames)
        timings: dict[str, float] = {}

        image = self.view.image(capture)
        if image is None:
            raise ValueError(
                f"No frame available for '{self.view.mode}' segmentation; "
                "check has_input() before stepping"
            )
        t1 = time.perf_counter()

        frame = self.preprocessor.process(image)
        frame.timestamp = capture.timestamp
        t2 = time.perf_counter()
        timings["preprocess"] = (t2 - t1) * 1000.0

        perception = self.segmenter.predict(frame, self.valid_roi)
        t3 = time.perf_counter()
        timings["segmentation"] = (t3 - t2) * 1000.0

        stabilized = self.stabilizer.stabilize(perception)
        t4 = time.perf_counter()
        timings["postprocess"] = (t4 - t3) * 1000.0

        plan = self.planner.plan(
            stabilized,
            vehicle_state=vehicle_state,
        )
        t5 = time.perf_counter()
        timings["planning"] = (t5 - t4) * 1000.0
        timings["perception_planning"] = (t5 - t1) * 1000.0
        frame_age_ms = None
        invalid_timestamp = not math.isfinite(capture.timestamp) or capture.timestamp < 0
        if capture.timestamp > 0 and math.isfinite(capture.timestamp):
            frame_age_ms = (t5 - capture.timestamp) * 1000.0
            timings["frame_age"] = frame_age_ms
            invalid_timestamp = frame_age_ms < 0 or (
                self._max_frame_age_ms > 0 and frame_age_ms > self._max_frame_age_ms
            )
        if invalid_timestamp:
            self.planner.reset()
            self.stabilizer.reset()
            plan = PathPlan(
                centerline=np.empty((0, 2), dtype=np.float32),
                fallback_active=True,
                fallback_reason="frame timestamp is stale or ahead of the monotonic clock",
                speed_scale=0.0,
                valid=False,
                status="stale",
            )
        self.stats.record_many(timings)
        return PlanningStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            capture=capture,
            timings_ms=timings,
            generated_timestamp=t5,
            frame_age_ms=frame_age_ms,
        )

    def _as_capture(self, frames: CameraFrame | np.ndarray) -> CameraFrame:
        """Bare frames are accepted so monocular tools can drive the pipeline."""
        if isinstance(frames, CameraFrame):
            return frames
        return CameraFrame(image=frames, timestamp=time.perf_counter())

    def reset(self) -> None:
        self.stabilizer.reset()
        self.planner.reset()


class AutonomyPipeline(RGBPlanningPipeline):
    """BeamNG compatibility wrapper around RGB perception and planning."""

    def __init__(self, config: PipelineConfig) -> None:
        super().__init__(config)
        if config.control_mode == "stanley":
            self.controller = StanleyController(config, camera=self.view.camera)
        else:
            self.controller = build_controller(
                config, camera=self.view.camera, grid_source=lambda: self.planner.grid
            )

    def step(self, frames: CameraFrame | np.ndarray, vehicle_state: VehicleState) -> ControlCommand:
        return self.step_result(frames, vehicle_state).command

    def step_result(
        self, frames: CameraFrame | np.ndarray, vehicle_state: VehicleState
    ) -> PipelineStepResult:
        result = self.plan_result(frames, vehicle_state)
        start = time.perf_counter()
        command = self.controller.compute(result.plan, vehicle_state)
        elapsed = (time.perf_counter() - start) * 1000.0
        result.timings_ms["control"] = elapsed
        self.stats.record("control", elapsed)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "step %d: seg=%.0fms road=%.1f%% stability=%.3f steer=%.3f",
                result.capture.frame_id,
                result.timings_ms["segmentation"],
                result.perception.road_fraction * 100.0,
                result.stabilized.stability_score,
                command.steering,
            )
        return PipelineStepResult(
            frame=result.frame,
            perception=result.perception,
            stabilized=result.stabilized,
            plan=result.plan,
            command=command,
            capture=result.capture,
            timings_ms=result.timings_ms,
            bev=self.planner.bev_snapshot(),
        )

    def reset(self) -> None:
        super().reset()
        self.controller.reset()
