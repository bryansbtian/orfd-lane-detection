"""Stage orchestration: preprocess, segment, stabilise, plan and control."""

from __future__ import annotations

import logging
import time

import numpy as np

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
    PipelineConfig,
    PipelineStepResult,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.pipeline")


class AutonomyPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self.stats = RuntimeStats()

        self.view = PerceptionView(config)
        self.valid_roi = self.view.valid_roi
        self.preprocessor = ImagePreprocessor(config, target_size=self.view.size)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config, camera=self.view.camera)
        self.controller = StanleyController(config, camera=self.view.camera)

    @property
    def ego_coverage(self) -> float:
        return self.view.ego_coverage

    def has_input(self, frames: CameraFrame | np.ndarray) -> bool:
        capture = self._as_capture(frames)
        return capture.image is not None

    def step(
        self,
        frames: CameraFrame | np.ndarray,
        vehicle_state: VehicleState,
    ) -> ControlCommand:
        return self.step_result(frames, vehicle_state).command

    def step_result(
        self,
        frames: CameraFrame | np.ndarray,
        vehicle_state: VehicleState,
    ) -> PipelineStepResult:
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

        command = self.controller.compute(plan, vehicle_state)
        t6 = time.perf_counter()
        timings["control"] = (t6 - t5) * 1000.0

        self.stats.record_many(timings)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "step %d: seg=%.0fms road=%.1f%% stability=%.3f steer=%.3f",
                capture.frame_id,
                timings["segmentation"],
                perception.road_fraction * 100.0,
                stabilized.stability_score,
                command.steering,
            )

        return PipelineStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            command=command,
            capture=capture,
            timings_ms=timings,
        )

    def _as_capture(self, frames: CameraFrame | np.ndarray) -> CameraFrame:
        """Bare frames are accepted so monocular tools can drive the pipeline."""
        if isinstance(frames, CameraFrame):
            return frames
        return CameraFrame(image=frames, timestamp=time.perf_counter())

    def reset(self) -> None:
        self.stabilizer.reset()
        self.planner.reset()
        self.controller.reset()
