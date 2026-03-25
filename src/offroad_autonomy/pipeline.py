"""Pipeline orchestration for the offroad_autonomy stack.

``AutonomyPipeline`` owns references to every processing stage and
executes them in sequence: preprocess → perceive → stabilise → plan →
control.  The ``step`` method runs exactly one iteration of the loop
and returns the resulting ``ControlCommand``.
"""

from __future__ import annotations

import logging

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.types import ControlCommand, PipelineConfig, VehicleState

import numpy as np

logger = logging.getLogger("offroad_autonomy.pipeline")


class AutonomyPipeline:
    """Single-step orchestrator for the full autonomy stack."""

    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self.preprocessor = ImagePreprocessor(config)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config)
        self.controller = StanleyController(config)

    def step(
        self,
        raw_frame: np.ndarray,
        vehicle_state: VehicleState,
    ) -> ControlCommand:
        """Run one full cycle of the autonomy stack.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR uint8 camera frame from BeamNG.
        vehicle_state : VehicleState
            Current vehicle telemetry.

        Returns
        -------
        ControlCommand
            Steering, throttle, and brake values to send to the simulator.
        """
        frame = self.preprocessor.process(raw_frame)
        perception = self.segmenter.predict(frame)
        stabilized = self.stabilizer.stabilize(perception)
        plan = self.planner.plan(stabilized)
        command = self.controller.compute(plan, vehicle_state)

        logger.debug(
            "step: infer=%.0fms  stability=%.3f  kalman=%s  steer=%.3f",
            perception.inference_time_ms,
            stabilized.stability_score,
            plan.kalman_active,
            command.steering,
        )

        return command

    def reset(self) -> None:
        """Clear all temporal state (e.g. on map reload)."""
        self.stabilizer.reset()
        self.planner.reset()
