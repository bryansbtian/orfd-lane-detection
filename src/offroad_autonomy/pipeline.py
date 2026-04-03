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
from offroad_autonomy.types import (
    ControlCommand,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    VehicleState,
    FramePacket,
)

import cv2
import numpy as np

logger = logging.getLogger("offroad_autonomy.pipeline")


class AutonomyPipeline:
    """Single-step orchestrator for the full autonomy stack."""

    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self._config = config
        self.preprocessor = ImagePreprocessor(config)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config)
        self.controller = StanleyController(config)
        self._annotation_colors_logged = False
        self._road_color = np.array(config.annotation_road_color, dtype=np.int16)
        self._annotation_map: dict = {}
        if config.debug_mode != "normal":
            logger.info("Debug mode: %s", config.debug_mode)

    def set_annotation_map(self, color_map: dict) -> None:
        """Provide the full get_annotations() map for class reverse-lookup."""
        self._annotation_map = color_map

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
        return self.step_result(raw_frame, vehicle_state).command

    def step_result(
        self,
        raw_frame: np.ndarray,
        vehicle_state: VehicleState,
        annotation_frame: np.ndarray | None = None,
        road_centerline: np.ndarray | None = None,
        road_mask_gt: np.ndarray | None = None,
    ) -> PipelineStepResult:
        """Run one full cycle of the autonomy stack and retain stage outputs."""
        frame = self.preprocessor.process(raw_frame)
        h, w = frame.preprocessed.shape[:2]

        mode = self._config.debug_mode

        # Resolve GT mask: road_mask_gt (from road edges) takes priority over annotation camera
        gt_mask = None
        if road_mask_gt is not None:
            gt_mask = road_mask_gt
        elif annotation_frame is not None and mode in ("gt_mask", "gt_centerline"):
            gt_mask = self._mask_from_annotation(annotation_frame, frame)

        if mode == "gt_centerline" and road_centerline is not None and len(road_centerline) >= 2:
            # Level 2: GT road centerline from road edges → controller only
            blank = np.zeros((h, w), dtype=bool)
            perception = PerceptionResult(mask=blank, confidences=[1.0], num_detections=1)
            stabilized = self.stabilizer.stabilize(perception)
            plan = self._plan_from_centerline_pts(road_centerline)
        elif gt_mask is not None and mode == "gt_mask":
            # Level 1: GT mask → normal planner → controller
            perception = PerceptionResult(mask=gt_mask, confidences=[1.0], num_detections=1)
            stabilized = self.stabilizer.stabilize(perception)
            plan = self.planner.plan(stabilized, vehicle_state=vehicle_state)
        else:
            perception = self.segmenter.predict(frame)
            stabilized = self.stabilizer.stabilize(perception)
            plan = self.planner.plan(stabilized, vehicle_state=vehicle_state)

        command = self.controller.compute(plan, vehicle_state)

        logger.debug(
            "step: infer=%.0fms  stability=%.3f  kalman=%s  steer=%.3f",
            perception.inference_time_ms,
            stabilized.stability_score,
            plan.kalman_active,
            command.steering,
        )

        return PipelineStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            command=command,
        )

    def _mask_from_annotation(self, annotation_frame: np.ndarray, frame: "FramePacket") -> np.ndarray:
        """Threshold annotation frame by road color → bool mask.

        On the first call, auto-detects the road color by sampling the
        bottom-center of the annotation frame (car is always on road there),
        then reverse-looks it up in the annotation class map if available.
        """
        ann = cv2.resize(
            annotation_frame,
            (frame.preprocessed.shape[1], frame.preprocessed.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("Annotation Camera", ann)
        cv2.waitKey(1)

        if not self._annotation_colors_logged:
            self._annotation_colors_logged = True
            h_ann, w_ann = ann.shape[:2]
            # Sample bottom-center patch — car is always on road here
            patch = ann[int(h_ann * 0.80):, int(w_ann * 0.35):int(w_ann * 0.65)].reshape(-1, 3)
            pu, pc = np.unique(patch, axis=0, return_counts=True)
            detected_bgr = pu[np.argmax(pc)].astype(np.int16)
            self._road_color = detected_bgr

            # Reverse-lookup in annotation map to name the class
            if self._annotation_map:
                matched = None
                for cls, rgb in self._annotation_map.items():
                    b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
                    if np.abs(np.array([b, g, r], dtype=np.int16) - detected_bgr).max() <= 5:
                        matched = cls
                        break
                logger.info(
                    "Road color auto-detected: BGR=(%d,%d,%d)  class=%s",
                    *detected_bgr, matched or "unknown",
                )
            else:
                logger.info("Road color auto-detected: BGR=(%d,%d,%d)", *detected_bgr)

        diff = np.abs(ann.astype(np.int16) - self._road_color)
        mask = (diff.max(axis=2) <= 15).astype(bool)
        logger.debug("Annotation mask: %d road pixels", int(mask.sum()))
        return mask

    def _centerline_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """Scan rows for left/right road edges; midpoints give centerline (pts[-1]=nearest)."""
        import math
        h, w = mask.shape[:2]
        row_indices = np.linspace(int(h * 0.10), int(h * 0.95), self._config.centerline_samples, dtype=int)
        points = []
        for y in row_indices:
            cols = np.where(mask[y])[0]
            if len(cols) >= 2:
                points.append((float(cols[0] + cols[-1]) / 2.0, float(y)))
        if len(points) < 2:
            return np.empty((0, 2))
        return np.array(points, dtype=np.float32)

    def _plan_from_centerline_pts(self, points: np.ndarray) -> PathPlan:
        """Build a PathPlan from centerline (x, y) points where pts[-1] is nearest."""
        import math
        pt_near = points[-1]
        pt_far = points[max(0, len(points) - 1 - max(1, len(points) // 3))]
        dx = pt_far[0] - pt_near[0]
        dy = pt_near[1] - pt_far[1]
        heading = float(math.atan2(dx, dy)) if dy > 1e-6 else 0.0
        return PathPlan(centerline=points, heading_rad=heading, kalman_active=False)

    def reset(self) -> None:
        """Clear all temporal state (e.g. on map reload)."""
        self.stabilizer.reset()
        self.planner.reset()
        self.controller.reset()
