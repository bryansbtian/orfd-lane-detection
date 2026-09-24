"""Stage orchestration.

Two loops run at their own rates::

    control loop (every frame, never waits on stereo)
        pair -> [stitch] -> preprocess -> segment -> fuse(latest depth)
             -> stabilise -> plan -> control

    stereo worker (background, rate-capped, latest frame only)
        pair -> rectify -> SGBM -> filter -> ROI gate -> reproject -> terrain
             -> publish latest StereoGeometry

Stereo is decoupled because SGBM costs more than the whole control budget.
When depth is off, not ready or stale the stack runs on segmentation alone,
so missing stereo costs accuracy, never control.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace

import numpy as np

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.fusion import build_traversability_map, fuse_rgb_depth
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.perception.stereo_depth import StereoDepthEstimator
from offroad_autonomy.perception.terrain_analyzer import TerrainAnalyzer
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.runtime.stereo_worker import StereoJob, StereoWorker
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import (
    ControlCommand,
    PipelineConfig,
    PipelineStepResult,
    StereoFramePair,
    StereoGeometry,
    TerrainAnalysis,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.pipeline")


class AutonomyPipeline:
    def __init__(self, config: PipelineConfig, start_worker: bool = True) -> None:
        logger.info("Initialising pipeline stages")
        self._config = config
        self.stats = RuntimeStats()

        self.view = PerceptionView(config)
        self.valid_roi = self.view.valid_roi
        self.preprocessor = ImagePreprocessor(config, target_size=self.view.size)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config, camera=self.view.camera)
        self.controller = StanleyController(config, camera=self.view.camera)

        self.depth_estimator: StereoDepthEstimator | None = None
        self.terrain_analyzer: TerrainAnalyzer | None = None
        self.stereo_worker: StereoWorker | None = None
        if config.depth_enabled:
            try:
                self.depth_estimator = StereoDepthEstimator(config, target=self.view.camera)
                self.terrain_analyzer = TerrainAnalyzer(config, camera=self.view.camera)
            except ValueError as exc:
                # A misconfigured rig must not take the whole stack down;
                # driving on appearance alone is still safe.
                logger.error("Stereo depth disabled - invalid rig: %s", exc)
                self.depth_estimator = None
                self.terrain_analyzer = None
        else:
            logger.info("Stereo depth disabled by configuration")

        if self.depth_estimator is not None:
            self.stereo_worker = StereoWorker(
                self._compute_geometry,
                rate_hz=config.stereo_rate_hz,
                asynchronous=config.stereo_async,
            )
            if start_worker:
                self.stereo_worker.start()

        self._max_depth_age_s = float(config.stereo_max_age_s)
        self._last_mask: np.ndarray | None = None

    @property
    def ego_coverage(self) -> float:
        return self.view.ego_coverage

    @property
    def stereo_stats(self) -> RuntimeStats | None:
        if self.stereo_worker is None:
            return None
        return self.stereo_worker.stats

    def has_input(self, frames: StereoFramePair | np.ndarray) -> bool:
        pair = self._as_pair(frames)
        if self.view.mode == "stitched":
            return pair.left is not None and pair.right is not None
        return pair.frame(self.view.mode) is not None

    def step(
        self,
        frames: StereoFramePair | np.ndarray,
        vehicle_state: VehicleState,
    ) -> ControlCommand:
        return self.step_result(frames, vehicle_state).command

    def step_result(
        self,
        frames: StereoFramePair | np.ndarray,
        vehicle_state: VehicleState,
    ) -> PipelineStepResult:
        pair = self._as_pair(frames)
        timings: dict[str, float] = {}

        # Submitted first so stereo overlaps with segmentation.
        self._submit_stereo(pair)

        t0 = time.perf_counter()
        image = self.view.image(pair)
        if image is None:
            raise ValueError(
                f"No frame available for '{self.view.mode}' segmentation; "
                "check has_input() before stepping"
            )
        t1 = time.perf_counter()
        if self.view.mode == "stitched":
            timings["stitching"] = (t1 - t0) * 1000.0

        frame = self.preprocessor.process(image)
        t2 = time.perf_counter()
        timings["preprocess"] = (t2 - t1) * 1000.0

        perception = self.segmenter.predict(frame, self.valid_roi)
        t3 = time.perf_counter()
        timings["segmentation"] = (t3 - t2) * 1000.0

        geometry, age_s, fallback = self._fresh_geometry(t3)
        depth = None
        terrain = None
        if geometry is not None:
            depth = geometry.depth
            terrain = geometry.terrain
        planner_terrain = self._age_corrected(terrain, vehicle_state.speed_mps, age_s)
        if terrain is not None:
            perception = fuse_rgb_depth(perception, terrain, self._config)
        t4 = time.perf_counter()
        timings["fusion"] = (t4 - t3) * 1000.0

        stabilized = self.stabilizer.stabilize(perception)
        self._last_mask = stabilized.mask
        t5 = time.perf_counter()
        timings["postprocess"] = (t5 - t4) * 1000.0

        plan = self.planner.plan(
            stabilized,
            vehicle_state=vehicle_state,
            terrain=planner_terrain,
        )
        t6 = time.perf_counter()
        timings["planning"] = (t6 - t5) * 1000.0

        command = self.controller.compute(plan, vehicle_state)
        t7 = time.perf_counter()
        timings["control"] = (t7 - t6) * 1000.0

        traversability = None
        if terrain is not None:
            clearance = math.inf
            if isinstance(planner_terrain, TerrainAnalysis):
                clearance = planner_terrain.min_forward_clearance_m
            analysis = None
            if isinstance(terrain, TerrainAnalysis):
                analysis = terrain
            traversability = build_traversability_map(
                stabilized.mask,
                analysis,
                focal_px=self.view.camera.focal_px,
                traversability=stabilized.traversability,
                forward_clearance_m=clearance,
                depth_age_s=age_s,
            )
            timings["fusion"] += (time.perf_counter() - t7) * 1000.0

        self.stats.record_many(timings)

        if logger.isEnabledFor(logging.DEBUG):
            depth_note = fallback
            if depth is not None:
                depth_note = f"{depth.coverage:.0%} ({age_s * 1000:.0f} ms old)"
            logger.debug(
                "step %d: seg=%.0fms road=%.1f%% depth=%s stability=%.3f steer=%.3f",
                pair.frame_id,
                timings["segmentation"],
                perception.road_fraction * 100.0,
                depth_note,
                stabilized.stability_score,
                command.steering,
            )

        return PipelineStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            command=command,
            frames=pair,
            depth=depth,
            terrain=planner_terrain,
            traversability=traversability,
            depth_age_s=age_s,
            depth_fallback=fallback,
            timings_ms=timings,
        )

    def _submit_stereo(self, pair: StereoFramePair) -> None:
        if self.stereo_worker is None:
            return
        if not pair.has_stereo:
            if pair.left is not None and pair.right is not None:
                logger.debug("Stereo skipped for frame %d: %s", pair.frame_id, pair.sync_note)
            return
        if not pair.is_new:
            return
        self.stereo_worker.submit(
            StereoJob(pair=pair, road_mask=self._last_mask, valid_roi=self.valid_roi)
        )

    def _compute_geometry(self, job: StereoJob) -> StereoGeometry | None:
        assert self.depth_estimator is not None and self.terrain_analyzer is not None
        pair = job.pair
        depth = self.depth_estimator.compute(
            pair.left,
            pair.right,
            road_mask=job.road_mask,
            valid_roi=job.valid_roi,
            frame_id=pair.frame_id,
            timestamp=pair.timestamp,
        )
        if depth is None:
            return None

        t0 = time.perf_counter()
        terrain = self.terrain_analyzer.analyze(depth, job.valid_roi)
        if isinstance(depth.timings_ms, dict):
            depth.timings_ms["terrain"] = (time.perf_counter() - t0) * 1000.0
        return StereoGeometry(
            depth=depth,
            terrain=terrain,
            frame_id=pair.frame_id,
            capture_time=pair.timestamp,
            completed_time=time.perf_counter(),
        )

    def _fresh_geometry(self, now: float) -> tuple[StereoGeometry | None, float, str]:
        if self.stereo_worker is None:
            return None, math.inf, "off"
        geometry = self.stereo_worker.latest()
        if geometry is None:
            return None, math.inf, "warming up"
        age = geometry.age_s(now)
        if age > self._max_depth_age_s:
            return None, age, "stale"
        return geometry, age, ""

    @staticmethod
    def _age_corrected(terrain, speed_mps: float, age_s: float):
        """Depth is a few frames old by the time it is used. The obstacle has
        not moved, the vehicle has, so the reported clearance is too generous
        by ``speed * age``. Correcting that one scalar is cheap and errs on the
        safe side; the per-pixel maps are left as measured.
        """
        if not isinstance(terrain, TerrainAnalysis):
            return terrain
        travelled = max(0.0, speed_mps) * max(0.0, age_s)
        if travelled <= 0.0 or not math.isfinite(terrain.min_forward_clearance_m):
            return terrain
        return replace(
            terrain,
            min_forward_clearance_m=max(0.0, terrain.min_forward_clearance_m - travelled),
        )

    def _as_pair(self, frames: StereoFramePair | np.ndarray) -> StereoFramePair:
        """Bare frames are accepted so monocular tools can drive the pipeline."""
        if isinstance(frames, StereoFramePair):
            return frames
        left = frames
        right = None
        if self.view.mode == "right":
            left = None
            right = frames
        return StereoFramePair(
            left=left,
            right=right,
            timestamp=time.perf_counter(),
            synchronized=False,
            sync_note="single frame",
        )

    def reset(self) -> None:
        self.stabilizer.reset()
        self.planner.reset()
        self.controller.reset()
        self._last_mask = None
        if self.stereo_worker is not None:
            self.stereo_worker.reset()

    def close(self) -> None:
        if self.stereo_worker is not None:
            self.stereo_worker.stop()
