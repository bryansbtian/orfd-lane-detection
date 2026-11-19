"""Bird's-eye grid planner: gated road evidence fused over time on the
ground, then a fan of steerable arcs scored on it. Selected by
``experiment.planner: grid``."""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from experiment.config import ExperimentConfig
from experiment.grid.arc_planner import ArcPlanner
from experiment.grid.bev_grid import TraversabilityGrid, ground_pose
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.planning.perception_gate import GateDecision
from offroad_autonomy.types import BEVSnapshot, PathPlan, StabilizedResult, VehicleState

logger = logging.getLogger("experiment.grid")


class GridPlanner(CenterlinePlanner):
    def __init__(self, config: ExperimentConfig, camera: CameraModel | None = None) -> None:
        super().__init__(config, camera=camera)
        self._grid_config = config.grid
        # Built on the first frame, once the mask size and ego exclusion are known.
        self._grid: TraversabilityGrid | None = None
        self._grid_time: float | None = None
        self._grid_path = None
        # Heading at the first known pose; the BEV view keeps it pointing up.
        self._heading_origin: float | None = None
        self._heading: float | None = None
        self._direction_warned = False
        self._arcs = ArcPlanner(
            config.grid,
            wheelbase_m=config.wheelbase_m,
            max_wheel_angle_deg=config.max_wheel_angle_deg,
            vehicle_half_width_m=config.vehicle_half_width_m,
            rear_axle_behind_camera_m=config.camera_ahead_of_rear_axle_m,
        )
        logger.info("Planning mode: grid")

    def plan(
        self,
        stabilized: StabilizedResult,
        vehicle_state: VehicleState | None = None,
    ) -> PathPlan:
        return self._plan_grid(stabilized, self._gate_decision(stabilized), vehicle_state)

    def _plan_grid(
        self,
        stabilized: StabilizedResult,
        decision: GateDecision,
        vehicle_state: VehicleState | None,
    ) -> PathPlan:
        """The grid keeps tracking the vehicle on every frame, but only a
        frame that passed the gate adds evidence: a rejected mask must not
        teach the grid road that perception itself did not trust."""
        self._grid_path = None
        if self._grid is None:
            self._grid = TraversabilityGrid(
                self._grid_config,
                self._camera,
                stabilized.valid_roi,
                self._grid_top_row(),
            )
        now = time.perf_counter()
        dt = 0.0
        if self._grid_time is not None:
            dt = now - self._grid_time
        self._grid_time = now

        self._check_direction(vehicle_state)
        observation = None
        if decision.ok:
            # Only the trusted road connected to the gate's chosen corridor
            # may teach the grid. Keep its continuation beyond the gate ROI.
            observation = self._gate.connected_evidence(stabilized, decision.component)
        pose = ground_pose(vehicle_state, self._camera)
        self._track_heading(pose)
        self._grid.update(observation, pose, dt)

        if not decision.ok:
            return self._hold(decision)

        choice = self._arcs.choose(self._grid, pose)
        if choice is None:
            decision.reason = "no drivable arc in the grid"
            return self._hold(decision)

        self._grid_path = np.column_stack((choice.forward_m, choice.right_m)).copy()
        curvature = choice.curvature_profile
        if curvature is None:
            curvature = np.full(len(choice.forward_m), choice.curvature)
        # Pixels far -> near, the order Stanley expects from every planner.
        centerline = self._camera.ground_to_image(choice.forward_m, choice.right_m)[::-1].copy()
        plan = PathPlan(
            centerline=centerline,
            heading_rad=self._estimate_heading(centerline),
            curvature=self._estimate_curvature(centerline),
            road_width_px=self._estimate_road_width(decision.component, centerline),
        )
        plan.planner_mask = decision.component
        plan.roi_top = decision.roi_top
        plan.ground_path = self._grid_path.copy()
        plan.ground_curvature = np.asarray(curvature, dtype=np.float64).copy()
        plan.status = "grid_estimate"
        self._last_good = plan
        self._held = 0
        return plan

    def _grid_top_row(self) -> int:
        """Highest image row the grid reads: where flat ground ``max_range_m``
        ahead appears. The gate's ROI stops nearer (~15 m); past ~25 m one
        row spans metres of ground and a small pitch moves it by metres more."""
        far = self._camera.ground_to_image(
            np.array([self._grid_config.max_range_m]), np.array([0.0])
        )[0][1]
        return max(0, int(math.ceil(float(far))))

    def _track_heading(self, pose) -> None:
        if pose is None:
            self._heading = None
            return
        heading = math.atan2(pose.forward_y, pose.forward_x)
        if self._heading_origin is None:
            self._heading_origin = heading
        self._heading = math.remainder(heading - self._heading_origin, 2.0 * math.pi)

    def bev_snapshot(self) -> BEVSnapshot | None:
        """Copy grid evidence so rendering cannot race the next planner update."""
        grid = self._grid
        if grid is None:
            return None
        arcs = self._arcs
        path = None
        if self._grid_path is not None:
            path = self._grid_path.copy()
        return BEVSnapshot(
            road=grid.road().copy(),
            blocked=grid.blocked().copy(),
            visible=grid.in_view.copy(),
            cell_m=grid.cell_m,
            ahead_m=self._grid_config.ahead_m,
            half_width_m=self._grid_config.half_width_m,
            path=path,
            candidates=arcs.preview_arcs(),
            branch_count=arcs.branch_count,
            branch_locked=arcs.branch_locked,
            heading_rad=self._heading,
            confidence=np.clip(
                np.abs(grid.logodds) / self._grid_config.logodds_limit, 0.0, 1.0
            ).astype(np.float32),
        )

    def _check_direction(self, vehicle_state: VehicleState | None) -> None:
        """Driving forward, estimated velocity and the local direction vector must
        agree; if they do not, the grid would shift its memory the wrong way."""
        if self._direction_warned or vehicle_state is None or vehicle_state.direction is None:
            return
        if vehicle_state.speed_mps < 2.0:
            return
        vx, vy = vehicle_state.velocity[0], vehicle_state.velocity[1]
        dx, dy = vehicle_state.direction[0], vehicle_state.direction[1]
        if vx * dx + vy * dy < 0.0:
            logger.warning(
                "Vehicle direction points against its velocity; grid memory may be mirrored"
            )
            self._direction_warned = True

    @property
    def grid(self) -> TraversabilityGrid | None:
        return self._grid

    def reset(self) -> None:
        super().reset()
        if self._grid is not None:
            self._grid.reset()
        self._grid_time = None
        self._heading_origin = None
        self._heading = None
        self._arcs.reset()
