"""Bird's-eye traversability grid, fused over time from the dashcam road mask.

The grid is centred on the point of ground under the camera, with rows
running forward (row 0 is the far edge) and columns running right. Each cell
holds the log-odds that it is road.

Why a grid instead of planning in image rows
--------------------------------------------
On the roof-mounted dashcam the hood hides the first ~3 m of ground and a
trail wider than ~11 m runs off the sides of the image. A planner that reads
image rows mistakes a row cut by the frame edge for the road's centre, and
forgets the ground under the hood that it saw a moment earlier. Here a cell
outside the view is *unknown*, never "not road", and the car's own motion
carries what was seen into the region the camera can no longer see.

Library choices: OpenCV ``remap`` projects the mask onto the ground (inverse
perspective mapping through the existing camera model), ``warpAffine`` moves
the grid with the vehicle, and ``distanceTransform`` measures clearance.
The projection assumes flat ground, so slopes and crests misplace far cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.types import VehicleState


@dataclass(frozen=True)
class GroundPose:
    """Estimated camera ground point and direction in the local odometry frame."""

    x: float
    y: float
    forward_x: float
    forward_y: float

    @property
    def left_x(self) -> float:
        # BeamNG's world is right-handed with +Z up, so left is forward turned 90 deg anticlockwise.
        return -self.forward_y

    @property
    def left_y(self) -> float:
        return self.forward_x


def ground_pose(state: VehicleState | None, camera: CameraModel) -> GroundPose | None:
    """``None`` when the state cannot register frames against each other;
    the grid then forgets instead of smearing old evidence in the wrong place."""
    if state is None or not state.valid or state.direction is None:
        return None
    fx, fy = float(state.direction[0]), float(state.direction[1])
    norm = math.hypot(fx, fy)
    if not math.isfinite(norm) or norm < 1e-6:
        return None
    fx, fy = fx / norm, fy / norm
    # Camera position is in vehicle space: +X left, -Y forward.
    cam_forward = -float(camera.position[1])
    cam_left = float(camera.position[0])
    x = float(state.position[0]) + cam_forward * fx - cam_left * fy
    y = float(state.position[1]) + cam_forward * fy + cam_left * fx
    return GroundPose(x=x, y=y, forward_x=fx, forward_y=fy)


class TraversabilityGrid:
    def __init__(
        self,
        config: GridPlannerConfig,
        camera: CameraModel,
        valid_roi: np.ndarray | None,
        roi_top: int,
    ) -> None:
        self.config = config
        self.cell_m = float(config.cell_m)
        self.rows = int(round((config.ahead_m + config.behind_m) / self.cell_m))
        self.cols = int(round(2.0 * config.half_width_m / self.cell_m))
        self._ahead_m = float(config.ahead_m)
        self._half_width_m = float(config.half_width_m)
        self._decay_s = float(config.memory_s)
        self.logodds = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._pose: GroundPose | None = None

        forward, right = self.cell_centres()
        pixels = camera.ground_to_image(forward.ravel(), right.ravel())
        u = pixels[:, 0].reshape(forward.shape)
        v = pixels[:, 1].reshape(forward.shape)
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        in_image = (
            (forward > 0.1) & (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)
        )
        # Rows above ``roi_top`` are sky, tree line or ground too far away to
        # place; the planner sets it from planning.grid.max_range_m.
        in_image &= vi >= roi_top
        if valid_roi is not None and valid_roi.shape == (camera.height, camera.width):
            in_image[in_image] &= valid_roi[vi[in_image], ui[in_image]]
        self.in_view = in_image
        self._map_x = np.where(in_image, u, -1.0).astype(np.float32)
        self._map_y = np.where(in_image, v, -1.0).astype(np.float32)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """``(forward_m, right_m)`` of every cell, each ``(rows, cols)``."""
        rows = np.arange(self.rows, dtype=np.float64)
        cols = np.arange(self.cols, dtype=np.float64)
        forward = self._ahead_m - (rows + 0.5) * self.cell_m
        right = (cols + 0.5) * self.cell_m - self._half_width_m
        return np.meshgrid(forward, right, indexing="ij")

    def to_cell(self, forward_m: np.ndarray, right_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integer ``(row, col)``; out-of-grid points fall outside ``[0, rows)``/``[0, cols)``."""
        row = np.floor((self._ahead_m - np.asarray(forward_m)) / self.cell_m).astype(int)
        col = np.floor((np.asarray(right_m) + self._half_width_m) / self.cell_m).astype(int)
        return row, col

    def reset(self) -> None:
        self.logodds.fill(0.0)
        self._pose = None

    def update(
        self,
        road_mask: np.ndarray | None,
        pose: GroundPose | None,
        dt_s: float,
    ) -> None:
        """Move the grid with the vehicle, let old evidence fade, then add
        this frame's mask. ``road_mask`` is ``None`` when the frame is not
        trusted, so the grid still tracks the vehicle but learns nothing."""
        self._advance(pose, dt_s)
        if road_mask is not None:
            self._observe(road_mask)

    def _advance(self, pose: GroundPose | None, dt_s: float) -> None:
        if pose is None or self._pose is None:
            self.logodds.fill(0.0)
        else:
            self._shift(self._pose, pose)
        self._pose = pose

        if dt_s > 0.0:
            self.logodds *= np.float32(math.exp(-dt_s / self._decay_s))

    def _observe(self, road_mask: np.ndarray) -> None:
        observed = cv2.remap(
            road_mask.astype(np.uint8),
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        cfg = self.config
        evidence = np.where(observed, cfg.road_logodds, cfg.offroad_logodds).astype(np.float32)
        self.logodds[self.in_view] += evidence[self.in_view]
        np.clip(self.logodds, -cfg.logodds_limit, cfg.logodds_limit, out=self.logodds)

    def road(self) -> np.ndarray:
        return self.logodds > self.config.road_threshold

    def blocked(self) -> np.ndarray:
        return self.logodds < self.config.blocked_threshold

    def clearance_m(self) -> np.ndarray:
        """Distance from every cell to the nearest cell known not to be road.
        Unknown cells do not count as obstacles, so the hood's blind zone
        does not close the corridor before memory has filled it."""
        free = (~self.blocked()).astype(np.uint8)
        distance = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return distance * np.float32(self.cell_m)

    def _shift(self, previous: GroundPose, current: GroundPose) -> None:
        # Affine map from a current cell to where the same ground sat in the
        # previous grid, so warpAffine can pull the old evidence across.
        def metric_to_world(pose: GroundPose) -> np.ndarray:
            # Columns: forward_m, right_m, 1 -> world x, y.
            return np.array(
                [
                    [pose.forward_x, -pose.left_x, pose.x],
                    [pose.forward_y, -pose.left_y, pose.y],
                    [0.0, 0.0, 1.0],
                ]
            )

        def world_to_metric(pose: GroundPose) -> np.ndarray:
            rotation = np.array(
                [
                    [pose.forward_x, pose.forward_y],
                    [-pose.left_x, -pose.left_y],
                ]
            )
            out = np.eye(3)
            out[:2, :2] = rotation
            out[:2, 2] = -rotation @ np.array([pose.x, pose.y])
            return out

        # Cell (col, row) <-> metric (forward, right).
        cell_to_metric = np.array(
            [
                [0.0, -self.cell_m, self._ahead_m - 0.5 * self.cell_m],
                [self.cell_m, 0.0, 0.5 * self.cell_m - self._half_width_m],
                [0.0, 0.0, 1.0],
            ]
        )
        metric_to_cell = np.linalg.inv(cell_to_metric)
        current_to_previous = (
            metric_to_cell @ world_to_metric(previous) @ metric_to_world(current) @ cell_to_metric
        )
        self.logodds = cv2.warpAffine(
            self.logodds,
            current_to_previous[:2].astype(np.float64),
            (self.cols, self.rows),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
