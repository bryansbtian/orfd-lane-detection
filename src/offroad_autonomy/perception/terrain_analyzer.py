"""Turn a depth map into the geometry the planner actually needs.

Appearance tells you where the trail *looks* drivable; geometry tells you
whether a vehicle fits. This stage answers the geometric half:

* **Ground plane** - a robust local fit through the near-field points, so
  height is measured against the terrain the vehicle is on rather than
  against an absolute datum. That matters on a slope, where every point
  would otherwise read as "high".
* **Height above ground** - positive obstacles (rocks, berms, stumps) and
  negative ones (ruts, washouts, drop-offs).
* **Slope** - surface normals from the point cloud, so a smooth but steep
  bank is rejected even though it is obstacle-free.
* **Metric clearance** - a bird's-eye occupancy grid distance-transformed in
  metres, which is the quantity that decides whether a gap is wide enough.

Everything image-shaped comes back on the segmentation view's grid, aligned
with the mask so the two can be fused pixel for pixel. Fusion itself lives in
``perception.fusion``.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import (
    DepthResult,
    PipelineConfig,
    TerrainAnalysis,
)

logger = logging.getLogger("offroad_autonomy.perception.terrain")


class TerrainAnalyzer:
    _SLOPE_KERNEL = (5, 5)
    #: A normal differenced across holes is noise, so nearly the whole
    #: stencil must be measured.
    _SLOPE_MIN_SUPPORT = 0.9

    def __init__(self, config: PipelineConfig, camera: CameraModel | None = None) -> None:
        self._obstacle_h = float(config.obstacle_height_m)
        self._drop_h = float(config.drop_height_m)
        self._max_slope = math.radians(float(config.max_slope_deg))
        self._max_point_h = float(config.max_point_height_m)
        self._min_point_h = float(config.min_point_height_m)
        self._fit_min_points = int(config.ground_fit_min_points)
        self._fit_near = float(config.ground_fit_near_m)
        self._fit_far = float(config.ground_fit_far_m)
        self._fit_lateral = float(config.ground_fit_lateral_m)

        self._bev_forward = float(config.bev_forward_m)
        self._bev_lateral = float(config.bev_lateral_m)
        self._bev_cell = max(0.05, float(config.bev_cell_m))
        self._bev_min_points = max(1, int(config.bev_min_points_per_cell))
        self._bev_min_ratio = float(np.clip(config.bev_min_obstacle_ratio, 0.0, 1.0))
        self._half_width = float(config.vehicle_half_width_m)
        self._lookahead = float(config.clearance_lookahead_m)

        self._bev_rows = max(1, int(round(self._bev_forward / self._bev_cell)))
        self._bev_cols = max(1, int(round(2.0 * self._bev_lateral / self._bev_cell)))

        self._fill_enabled = bool(config.ground_fill_enabled)
        self._fill_max_m = float(config.ground_fill_max_m)
        self._inferred_support = float(np.clip(config.depth_inferred_support, 0.0, 1.0))

        # Rays are fixed per pixel, so they are built once for the near-field
        # fill. ``camera`` must be the view depth was reprojected into.
        if camera is None:
            camera = CameraModel(
                config.segmentation_camera, config.preprocess_width, config.preprocess_height
            )
        self._camera = camera
        self._ray_origin = self._camera.position.astype(np.float32)
        self._ray_dirs = self._build_ray_directions(self._camera)

        # Ground plane persists between frames so a sparse frame does not
        # throw the height reference away.
        self._plane = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._has_plane = False

    def analyze(
        self,
        depth: DepthResult,
        valid_roi: np.ndarray | None = None,
    ) -> TerrainAnalysis:
        """``valid_roi`` limits only the image-space products. The point cloud
        is left whole: every point in it was genuinely triangulated, and the
        bird's-eye grid is indexed in metres, not pixels.
        """
        t0 = time.perf_counter()

        points = depth.points_vehicle.copy()

        plane = self._fit_ground_plane(depth.cloud_vehicle)

        # Discard triangulation blunders before anything reasons about them.
        measured_height = self._height_above_plane(points, depth.valid, plane)
        measured = (
            depth.valid
            & (measured_height <= self._max_point_h)
            & (measured_height >= self._min_point_h)
        )
        cloud = self._plausible_cloud(depth.cloud_vehicle, plane)

        if valid_roi is not None and valid_roi.shape != measured.shape:
            logger.debug("Ignoring ROI of shape %s for %s terrain", valid_roi.shape, measured.shape)
            valid_roi = None

        # Never invent ground behind our own bodywork - the camera has no
        # evidence there, and a filled pixel would look like clear terrain.
        inferred = self._fill_from_ground_plane(points, measured, plane, valid_roi)
        valid = measured | inferred

        height = self._height_above_plane(points, valid, plane)
        slope = self._surface_slope(points, measured, plane)

        occupied, bev_clearance = self._bev_grids(cloud, plane)
        clearance = self._sample_clearance(points, valid, bev_clearance)
        min_forward = self._corridor_clearance(bev_clearance)

        # Only triangulated pixels may raise an obstacle. An inferred pixel
        # sits on the ground plane by construction, so it can never disagree
        # with it - treating that as evidence of flat ground would be a lie.
        #
        # A pixel also has to be backed by its bird's-eye cell. A single bad
        # stereo match can put one point metres below the ground; a rock puts
        # a whole cluster above it. Requiring cell-level agreement keeps the
        # rock and discards the mismatch.
        obstacle = (
            measured
            & ((height > self._obstacle_h) | (height < self._drop_h) | (slope > self._max_slope))
            & self._sample_occupied(points, measured, occupied)
        )
        if valid_roi is not None:
            # Bodywork is not an obstacle; it is simply not terrain. The
            # cloud above already carried its geometry into the BEV grid.
            obstacle &= valid_roi

        range_m = self._range_along_axis(points, valid)

        traversability = self._traversability(height, slope, clearance, valid)
        if inferred.any():
            traversability[inferred] = np.minimum(traversability[inferred], self._inferred_support)
        if valid_roi is not None:
            traversability[~valid_roi] = 0.0

        obstacle_fraction = 0.0
        if measured.any():
            obstacle_fraction = float(obstacle.sum() / max(int(measured.sum()), 1))
        coverage = 0.0
        measured_coverage = 0.0
        if valid.size:
            coverage = float(valid.mean())
            measured_coverage = float(measured.mean())
        ground_slope_deg = math.degrees(math.atan(math.hypot(float(plane[0]), float(plane[1]))))

        return TerrainAnalysis(
            height_above_ground=height,
            slope_rad=slope,
            obstacle_mask=obstacle,
            clearance_m=clearance,
            traversability=traversability,
            valid=valid,
            range_m=range_m,
            measured=measured,
            inferred=inferred,
            ground_plane=(float(plane[0]), float(plane[1]), float(plane[2])),
            ground_slope_deg=ground_slope_deg,
            obstacle_fraction=obstacle_fraction,
            coverage=coverage,
            measured_coverage=measured_coverage,
            min_forward_clearance_m=min_forward,
            analyze_time_ms=(time.perf_counter() - t0) * 1000.0,
        )

    @staticmethod
    def _build_ray_directions(camera: CameraModel) -> np.ndarray:
        uu, vv = camera.pixel_grid()
        dirs_cam = np.stack(
            [
                (uu - np.float32(camera.cx)) / np.float32(camera.focal_px),
                (vv - np.float32(camera.cy)) / np.float32(camera.focal_px),
                np.ones_like(uu),
            ],
            axis=-1,
        )
        dirs = dirs_cam @ camera.rotation.astype(np.float32)
        norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return (dirs / np.maximum(norm, 1e-9)).astype(np.float32)

    def _fill_from_ground_plane(
        self,
        points: np.ndarray,
        measured: np.ndarray,
        plane: np.ndarray,
        valid_roi: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stereo leaves holes (textureless dirt, rows outside the ROI, the
        near field below the minimum resolvable depth), often exactly where
        the planner starts its path. Each missing ray is intersected with the
        fitted ground plane instead; this fills the maps and never creates an
        obstacle. Writes into ``points`` in place.
        """
        filled = np.zeros(measured.shape, dtype=bool)
        if not self._fill_enabled or not self._has_plane:
            return filled

        missing = ~measured
        if valid_roi is not None:
            missing &= valid_roi
        if not missing.any():
            return filled

        a, b, c = (np.float32(plane[0]), np.float32(plane[1]), np.float32(plane[2]))
        origin = self._ray_origin
        dirs = self._ray_dirs

        # Solve O_z + t*D_z = a(O_x + t*D_x) + b(O_y + t*D_y) + c for t.
        denominator = dirs[..., 2] - a * dirs[..., 0] - b * dirs[..., 1]
        numerator = a * origin[0] + b * origin[1] + c - origin[2]

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(np.abs(denominator) > 1e-6, numerator / denominator, np.float32(-1.0))

        hit = missing & (t > 0.0) & (t < self._fill_max_m) & np.isfinite(t)
        if not hit.any():
            return filled

        # Whole-array arithmetic then a masked copy; the equivalent fancy
        # index costs roughly three times as much on a full frame.
        np.copyto(points, origin + dirs * t[..., None], where=hit[..., None])
        filled |= hit
        return filled

    def _fit_ground_plane(self, cloud: np.ndarray) -> np.ndarray:
        """``z = a*x + b*y + c`` in two passes: the first is pulled upward by
        anything tall in the corridor, the second refits on the points the
        first placed near the surface."""
        if cloud.size == 0:
            return self._plane

        forward = -cloud[:, 1]
        lateral = cloud[:, 0]
        window = (
            (forward >= self._fit_near)
            & (forward <= self._fit_far)
            & (np.abs(lateral) <= self._fit_lateral)
        )
        selected = cloud[window]
        if len(selected) < self._fit_min_points:
            if not self._has_plane:
                logger.debug(
                    "Ground fit skipped: %d points in window (need %d)",
                    len(selected),
                    self._fit_min_points,
                )
            return self._plane

        plane = self._least_squares_plane(selected)
        if plane is None:
            return self._plane

        residual = selected[:, 2] - (
            plane[0] * selected[:, 0] + plane[1] * selected[:, 1] + plane[2]
        )
        # Median absolute deviation gives a scale that tall obstacles do not
        # inflate the way a standard deviation would.
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if scale > 1e-3:
            inliers = selected[np.abs(residual) < 2.5 * scale]
            if len(inliers) >= self._fit_min_points // 2:
                refined = self._least_squares_plane(inliers)
                if refined is not None:
                    plane = refined

        self._plane = plane
        self._has_plane = True
        return plane

    @staticmethod
    def _least_squares_plane(points: np.ndarray) -> np.ndarray | None:
        design = np.column_stack(
            [points[:, 0], points[:, 1], np.ones(len(points), dtype=points.dtype)]
        ).astype(np.float64)
        try:
            solution, *_ = np.linalg.lstsq(design, points[:, 2].astype(np.float64), rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(solution)):
            return None
        return solution

    @staticmethod
    def _height_above_plane(
        points: np.ndarray,
        valid: np.ndarray,
        plane: np.ndarray,
    ) -> np.ndarray:
        height = np.zeros(valid.shape, dtype=np.float32)
        if not valid.any():
            return height
        a = np.float32(plane[0])
        b = np.float32(plane[1])
        c = np.float32(plane[2])
        ground = (a * points[..., 0] + b * points[..., 1] + c).astype(np.float32)
        np.subtract(points[..., 2], ground, out=height, where=valid)
        return height

    def _range_along_axis(self, points: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Includes inferred pixels so the depth view shows no hole exactly
        where the vehicle is about to drive."""
        range_m = np.zeros(valid.shape, dtype=np.float32)
        if not valid.any():
            return range_m
        forward = self._camera.rotation[2].astype(np.float32)
        offsets = points - self._ray_origin
        np.copyto(range_m, offsets @ forward, where=valid)
        return np.maximum(range_m, 0.0, out=range_m)

    def _plausible_cloud(self, cloud: np.ndarray, plane: np.ndarray) -> np.ndarray:
        if cloud.size == 0:
            return cloud
        ground = plane[0] * cloud[:, 0] + plane[1] * cloud[:, 1] + plane[2]
        height = cloud[:, 2] - ground
        # A wrong stereo match does not produce a slightly wrong point, it
        # produces one metres off the surface. Ruts a vehicle can survive sit
        # well inside this band, so anything outside it is a matching failure
        # rather than a hole in the ground.
        keep = (height <= self._max_point_h) & (height >= self._min_point_h)
        return cloud[keep]

    def _surface_slope(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        plane: np.ndarray,
    ) -> np.ndarray:
        """A normal differenced across holes is noise, so support is checked
        first and the expensive work is skipped when sparse stereo would have
        had every result thrown away anyway."""
        slope = np.zeros(valid.shape, dtype=np.float32)
        if int(valid.sum()) < 16:
            return slope

        weight_blur = cv2.blur(valid.astype(np.float32), self._SLOPE_KERNEL)
        supported = valid & (weight_blur > self._SLOPE_MIN_SUPPORT)
        if not supported.any():
            return slope

        # Stereo fills a band of the frame, not the whole of it. Working
        # inside the bounding box of that band - with a margin for the
        # differencing stencil - keeps this off the empty sky and near field.
        window = self._support_window(supported, points.shape[:2])
        rows, cols = window
        points_win = points[rows, cols]
        valid_win = valid[rows, cols]
        weight_win = weight_blur[rows, cols]
        supported_win = supported[rows, cols]

        # Smooth across valid neighbours only, so holes do not drag the
        # averaged points toward the origin and fake a cliff at every gap.
        filled = np.empty_like(points_win)
        for axis in range(3):
            channel = np.where(valid_win, points_win[..., axis], 0.0).astype(np.float32)
            blurred = cv2.blur(channel, self._SLOPE_KERNEL)
            filled[..., axis] = np.divide(
                blurred,
                weight_win,
                out=np.zeros_like(blurred),
                where=weight_win > 1e-3,
            )

        # Sobel with a 1/8 scale matches a central difference and is markedly
        # faster than np.gradient.
        d_du = np.empty_like(filled)
        d_dv = np.empty_like(filled)
        for axis in range(3):
            channel = filled[..., axis]
            d_du[..., axis] = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
            d_dv[..., axis] = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)

        normal = np.cross(d_du, d_dv)
        norm = np.linalg.norm(normal, axis=-1)

        usable = supported_win & (norm > 1e-6)
        if not usable.any():
            return slope

        # Ground normal for z = a*x + b*y + c is (-a, -b, 1), normalised.
        ground_normal = np.array([-plane[0], -plane[1], 1.0], dtype=np.float32)
        ground_normal /= float(np.linalg.norm(ground_normal))

        cos_angle = np.abs((normal[usable] @ ground_normal) / norm[usable])
        slope_win = np.zeros(usable.shape, dtype=np.float32)
        slope_win[usable] = np.arccos(np.clip(cos_angle, 0.0, 1.0)).astype(np.float32)
        slope[rows, cols] = slope_win
        return slope

    def _support_window(
        self,
        supported: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[slice, slice]:
        rows = np.flatnonzero(supported.any(axis=1))
        cols = np.flatnonzero(supported.any(axis=0))
        pad = max(self._SLOPE_KERNEL) // 2 + 1
        r0 = max(0, int(rows[0]) - pad)
        r1 = min(shape[0], int(rows[-1]) + pad + 1)
        c0 = max(0, int(cols[0]) - pad)
        c1 = min(shape[1], int(cols[-1]) + pad + 1)
        return slice(r0, r1), slice(c0, c1)

    def _bev_grids(
        self,
        cloud: np.ndarray,
        plane: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(occupied, clearance_m)``; rows run forward, columns run right."""
        shape = (self._bev_rows, self._bev_cols)
        occupied = np.zeros(shape, dtype=bool)
        clearance = np.full(shape, self._bev_forward, dtype=np.float32)
        if cloud.size == 0:
            return occupied, clearance

        forward = -cloud[:, 1]
        # +X is the vehicle's left, so negate to make columns run rightward.
        lateral = -cloud[:, 0]
        ground = plane[0] * cloud[:, 0] + plane[1] * cloud[:, 1] + plane[2]
        height = cloud[:, 2] - ground

        inside = (
            (forward >= 0.0) & (forward < self._bev_forward) & (np.abs(lateral) < self._bev_lateral)
        )
        if not inside.any():
            return occupied, clearance

        rows = (forward[inside] / self._bev_cell).astype(np.int32)
        cols = ((lateral[inside] + self._bev_lateral) / self._bev_cell).astype(np.int32)
        np.clip(rows, 0, self._bev_rows - 1, out=rows)
        np.clip(cols, 0, self._bev_cols - 1, out=cols)

        cell_count = self._bev_rows * self._bev_cols
        flat_idx = rows * self._bev_cols + cols
        blocking = (height[inside] > self._obstacle_h) | (height[inside] < self._drop_h)

        total = np.bincount(flat_idx, minlength=cell_count).reshape(shape)
        hits = np.bincount(flat_idx[blocking], minlength=cell_count).reshape(shape)

        # Two independent conditions. The absolute count rejects lone speckles;
        # the ratio rejects a cell where a handful of bad matches are outvoted
        # by points that agree with the ground.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(hits, np.maximum(total, 1)).astype(np.float32)
        occupied = (hits >= self._bev_min_points) & (ratio >= self._bev_min_ratio)
        if not occupied.any():
            return occupied, clearance

        free = (~occupied).astype(np.uint8)
        distance_cells = cv2.distanceTransform(free, cv2.DIST_L2, 3)
        clearance = np.minimum(distance_cells * self._bev_cell, self._bev_forward).astype(
            np.float32
        )
        return occupied, clearance

    def _sample_occupied(
        self,
        points: np.ndarray,
        measured: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        backed = np.zeros(measured.shape, dtype=bool)
        if not measured.any() or not occupied.any():
            return backed

        forward = -points[..., 1]
        lateral = -points[..., 0]
        rows = (forward / self._bev_cell).astype(np.int32)
        cols = ((lateral + self._bev_lateral) / self._bev_cell).astype(np.int32)

        inside = (
            measured & (rows >= 0) & (rows < self._bev_rows) & (cols >= 0) & (cols < self._bev_cols)
        )
        if not inside.any():
            return backed

        backed[inside] = occupied[rows[inside], cols[inside]]
        return backed

    def _sample_clearance(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        bev_clearance: np.ndarray,
    ) -> np.ndarray:
        clearance = np.full(valid.shape, self._bev_forward, dtype=np.float32)
        if not valid.any():
            return clearance

        forward = -points[..., 1]
        lateral = -points[..., 0]
        rows = (forward / self._bev_cell).astype(np.int32)
        cols = ((lateral + self._bev_lateral) / self._bev_cell).astype(np.int32)

        inside = (
            valid & (rows >= 0) & (rows < self._bev_rows) & (cols >= 0) & (cols < self._bev_cols)
        )
        if not inside.any():
            return clearance

        clearance[inside] = bev_clearance[rows[inside], cols[inside]]
        return clearance

    def _corridor_clearance(self, bev_clearance: np.ndarray) -> float:
        half_cols = max(1, int(round(self._half_width / self._bev_cell)))
        centre = self._bev_cols // 2
        left = max(0, centre - half_cols)
        right = min(self._bev_cols, centre + half_cols + 1)
        rows = min(self._bev_rows, max(1, int(round(self._lookahead / self._bev_cell))))

        corridor = bev_clearance[:rows, left:right]
        if corridor.size == 0:
            return float("inf")

        blocked_rows = np.nonzero((corridor <= 1e-3).any(axis=1))[0]
        if len(blocked_rows) == 0:
            return float("inf")
        return float(blocked_rows[0] * self._bev_cell)

    def _traversability(
        self,
        height: np.ndarray,
        slope: np.ndarray,
        clearance: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        score = np.zeros(valid.shape, dtype=np.float32)
        if not valid.any():
            return score

        height_cost = np.clip(
            np.where(
                height >= 0.0,
                height / max(self._obstacle_h, 1e-3),
                height / min(self._drop_h, -1e-3),
            ),
            0.0,
            1.0,
        )
        slope_cost = np.clip(slope / max(self._max_slope, 1e-3), 0.0, 1.0)
        # One vehicle half-width of clearance counts as fully clear.
        clearance_gain = np.clip(clearance / max(self._half_width, 1e-3), 0.0, 1.0)

        blended = (1.0 - height_cost) * (1.0 - 0.7 * slope_cost)
        blended = blended * (0.35 + 0.65 * clearance_gain)
        score[valid] = np.clip(blended[valid], 0.0, 1.0).astype(np.float32)
        return score
