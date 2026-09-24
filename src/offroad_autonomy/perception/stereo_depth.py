"""Metric depth from the front stereo pair.

Pipeline for one synchronised pair:

1. **Rectify** the original left/right frames with maps precomputed in
   ``StereoRectifier`` - never a stitched or otherwise derived image.
2. **Match** with semi-global block matching, only inside the forward ROI
   row band when the ROI is enabled (the sky and the far horizon cost as much
   to match as the road and are worth nothing to the planner).
3. **Filter**: ``Z = f * B / d`` in the rectified-left frame, rejecting
   invalid, zero and negative disparities and anything outside
   ``[min_depth_m, max_depth_m]``.
4. **Gate and reproject**: lift each valid pixel to a vehicle-space point,
   keep it only inside the forward corridor and near the traversable mask,
   and scatter it into the *segmentation* view with a depth-sorted write so
   the nearest surface wins.

Raw disparity and filtered depth are kept on the rectified grid for display;
the reprojected maps share the segmentation mask's grid for fusion.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.stereo_rectification import StereoRectifier
from offroad_autonomy.types import DepthResult, PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception.depth")


class StereoDepthEstimator:
    def __init__(self, config: PipelineConfig, target: CameraModel | None = None) -> None:
        self._min_depth = float(config.stereo_min_depth_m)
        self._max_depth = float(config.stereo_max_depth_m)
        if not 0.0 < self._min_depth < self._max_depth:
            raise ValueError("depth.min_depth_m must be positive and below max_depth_m")
        self._median_blur = int(config.stereo_median_blur)

        self.rectifier = StereoRectifier(config)
        self.rig = self.rectifier.rig
        self.rectified = self.rectifier.left_camera
        # Depth is delivered on the segmentation grid so it fuses pixel for
        # pixel with the mask.
        if target is None:
            target = CameraModel(
                config.segmentation_camera, config.preprocess_width, config.preprocess_height
            )
        self.target = target

        self.focal_px = self.rectifier.focal_px
        self.baseline_m = self.rectifier.baseline_m
        self._fb = np.float32(self.focal_px * self.baseline_m)

        num_disp = self._resolve_disparity_range(config)
        block = int(config.stereo_block_size)
        if block % 2 == 0:
            block += 1
        block = max(3, block)

        self._matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disp,
            blockSize=block,
            P1=int(config.stereo_p1_factor) * 3 * block * block,
            P2=int(config.stereo_p2_factor) * 3 * block * block,
            disp12MaxDiff=int(config.stereo_disp12_max_diff),
            uniquenessRatio=int(config.stereo_uniqueness_ratio),
            speckleWindowSize=int(config.stereo_speckle_window_size),
            speckleRange=int(config.stereo_speckle_range),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.num_disparities = num_disp

        # Disparity below this is beyond max range (and mostly noise); above
        # the search range it cannot have been measured.
        self._min_disparity = max(1.0, self._disparity_for(self._max_depth))
        self._max_disparity = float(num_disp)

        height = self.rectified.height
        self._roi_enabled = bool(config.depth_roi_enabled)
        if self._roi_enabled:
            top = float(np.clip(config.depth_roi_row_top, 0.0, 1.0))
            bottom = float(np.clip(config.depth_roi_row_bottom, top, 1.0))
            self._row0 = int(math.floor(top * height))
            self._row1 = max(self._row0 + 1, int(math.ceil(bottom * height)))
        else:
            self._row0, self._row1 = 0, height
        self._use_mask = self._roi_enabled and bool(config.depth_roi_use_mask)
        dilation = max(0, int(config.depth_roi_mask_dilation_px))
        self._mask_kernel = None
        if dilation > 0:
            self._mask_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation + 1,) * 2)
        self._corridor_half = float(config.depth_roi_corridor_half_width_m)
        self._corridor_length = float(config.depth_roi_corridor_length_m)
        self._vehicle_half = float(config.vehicle_half_width_m)
        self._rig_y = float(self.rectified.position[1])

        logger.info("Stereo rig: %s", self.rig.describe())
        logger.info(
            "Stereo matching: %d disparities, block %d, rows %d..%d of %d (ROI %s)",
            num_disp,
            block,
            self._row0,
            self._row1,
            height,
            self._roi_enabled,
        )
        near_limit = self._disparity_for(self._min_depth)
        if near_limit > num_disp:
            logger.warning(
                "num_disparities=%d cannot resolve closer than %.1f m; raise "
                "depth.max_disparities or depth.min_depth_m",
                num_disp,
                float(self._fb) / num_disp,
            )

    def _disparity_for(self, depth_m: float) -> float:
        if depth_m <= 0.0:
            return float("inf")
        return float(self._fb) / depth_m

    def _resolve_disparity_range(self, config: PipelineConfig) -> int:
        """Derived from the rig rather than chosen, so moving the cameras can
        never leave the near field silently outside the search: the closest
        resolvable depth is ``f * B / numDisparities``."""
        cap = max(16, int(math.ceil(config.stereo_max_disparities / 16.0)) * 16)

        override = int(config.stereo_num_disparities)
        if override > 0:
            chosen = max(16, int(math.ceil(override / 16.0)) * 16)
            logger.info("Disparity range pinned by config: %d", chosen)
            return min(chosen, cap)

        needed = max(16, int(math.ceil(self._disparity_for(self._min_depth) / 16.0)) * 16)
        chosen = min(needed, cap)
        logger.info(
            "Disparity range derived: f=%.1f px * B=%.2f m / %.1f m = %.0f px -> %d (needed %d)",
            self.focal_px,
            self.baseline_m,
            self._min_depth,
            self._disparity_for(self._min_depth),
            chosen,
            needed,
        )
        return chosen

    def compute(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        road_mask: np.ndarray | None = None,
        valid_roi: np.ndarray | None = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> DepthResult | None:
        """``None`` when the pair yields no usable correspondences, so the
        caller falls back to appearance-only perception."""
        if left_bgr is None or right_bgr is None or left_bgr.size == 0 or right_bgr.size == 0:
            return None
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        rect_left, rect_right = self.rectifier.rectify(left_bgr, right_bgr)
        t1 = time.perf_counter()
        timings["rectification"] = (t1 - t0) * 1000.0

        band_left = self._gray(rect_left[self._row0 : self._row1])
        band_right = self._gray(rect_right[self._row0 : self._row1])
        band = self._matcher.compute(band_left, band_right).astype(np.float32) / 16.0
        if self._median_blur >= 3:
            blur = min(5, self._median_blur + (1 - self._median_blur % 2))
            band = cv2.medianBlur(band, blur)
        t2 = time.perf_counter()
        timings["stereo_matching"] = (t2 - t1) * 1000.0

        disparity = np.full(rect_left.shape[:2], -1.0, dtype=np.float32)
        disparity[self._row0 : self._row1] = band

        # Invalid (SGBM writes minDisparity - 1), zero and negative disparity
        # all fail the lower bound; the upper bound rejects the search edge.
        valid = (
            np.isfinite(disparity)
            & (disparity > self._min_disparity)
            & (disparity < self._max_disparity)
        )
        band_pixels = max(1, band.size)
        valid_fraction = float(valid.sum()) / band_pixels

        depth_rect = np.zeros_like(disparity)
        np.divide(self._fb, disparity, out=depth_rect, where=valid)
        in_range = valid & (depth_rect >= self._min_depth) & (depth_rect <= self._max_depth)
        depth_rect[~in_range] = 0.0
        if not in_range.any():
            logger.debug("Stereo produced no in-range depth")
            return None

        roi_pixels = self._roi_pixels(road_mask, valid_roi)
        cloud, points_image, depth_image, valid_image = self._reproject(
            depth_rect, in_range, roi_pixels
        )
        t3 = time.perf_counter()
        timings["depth_filtering"] = (t3 - t2) * 1000.0

        denominator = roi_pixels
        if denominator is None:
            denominator = valid_roi
        if denominator is not None and denominator.shape == valid_image.shape:
            coverage = float((valid_image & denominator).sum()) / max(int(denominator.sum()), 1)
        else:
            coverage = float(valid_image.mean())

        median_fwd, min_fwd = self._corridor_stats(cloud)

        return DepthResult(
            depth_m=depth_image,
            valid=valid_image,
            points_vehicle=points_image,
            cloud_vehicle=cloud,
            disparity=disparity,
            depth_rect=depth_rect,
            rectified_left=rect_left,
            rectified_right=rect_right,
            coverage=coverage,
            valid_disparity_fraction=valid_fraction,
            median_forward_depth_m=median_fwd,
            min_corridor_depth_m=min_fwd,
            compute_time_ms=(t3 - t0) * 1000.0,
            frame_id=frame_id,
            timestamp=timestamp,
            timings_ms=timings,
        )

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _roi_pixels(
        self,
        road_mask: np.ndarray | None,
        valid_roi: np.ndarray | None,
    ) -> np.ndarray | None:
        """``dilate(road) AND valid``. Dilation keeps the road edge and anything
        standing on it. With no road in view the gate is dropped rather than
        blinding depth exactly when appearance has failed.
        """
        target_shape = (self.target.height, self.target.width)
        if valid_roi is not None and valid_roi.shape != target_shape:
            valid_roi = None
        if not self._use_mask or road_mask is None or road_mask.shape != target_shape:
            return valid_roi
        if not road_mask.any():
            return valid_roi
        gate = road_mask.astype(np.uint8)
        if self._mask_kernel is not None:
            gate = cv2.dilate(gate, self._mask_kernel)
        gate = gate.astype(bool)
        if valid_roi is None:
            return gate
        return gate & valid_roi

    def roi_preview(
        self,
        road_mask: np.ndarray | None,
        valid_roi: np.ndarray | None,
    ) -> np.ndarray | None:
        """Scaling the band's top row is exact for a parallel pair segmented on
        either camera and only approximate for the stitched view, which is
        acceptable for a debug view."""
        if not self._roi_enabled:
            return None
        shape = (self.target.height, self.target.width)
        roi = self._roi_pixels(road_mask, valid_roi)
        if roi is None:
            roi = np.ones(shape, dtype=bool)
        else:
            roi = roi.copy()
        cutoff = int(round(self._row0 * self.target.height / self.rectified.height))
        roi[:cutoff] = False
        return roi

    def _corridor_stats(self, cloud: np.ndarray) -> tuple[float, float]:
        if cloud.size == 0:
            return float("nan"), float("nan")
        forward = self._rig_y - cloud[:, 1]
        lane = (np.abs(cloud[:, 0]) <= self._vehicle_half) & (forward > 0.0)
        if not lane.any():
            return float("nan"), float("nan")
        values = forward[lane]
        return float(np.median(values)), float(values.min())

    def _reproject(
        self,
        depth_rect: np.ndarray,
        valid: np.ndarray,
        roi_pixels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(cloud, points_image, depth_image, valid_image)``; each target
        pixel holds the nearest surface along its ray."""
        source = self.rectified
        target = self.target

        vs, us = np.nonzero(valid)
        depths = depth_rect[vs, us]
        points_cam = source.backproject(us.astype(np.float32), vs.astype(np.float32), depths)
        cloud = source.to_vehicle(points_cam)

        keep = np.ones(len(cloud), dtype=bool)
        if self._roi_enabled:
            forward = self._rig_y - cloud[:, 1]
            keep &= (
                (np.abs(cloud[:, 0]) <= self._corridor_half)
                & (forward >= 0.0)
                & (forward <= self._corridor_length)
            )

        u_t, v_t, z_t = target.project(target.from_vehicle(cloud))
        ui = np.round(u_t).astype(np.int32)
        vi = np.round(v_t).astype(np.int32)
        visible = (z_t > 0.0) & (ui >= 0) & (ui < target.width) & (vi >= 0) & (vi < target.height)
        if roi_pixels is not None and roi_pixels.shape == (target.height, target.width):
            in_roi = np.zeros(len(cloud), dtype=bool)
            in_roi[visible] = roi_pixels[vi[visible], ui[visible]]
            keep &= in_roi

        depth_image = np.zeros((target.height, target.width), dtype=np.float32)
        points_image = np.zeros((target.height, target.width, 3), dtype=np.float32)
        valid_image = np.zeros((target.height, target.width), dtype=bool)

        kept_cloud = cloud[keep]
        draw = visible & keep
        if not draw.any():
            return kept_cloud, points_image, depth_image, valid_image

        flat_idx = vi[draw] * target.width + ui[draw]
        z_draw = z_t[draw]
        cloud_draw = cloud[draw]

        # Painter's algorithm: write far surfaces first so that the nearest
        # point at each pixel is the one that survives.
        order = np.argsort(-z_draw, kind="stable")
        idx = flat_idx[order]

        depth_image.reshape(-1)[idx] = z_draw[order]
        points_image.reshape(-1, 3)[idx] = cloud_draw[order]
        valid_image.reshape(-1)[idx] = True

        return kept_cloud, points_image, depth_image, valid_image
