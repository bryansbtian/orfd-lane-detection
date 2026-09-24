"""Stereo rectification for the front camera pair.

Rectification rotates both views so that every world point lands on the
same image row in the left and right images, which is what lets SGBM search
a single row instead of the whole frame.

Everything expensive happens once, in the constructor:

1. Intrinsics ``K`` and distortion ``D`` for each camera, at the resolution
   frames are *captured* at. By default ``K`` is derived from the lens and
   ``D`` is zero (BeamNG renders an ideal pinhole); both can be overridden
   from the config with a real calibration.
2. The rotation ``R`` and translation ``T`` between the cameras, derived from
   the two mounts unless overridden.
3. ``cv2.stereoRectify`` for the rectifying rotations ``R1``/``R2`` and new
   projections ``P1``/``P2`` at the stereo working size.
4. ``cv2.initUndistortRectifyMap`` for fixed-point lookup maps.

Per frame, rectification is two ``cv2.remap`` calls. The maps also fold in
the downscale from capture to working resolution, so there is no separate
resize pass.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import (
    CameraModel,
    StereoRig,
    relative_pose,
)
from offroad_autonomy.types import PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception.rectification")


def _matrix(raw, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{name} must have {int(np.prod(shape))} values, got {array.size}")
    return array.reshape(shape)


class StereoRectifier:
    def __init__(self, config: PipelineConfig) -> None:
        left_spec, right_spec = config.left_camera, config.right_camera
        self.capture_size = (left_spec.width, left_spec.height)
        self.work_size = (int(config.stereo_width), int(config.stereo_height))

        capture_left = CameraModel(left_spec)
        capture_right = CameraModel(right_spec)
        # The rig check runs at working size: it is the pair SGBM will see.
        self.rig = StereoRig(
            CameraModel(left_spec, *self.work_size),
            CameraModel(right_spec, *self.work_size),
        )

        derived_r, derived_t = relative_pose(capture_left, capture_right)
        self.K_left = self._override(
            config.stereo_K_left, capture_left.intrinsic_matrix, (3, 3), "K_left"
        )
        self.K_right = self._override(
            config.stereo_K_right, capture_right.intrinsic_matrix, (3, 3), "K_right"
        )
        self.D_left = self._override(config.stereo_D_left, np.zeros(5), (-1,), "D_left")
        self.D_right = self._override(config.stereo_D_right, np.zeros(5), (-1,), "D_right")
        self.R = self._override(config.stereo_R, derived_r, (3, 3), "R")
        self.T = self._override(config.stereo_T, derived_t, (3,), "T")

        alpha = float(np.clip(config.stereo_rectify_alpha, -1.0, 1.0))
        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.valid_left,
            self.valid_right,
        ) = cv2.stereoRectify(
            self.K_left,
            self.D_left,
            self.K_right,
            self.D_right,
            self.capture_size,
            self.R,
            self.T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha,
            newImageSize=self.work_size,
        )

        self._map_left = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, self.R1, self.P1, self.work_size, cv2.CV_16SC2
        )
        self._map_right = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, self.R2, self.P2, self.work_size, cv2.CV_16SC2
        )

        # After rectification the pair is an ideal parallel rig: same focal
        # length, same principal point, offset only along x.
        self.focal_px = float(self.P1[0, 0])
        self.baseline_m = float(abs(self.P2[0, 3]) / max(self.P2[0, 0], 1e-9))

        # An ordinary camera model, so depth measured on rectified pixels
        # backprojects with no special cases.
        self.left_camera = CameraModel.from_intrinsics(
            self.rig.left,
            f"{left_spec.name}_rectified",
            self.P1,
            self.R1,
            *self.work_size,
        )

        logger.info(
            "Rectification: %dx%d -> %dx%d  f=%.1f px  B=%.3f m  axes %.2f deg apart  R1 %.2f deg",
            *self.capture_size,
            *self.work_size,
            self.focal_px,
            self.baseline_m,
            self.rig.axis_angle_deg,
            np.degrees(np.arccos(np.clip((np.trace(self.R1) - 1.0) / 2.0, -1.0, 1.0))),
        )

    @staticmethod
    def _override(raw, default: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        if raw is None:
            return np.asarray(default, dtype=np.float64).reshape(shape)
        logger.info("Stereo calibration: %s taken from config", name)
        if shape == (-1,):
            shape = (len(raw),)
        return _matrix(raw, shape, name)

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._remap(left, self._map_left),
            self._remap(right, self._map_right),
        )

    def _remap(self, image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        if (image.shape[1], image.shape[0]) != self.capture_size:
            # The maps are built for the capture size; a different frame size
            # would silently index the wrong pixels, so bring it back first.
            image = cv2.resize(image, self.capture_size, interpolation=cv2.INTER_AREA)
        return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)


def draw_epipolar_pair(
    left: np.ndarray,
    right: np.ndarray,
    spacing_px: int = 32,
    color: tuple[int, int, int] = (0, 220, 255),
) -> np.ndarray:
    """A feature crossing a guide line in one half but not the other means
    the calibration is off."""

    def _bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()

    pair = np.hstack([_bgr(left), _bgr(right)])
    height, width = pair.shape[:2]
    for row in range(spacing_px // 2, height, max(4, spacing_px)):
        cv2.line(pair, (0, row), (width - 1, row), color, 1, cv2.LINE_AA)
    cv2.line(pair, (width // 2, 0), (width // 2, height - 1), (40, 40, 40), 2)
    return pair
