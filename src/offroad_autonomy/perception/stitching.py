"""Wide forward view from the stereo pair, by precomputed remapping.

The cameras are bolted to the vehicle, so the mapping from a virtual wide
camera's pixels to each real camera's pixels never changes. It is computed
once here - no feature matching, no per-frame homography estimation - and
each frame costs two ``cv2.remap`` calls plus a feathered blend.

The virtual camera sits at the midpoint of the pair, faces the mean of the
two optical axes, and keeps the working focal length so it has the same
angular resolution as a single camera. Its width is whatever the union of
the two fields of view needs.

Two limits worth knowing:

* Mapping is by *rotation only* (points at infinity). The cameras are
  ``baseline_m`` apart, so near-field objects show a parallax seam at the
  centre column. That is harmless for a view that is only looked at or
  segmented; it is exactly why this image must never be used for stereo.
* With parallel optical axes (``toe_out_deg: 0``) the two views cover the
  same angles and the stitched image is barely wider than one camera - the
  width only comes from toe-out, which costs stereo overlap.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import CameraSensor, CameraSpec, PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception.stitching")


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


class WideViewStitcher:
    #: A rectilinear image stretches badly near 90 deg off-axis.
    MAX_HALF_FOV_DEG = 75.0

    def __init__(
        self,
        left_spec: CameraSpec,
        right_spec: CameraSpec,
        height: int,
        focal_px: float,
        feather_px: int = 24,
    ) -> None:
        self._left = CameraModel(left_spec)
        self._right = CameraModel(right_spec)
        self.capture_size = (self._left.width, self._left.height)

        forward = _normalize(self._left.rotation[2] + self._right.rotation[2])
        # Camera row 1 is "down", so the mean up vector is its negation.
        up = _normalize(-(self._left.rotation[1] + self._right.rotation[1]))
        centre = (self._left.position + self._right.position) / 2.0

        half = self._half_fov(forward, up)
        width = 2 * int(math.ceil(focal_px * math.tan(half)))
        sensor = CameraSensor(
            model="stitched",
            width=width,
            height=int(height),
            fov_x_deg=math.degrees(2.0 * math.atan((width / 2.0) / focal_px)),
            target_fps=left_spec.sensor.target_fps,
        )
        spec = CameraSpec(
            name="wide_front",
            pos=tuple(float(v) for v in centre),
            dir=tuple(float(v) for v in forward),
            up=tuple(float(v) for v in up),
            sensor=sensor,
        )
        self.camera = CameraModel(spec, width, int(height))

        rays = self._virtual_rays()
        map_left, valid_left = self._source_map(rays, self._left)
        map_right, valid_right = self._source_map(rays, self._right)
        self._maps_left = cv2.convertMaps(map_left[..., 0], map_left[..., 1], cv2.CV_16SC2)
        self._maps_right = cv2.convertMaps(map_right[..., 0], map_right[..., 1], cv2.CV_16SC2)

        # A feathered seam hides the exposure step between the two cameras.
        offset = self.camera.pixel_grid()[0] - np.float32(self.camera.cx)
        feather = max(float(feather_px), 1e-3)
        ramp = np.clip(0.5 - offset / feather, 0.0, 1.0).astype(np.float32)
        both = valid_left & valid_right
        self._weight_left = np.where(both, ramp, valid_left.astype(np.float32)).astype(np.float32)
        self._weight_right = np.where(valid_right, 1.0 - self._weight_left, 0.0).astype(np.float32)
        self.coverage = valid_left | valid_right

        logger.info(
            "Stitched view: %dx%d, hfov %.1f deg, %.1f%% of pixels covered",
            width,
            height,
            sensor.fov_x_deg,
            100.0 * float(self.coverage.mean()),
        )

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "WideViewStitcher":
        work_left = CameraModel(
            config.left_camera, config.preprocess_width, config.preprocess_height
        )
        return cls(
            config.left_camera,
            config.right_camera,
            height=config.preprocess_height,
            focal_px=work_left.focal_px,
            feather_px=config.stitch_feather_px,
        )

    def _half_fov(self, forward: np.ndarray, up: np.ndarray) -> float:
        right_axis = _normalize(np.cross(forward, up))
        widest = 0.0
        for cam in (self._left, self._right):
            vs = np.linspace(0.0, cam.height - 1.0, 5, dtype=np.float32)
            for u in (0.0, cam.width - 1.0):
                us = np.full_like(vs, u)
                rays = (
                    np.stack(
                        [
                            (us - cam.cx) / cam.focal_px,
                            (vs - cam.cy) / cam.focal_px,
                            np.ones_like(vs),
                        ],
                        axis=-1,
                    )
                    @ cam.rotation
                )
                angles = np.arctan2(np.abs(rays @ right_axis), rays @ forward)
                widest = max(widest, float(angles.max()))
        return min(widest, math.radians(self.MAX_HALF_FOV_DEG))

    def _virtual_rays(self) -> np.ndarray:
        cam = self.camera
        uu, vv = cam.pixel_grid()
        rays = np.stack(
            [
                (uu - np.float32(cam.cx)) / np.float32(cam.focal_px),
                (vv - np.float32(cam.cy)) / np.float32(cam.focal_px),
                np.ones_like(uu),
            ],
            axis=-1,
        )
        return rays @ cam.rotation.astype(np.float32)

    @staticmethod
    def _source_map(rays_vehicle: np.ndarray, source: CameraModel) -> tuple[np.ndarray, np.ndarray]:
        local = rays_vehicle @ source.rotation.T.astype(np.float32)
        z = local[..., 2]
        safe_z = np.where(z > 1e-6, z, np.float32(1.0))
        u = np.float32(source.focal_px) * local[..., 0] / safe_z + np.float32(source.cx)
        v = np.float32(source.focal_px) * local[..., 1] / safe_z + np.float32(source.cy)
        valid = (
            (z > 1e-6)
            & (u >= 0.0)
            & (u <= source.width - 1.0)
            & (v >= 0.0)
            & (v <= source.height - 1.0)
        )
        # Park invalid lookups outside the image so remap returns the border.
        u = np.where(valid, u, np.float32(-10.0))
        v = np.where(valid, v, np.float32(-10.0))
        return np.stack([u, v], axis=-1).astype(np.float32), valid

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if (image.shape[1], image.shape[0]) != self.capture_size:
            image = cv2.resize(image, self.capture_size, interpolation=cv2.INTER_AREA)
        return image

    def stitch(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        warped_left = cv2.remap(
            self._prepare(left),
            *self._maps_left,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_right = cv2.remap(
            self._prepare(right),
            *self._maps_right,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return cv2.blendLinear(warped_left, warped_right, self._weight_left, self._weight_right)

    def warp_masks(self, left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
        """Pixels neither camera sees come back excluded: no data is not the
        same as clear ground."""

        def _warp(mask: np.ndarray, maps) -> np.ndarray:
            src = self._prepare(mask.astype(np.uint8))
            return cv2.remap(src, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(
                bool
            )

        warped_left = _warp(left_mask, self._maps_left)
        warped_right = _warp(right_mask, self._maps_right)
        from_left = self._weight_left >= 0.5
        combined = np.where(from_left, warped_left, warped_right)
        return combined | ~self.coverage
