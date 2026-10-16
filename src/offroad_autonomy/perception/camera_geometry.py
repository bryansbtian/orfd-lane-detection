"""Pinhole camera models and rigid transforms for the dashcam.

Two coordinate frames are in play.

**Vehicle space** is BeamNG's: ``+X`` left, ``-Y`` forward, ``+Z`` up.  This
is the frame every :class:`~offroad_autonomy.types.CameraSpec` is written in,
and the frame used by ground-plane path projection.

**Camera space** is the usual vision convention: ``+X`` right, ``+Y`` down,
``+Z`` along the viewing direction.  Projection is an exact pinhole - BeamNG
renders a rectilinear image with no lens distortion, so no undistortion step
is needed for the simulated lens.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from offroad_autonomy.types import CameraSpec

logger = logging.getLogger("offroad_autonomy.perception.geometry")


def _normalize(vec: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalise a zero-length vector: {vec!r}")
    return arr / norm


class CameraModel:
    """Intrinsics and extrinsics for one camera at a chosen working size.

    The focal length is derived from the horizontal field of view and the
    *working* width rather than stored, so downscaling a frame can never
    leave a stale native-resolution focal length behind.
    """

    def __init__(
        self,
        spec: CameraSpec,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.spec = spec
        self.sensor = spec.sensor
        self.name = spec.name
        if width is None:
            width = spec.width
        if height is None:
            height = spec.height
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{spec.name}: working resolution must be positive")

        native_aspect = spec.width / max(spec.height, 1)
        work_aspect = self.width / max(self.height, 1)
        if abs(native_aspect - work_aspect) > 1e-3:
            logger.debug(
                "%s: working aspect %.4f differs from native %.4f - a "
                "non-uniform resize requires independent fx/fy scaling",
                spec.name,
                work_aspect,
                native_aspect,
            )

        half_fov = math.radians(spec.fov_x_deg) / 2.0
        if not 0.0 < half_fov < math.pi / 2.0:
            raise ValueError(f"{spec.name}: fov_x_deg must be in (0, 180)")

        self.scale = self.width / spec.width
        self.focal_px = (self.width / 2.0) / math.tan(half_fov)
        self.focal_y_px = self.focal_px * (self.height / spec.height) / self.scale
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        if spec.intrinsics is not None:
            native = np.asarray(spec.intrinsics, dtype=np.float64)
            if native.shape != (4,) or not np.isfinite(native).all() or np.any(native[:2] <= 0):
                raise ValueError(
                    "Camera intrinsics must be finite fx, fy, cx, cy with positive focal lengths"
                )
            self.focal_px = float(native[0] * self.scale)
            self.focal_y_px = float(native[1] * self.height / spec.height)
            self.cx = float(native[2] * self.scale)
            self.cy = float(native[3] * self.height / spec.height)

        self.position = np.asarray(spec.pos, dtype=np.float64)

        forward = _normalize(spec.dir)
        up_hint = _normalize(spec.up)
        right = np.cross(forward, up_hint)
        if float(np.linalg.norm(right)) < 1e-6:
            raise ValueError(f"{spec.name}: dir and up are parallel")
        right = _normalize(right)
        down = np.cross(forward, right)

        # Rows are the camera axes expressed in vehicle space, so this matrix
        # maps a vehicle-space offset into camera space.
        self.rotation = np.stack([right, down, forward])

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(
            math.atan(self.cx / self.focal_px) + math.atan((self.width - self.cx) / self.focal_px)
        )

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(
            math.atan(self.cy / self.focal_y_px)
            + math.atan((self.height - self.cy) / self.focal_y_px)
        )

    def describe(self) -> str:
        return (
            f"{self.name}: {self.sensor.model} {self.sensor.width}x"
            f"{self.sensor.height} -> {self.width}x{self.height}, "
            f"f={self.focal_px:.1f} px, "
            f"hfov={self.horizontal_fov_deg:.1f} deg"
        )

    def from_vehicle(self, points_vehicle: np.ndarray) -> np.ndarray:
        offset = np.asarray(points_vehicle, dtype=np.float32) - self.position.astype(np.float32)
        return offset @ self.rotation.T.astype(np.float32)

    def to_vehicle(self, points_camera: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_camera, dtype=np.float32)
        return pts @ self.rotation.astype(np.float32) + self.position.astype(np.float32)

    def backproject(
        self,
        u: np.ndarray,
        v: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        f = np.float32(self.focal_px)
        x = (np.asarray(u, dtype=np.float32) - np.float32(self.cx)) * depth / f
        y = (
            (np.asarray(v, dtype=np.float32) - np.float32(self.cy))
            * depth
            / np.float32(self.focal_y_px)
        )
        return np.stack([x, y, np.asarray(depth, dtype=np.float32)], axis=-1)

    def project(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(u, v, z)``; callers must drop entries with ``z <= 0``, which sit
        behind the image plane and would otherwise project mirrored."""
        pts = np.asarray(points_camera, dtype=np.float32)
        z = pts[..., 2]
        safe_z = np.where(np.abs(z) < 1e-6, np.float32(1e-6), z)
        u = np.float32(self.focal_px) * pts[..., 0] / safe_z + np.float32(self.cx)
        v = np.float32(self.focal_y_px) * pts[..., 1] / safe_z + np.float32(self.cy)
        return u, v, z

    def image_to_ground(
        self, uv: np.ndarray, max_range_m: float = 30.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Intersect pixel rays with the flat ground plane (vehicle z = 0).

        Returns ``(forward_m, right_m, keep)``: metres ahead of and to the
        right of the camera, and which inputs hit the ground within
        ``max_range_m``. Vehicle space is BeamNG's: forward -Y, left +X, up +Z.
        """
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        f = float(self.focal_px)
        rays_cam = np.stack(
            [(pts[:, 0] - self.cx) / f, (pts[:, 1] - self.cy) / self.focal_y_px, np.ones(len(pts))],
            axis=1,
        )
        rays = rays_cam @ self.rotation  # camera axes -> vehicle space
        origin = self.position
        down = rays[:, 2] < -1e-6
        t = np.where(down, -origin[2] / np.where(down, rays[:, 2], -1.0), np.nan)
        hit = origin[None, :] + t[:, None] * rays
        forward = -(hit[:, 1] - origin[1])
        right = -(hit[:, 0] - origin[0])
        keep = down & np.isfinite(forward) & (forward > 0.0) & (forward < max_range_m)
        return forward, right, keep

    def reliable_ground(
        self,
        uv: np.ndarray,
        max_range_m: float = 30.0,
        pitch_uncertainty_deg: float = 1.0,
        max_relative_error: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reject rays sensitive to the configured ground-relative pitch bound.

        This is a sensitivity envelope, not an estimate of terrain slope or
        vehicle attitude. A crest cannot be certified using this RGB model.
        """
        forward, right, keep = self.image_to_ground(uv, max_range_m)
        if pitch_uncertainty_deg <= 0:
            return forward, right, keep
        angle = np.arctan2(self.position[2], forward)
        delta = math.radians(pitch_uncertainty_deg)
        low, high = angle - delta, angle + delta
        safe_low = np.maximum(low, 1e-8)
        far = self.position[2] / np.tan(safe_low)
        near = self.position[2] / np.tan(high)
        error = np.maximum(np.abs(far - forward), np.abs(near - forward))
        keep &= (low > 0) & (high < math.pi / 2) & (error <= max_relative_error * forward)
        return forward, right, keep

    def ground_to_image(self, forward_m: np.ndarray, right_m: np.ndarray) -> np.ndarray:
        forward_m = np.asarray(forward_m, dtype=np.float64)
        right_m = np.asarray(right_m, dtype=np.float64)
        origin = self.position
        vehicle = np.stack(
            [origin[0] - right_m, origin[1] - forward_m, np.zeros_like(forward_m)], axis=1
        )
        u, v, _ = self.project(self.from_vehicle(vehicle))
        return np.stack([u, v], axis=1).astype(np.float32)

    def pixel_grid(self) -> tuple[np.ndarray, np.ndarray]:
        u = np.arange(self.width, dtype=np.float32)
        v = np.arange(self.height, dtype=np.float32)
        return np.meshgrid(u, v)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.focal_px, 0.0, self.cx],
                [0.0, self.focal_y_px, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
