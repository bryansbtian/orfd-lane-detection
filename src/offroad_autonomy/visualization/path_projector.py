"""Draw the autonomy stack's output in the display camera's frame.

The mask and the planned path are measured in the *bumper* cameras' view.
The dashboard's main panel shows the *display* camera, which sits higher and
looks further down. Showing one in the other is a reprojection problem, and
this module is the whole of it.

The trick is that both pictures are of the same thing: the ground. A road
mask is a statement about where the ground is drivable, and the planned path
is a curve lying on it, so every pixel worth transferring sits on one plane
in vehicle space. A plane seen by two pinhole cameras is related by a
homography, which is exact, is a single 3x3 matrix, and - because both mounts
are rigidly bolted to the vehicle - never changes while the vehicle runs. It
is built once at start-up and applied with one ``warpPerspective`` per drawn
frame.

What this buys is the point of the split rig: the large view costs one
image warp, **not** a second segmentation. No model runs on the display
camera's frames, here or anywhere else.

Pixels above the horizon have no ground point to transfer and would map to
nonsense, so they are excluded by a validity mask computed from the source
camera's rays at start-up.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel

logger = logging.getLogger("offroad_autonomy.visualization.projector")


def ground_plane_matrix(camera: CameraModel, ground_z: float) -> np.ndarray:
    """On the plane ``z = ground_z`` the projection ``K R (X - C)`` is linear
    in ``(x, y, 1)``, so it collapses to one 3x3 matrix."""
    k = camera.intrinsic_matrix
    rotation = np.asarray(camera.rotation, dtype=np.float64)
    centre = np.asarray(camera.position, dtype=np.float64).reshape(3, 1)

    # X = plane @ (x, y, 1), so K R X - K R C gives the matrix below.
    plane = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, float(ground_z)]],
        dtype=np.float64,
    )
    kr = k @ rotation
    return kr @ plane - (kr @ centre) @ np.array([[0.0, 0.0, 1.0]])


class GroundProjector:
    """Perception view -> display camera, via the flat ground plane.

    The overlay drifts where real terrain departs from the plane, which is
    acceptable for something whose only job is to be looked at.
    """

    def __init__(
        self,
        source: CameraModel,
        target: CameraModel,
        ground_z: float = 0.0,
        min_range_m: float = 0.5,
        max_range_m: float = 60.0,
    ) -> None:
        self.source = source
        self.target = target
        self.ground_z = float(ground_z)

        source_matrix = ground_plane_matrix(source, self.ground_z)
        target_matrix = ground_plane_matrix(target, self.ground_z)
        if abs(float(np.linalg.det(source_matrix))) < 1e-12:
            raise ValueError(f"{source.name} looks along the ground plane - no homography exists")
        self.homography = target_matrix @ np.linalg.inv(source_matrix)

        self.ground_valid = self._ground_rays(source, min_range_m, max_range_m)
        if not self.ground_valid.any():
            raise ValueError(f"{source.name} sees no ground plane at all")

        logger.info(
            "Display reprojection: %s %dx%d -> %s %dx%d, %.0f%% of the source "
            "view lands on the ground plane",
            source.name,
            source.width,
            source.height,
            target.name,
            target.width,
            target.height,
            100.0 * self.ground_valid.mean(),
        )

    def _ground_rays(
        self, camera: CameraModel, min_range_m: float, max_range_m: float
    ) -> np.ndarray:
        """Rays that reach the plane hundreds of metres away land on a handful
        of target pixels where the flat-ground assumption stopped holding long
        before, so they are excluded along with rays above the horizon."""
        uu, vv = camera.pixel_grid()
        rays = np.stack(
            [
                (uu - camera.cx) / camera.focal_px,
                (vv - camera.cy) / camera.focal_px,
                np.ones_like(uu),
            ],
            axis=-1,
        ) @ np.asarray(camera.rotation, dtype=np.float32)

        drop = rays[..., 2]
        height = float(camera.position[2]) - self.ground_z
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(drop < -1e-6, -height / drop, np.inf)
        hit = np.isfinite(t) & (t > 0.0)
        # Range along the ground, not along the ray, so the limits read in
        # the same units as every other distance in the stack. Rays that miss
        # are zeroed first: an infinite t times a zero component is a NaN,
        # and a NaN would quietly pass a comparison it should fail.
        scale = np.where(hit, t, 0.0)
        forward = np.hypot(rays[..., 0] * scale, rays[..., 1] * scale)
        return hit & (forward >= min_range_m) & (forward <= max_range_m)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        source = (self._match_source(mask) & self.ground_valid).astype(np.uint8)
        warped = cv2.warpPerspective(
            source,
            self.homography,
            (self.target.width, self.target.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped.astype(bool)

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Points that miss the ground or land behind the target camera are
        dropped, so a path running up to the horizon comes back shorter rather
        than folding back on itself."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return pts

        columns = np.clip(np.round(pts[:, 0]).astype(int), 0, self.source.width - 1)
        rows = np.clip(np.round(pts[:, 1]).astype(int), 0, self.source.height - 1)
        on_ground = self.ground_valid[rows, columns]

        homogeneous = np.hstack([pts, np.ones((len(pts), 1))]) @ self.homography.T
        depth = homogeneous[:, 2]
        keep = on_ground & (np.abs(depth) > 1e-9) & (depth > 0.0)
        if not keep.any():
            return np.empty((0, 2), dtype=np.float32)
        return (homogeneous[keep, :2] / depth[keep, None]).astype(np.float32)

    def _match_source(self, mask: np.ndarray) -> np.ndarray:
        shape = (self.source.height, self.source.width)
        if mask.shape[:2] == shape:
            return mask.astype(bool)
        resized = cv2.resize(
            mask.astype(np.uint8),
            (self.source.width, self.source.height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized.astype(bool)
