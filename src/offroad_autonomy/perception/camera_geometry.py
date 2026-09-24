"""Pinhole camera models and rigid transforms for the front stereo pair.

Two coordinate frames are in play.

**Vehicle space** is BeamNG's: ``+X`` left, ``-Y`` forward, ``+Z`` up.  This
is the frame every :class:`~offroad_autonomy.types.CameraSpec` is written in,
and the frame terrain reasoning happens in.

**Camera space** is the usual vision convention: ``+X`` right, ``+Y`` down,
``+Z`` along the viewing direction.  Projection is an exact pinhole - BeamNG
renders a rectilinear image with no lens distortion, so no undistortion step
is needed for the simulated lens.  Rectification still accepts distortion
coefficients so a real, calibrated GMSL2 unit drops in through the config.
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
            logger.warning(
                "%s: working aspect %.4f differs from native %.4f - a "
                "non-uniform resize breaks the square-pixel assumption and "
                "the recovered geometry will be skewed",
                spec.name,
                work_aspect,
                native_aspect,
            )

        half_fov = math.radians(spec.fov_x_deg) / 2.0
        if not 0.0 < half_fov < math.pi / 2.0:
            raise ValueError(f"{spec.name}: fov_x_deg must be in (0, 180)")

        self.scale = self.width / spec.width
        self.focal_px = (self.width / 2.0) / math.tan(half_fov)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

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
        return math.degrees(2.0 * math.atan(self.cx / self.focal_px))

    @property
    def vertical_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.cy / self.focal_px))

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
        y = (np.asarray(v, dtype=np.float32) - np.float32(self.cy)) * depth / f
        return np.stack([x, y, np.asarray(depth, dtype=np.float32)], axis=-1)

    def project(self, points_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(u, v, z)``; callers must drop entries with ``z <= 0``, which sit
        behind the image plane and would otherwise project mirrored."""
        pts = np.asarray(points_camera, dtype=np.float32)
        z = pts[..., 2]
        safe_z = np.where(np.abs(z) < 1e-6, np.float32(1e-6), z)
        u = np.float32(self.focal_px) * pts[..., 0] / safe_z + np.float32(self.cx)
        v = np.float32(self.focal_px) * pts[..., 1] / safe_z + np.float32(self.cy)
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
            [(pts[:, 0] - self.cx) / f, (pts[:, 1] - self.cy) / f, np.ones(len(pts))], axis=1
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
                [0.0, self.focal_px, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_intrinsics(
        cls,
        base: "CameraModel",
        name: str,
        projection: np.ndarray,
        rotation_to_new: np.ndarray,
        width: int,
        height: int,
    ) -> "CameraModel":
        """A virtual camera sharing ``base``'s centre with new K and attitude.

        This is what a rectified camera is: the same optical centre, rotated
        by ``R1`` and re-projected with ``P1``. Modelling it as an ordinary
        ``CameraModel`` means backprojection and vehicle-space transforms
        work on rectified pixels with no special cases.
        """
        model = cls.__new__(cls)
        model.spec = base.spec
        model.sensor = base.sensor
        model.name = name
        model.width = int(width)
        model.height = int(height)
        model.scale = base.scale
        projection = np.asarray(projection, dtype=np.float64)
        # stereoRectify returns equal fx and fy for an ideal pinhole pair.
        model.focal_px = float(projection[0, 0])
        model.cx = float(projection[0, 2])
        model.cy = float(projection[1, 2])
        model.position = base.position.copy()
        model.rotation = np.asarray(rotation_to_new, dtype=np.float64) @ base.rotation
        return model


class StereoRig:
    """The front stereo pair: two mounts and the rigid transform between them.

    The pair no longer has to be rectified by construction - ``StereoRectifier``
    rectifies it properly - but it still has to be a sensible stereo pair. The
    constructor refuses configurations that no rectification can rescue:
    swapped sides, a zero baseline, or optical axes so far apart that the two
    views barely overlap.
    """

    #: Beyond this the views overlap too little to match.
    MAX_AXIS_ANGLE_DEG = 20.0

    def __init__(self, left: CameraModel, right: CameraModel) -> None:
        self.left = left
        self.right = right

        if abs(left.focal_px - right.focal_px) > 1e-6 or (
            left.width,
            left.height,
        ) != (right.width, right.height):
            raise ValueError(
                "Stereo cameras must share resolution and field of view; check "
                f"'{left.name}' and '{right.name}' use the same sensor."
            )

        cos_axes = float(np.clip(left.rotation[2] @ right.rotation[2], -1.0, 1.0))
        self.axis_angle_deg = math.degrees(math.acos(cos_axes))
        if self.axis_angle_deg > self.MAX_AXIS_ANGLE_DEG:
            raise ValueError(
                f"Stereo optical axes differ by {self.axis_angle_deg:.1f} deg "
                f"(limit {self.MAX_AXIS_ANGLE_DEG:.0f}); the views would barely "
                "overlap. Reduce stereo_rig.toe_out_deg."
            )

        self.rotation, self.translation = relative_pose(left, right)
        self.baseline_m = float(np.linalg.norm(self.translation))
        if self.baseline_m < 1e-3:
            raise ValueError("Stereo baseline is effectively zero")

        # The right camera must sit to the right of the left one, in the left
        # camera's own axes (OpenCV: T = -R @ C_right_in_left, so T_x < 0).
        offset = left.from_vehicle(right.position.reshape(1, 3))[0]
        if float(offset[0]) < 0.0:
            raise ValueError(
                f"'{left.name}' is to the right of '{right.name}'. Swap the "
                "left/right camera assignment - on this vehicle the +X axis "
                "points to the vehicle's left."
            )
        self.vertical_offset_m = float(offset[1])

    @property
    def focal_px(self) -> float:
        return self.left.focal_px

    def depth_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        disp = np.asarray(disparity, dtype=np.float32)
        safe = np.where(disp > 1e-3, disp, np.float32(np.nan))
        return np.float32(self.left.focal_px * self.baseline_m) / safe

    def disparity_for_depth(self, depth_m: float) -> float:
        if depth_m <= 0.0:
            return float("inf")
        return float(self.left.focal_px * self.baseline_m / depth_m)

    def disparity_range_for(self, min_depth_m: float) -> int:
        """Smallest SGBM search range that reaches ``min_depth_m``.

        The closest depth a pair can resolve is ``f * B / numDisparities``, so
        inverting that and rounding up to the multiple of 16 OpenCV requires
        gives the cheapest range that still covers the near field.
        """
        if min_depth_m <= 0.0:
            raise ValueError("min_depth_m must be positive")
        needed = self.disparity_for_depth(min_depth_m)
        return max(16, int(math.ceil(needed / 16.0)) * 16)

    def describe(self) -> str:
        return (
            f"{self.left.sensor.model} pair  baseline={self.baseline_m:.3f} m  "
            f"f={self.left.focal_px:.1f} px @{self.left.width}x{self.left.height}  "
            f"hfov={self.left.horizontal_fov_deg:.0f} deg  "
            f"axes {self.axis_angle_deg:.1f} deg apart  "
            f"disparity@10m={self.disparity_for_depth(10.0):.1f} px"
        )


def relative_pose(left: CameraModel, right: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """``(R, T)`` with ``X_right = R @ X_left + T``, the OpenCV convention.

    Both come straight from the mounts: ``R = R_r R_l^T`` and
    ``T = R_r (C_l - C_r)``. For a parallel pair ``R`` is the identity and
    ``T`` is ``(-baseline, 0, 0)``.
    """
    rotation = right.rotation @ left.rotation.T
    translation = right.rotation @ (left.position - right.position)
    return rotation, translation


def build_camera_models(
    config,
    work_width: int | None = None,
    work_height: int | None = None,
) -> tuple[CameraModel, CameraModel]:
    """The only door into stereo, and it has exactly two cameras: the display
    camera cannot reach the compute path through here."""
    if work_width is None:
        work_width = config.stereo_width
    if work_height is None:
        work_height = config.stereo_height
    width = int(work_width)
    height = int(work_height)
    left = CameraModel(config.left_camera, width, height)
    right = CameraModel(config.right_camera, width, height)
    return left, right
