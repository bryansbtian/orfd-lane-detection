"""Validate the complete output polyline against current RGB road evidence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel


@dataclass(frozen=True)
class CorridorCheck:
    ok: bool
    reason: str
    min_clearance_px: float = 0.0


def dense_path(centerline: np.ndarray) -> np.ndarray:
    """Sample at half-pixel steps, including the space between waypoints."""
    points = np.asarray(centerline, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return np.empty((0, 2))
    if not np.isfinite(points).all():
        return np.empty((0, 2))
    samples = []
    for start, end in pairwise(points):
        count = max(2, int(np.ceil(np.max(np.abs(end - start)) * 2)) + 1)
        samples.append(np.linspace(start, end, count))
    return np.concatenate(samples)


def validate_corridor(
    centerline: np.ndarray,
    component: np.ndarray,
    min_clearance_px: float = 1.0,
    camera: CameraModel | None = None,
    half_width_m: float = 0.0,
    footprint_component: np.ndarray | None = None,
) -> CorridorCheck:
    points = np.asarray(centerline)
    h, w = component.shape
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        return CorridorCheck(False, "non-finite or missing path")
    if np.any(points < 0) or np.any(points[:, 0] > w - 1) or np.any(points[:, 1] > h - 1):
        return CorridorCheck(False, "path outside camera frame")
    samples = dense_path(points)
    xs, ys = np.rint(samples).astype(int).T
    if not component[ys, xs].all():
        return CorridorCheck(False, "path leaves current road evidence")
    # Zero padding treats frame edges as unknown, rather than infinite road.
    distance = cv2.distanceTransform(np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 5)
    clearance = float(distance[ys + 1, xs + 1].min())
    if clearance < min_clearance_px:
        return CorridorCheck(False, "insufficient image boundary clearance", clearance)
    if camera is not None and half_width_m > 0:
        forward, right, valid = camera.image_to_ground(points)
        if not valid.all():
            return CorridorCheck(False, "corridor projection is unavailable", clearance)
        df, dr = np.gradient(forward), np.gradient(right)
        length = np.hypot(df, dr)
        if np.any(length < 1e-6):
            return CorridorCheck(False, "degenerate corridor direction", clearance)
        # The width is normal to the path, so a bend cannot squeeze a vehicle
        # through a gap simply by measuring its width along an image row.
        normal_f, normal_r = -dr / length * half_width_m, df / length * half_width_m
        left = camera.ground_to_image(forward + normal_f, right + normal_r)
        right_edge = camera.ground_to_image(forward - normal_f, right - normal_r)
        polygon = np.vstack([left, right_edge[::-1]])
        if (
            not np.isfinite(polygon).all()
            or np.any(polygon < 0)
            or np.any(polygon[:, 0] > w - 1)
            or np.any(polygon[:, 1] > h - 1)
        ):
            return CorridorCheck(
                False, "vehicle-width corridor extends outside camera frame", clearance
            )
        footprint = np.zeros((h, w), np.uint8)
        cv2.fillPoly(footprint, [np.rint(polygon).astype(np.int32)], 1)
        evidence = component
        if footprint_component is not None:
            evidence = footprint_component
        if np.any(footprint.astype(bool) & ~evidence):
            return CorridorCheck(
                False, "vehicle-width corridor leaves current road evidence", clearance
            )
    return CorridorCheck(True, "", clearance)
