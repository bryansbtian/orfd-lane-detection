"""Compare local paths at equal forward distances after ego-motion compensation."""

import numpy as np


def transform_path(points, previous_pose, current_pose):
    """Camera-ground [forward, right] points in the current vehicle frame."""
    old_forward = np.array([previous_pose.forward_x, previous_pose.forward_y])
    old_right = np.array([previous_pose.forward_y, -previous_pose.forward_x])
    world = (
        np.array([previous_pose.x, previous_pose.y])
        + points[:, :1] * old_forward
        + points[:, 1:] * old_right
    )
    offset = world - [current_pose.x, current_pose.y]
    forward = np.array([current_pose.forward_x, current_pose.forward_y])
    right = np.array([current_pose.forward_y, -current_pose.forward_x])
    return np.column_stack((offset @ forward, offset @ right))


def trajectory_distance(candidate, reference):
    """RMS lateral separation in metres over the shared forward extent.

    No overlap means no continuity preference, rather than extrapolating an
    old path into ground it never covered. Sampling density does not affect
    the score.
    """
    paths = []
    for points in (candidate, reference):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            return 0.0
        points = points[np.isfinite(points).all(axis=1)]
        forward, indices = np.unique(points[:, 0], return_index=True)
        if len(forward) < 2:
            return 0.0
        paths.append((forward, points[indices, 1]))
    lo = max(0.0, paths[0][0][0], paths[1][0][0])
    hi = min(paths[0][0][-1], paths[1][0][-1])
    if hi - lo < 0.5:
        return 0.0
    samples = np.linspace(lo, hi, 24)
    difference = np.interp(samples, *paths[0]) - np.interp(samples, *paths[1])
    return float(np.sqrt(np.mean(difference**2)))
