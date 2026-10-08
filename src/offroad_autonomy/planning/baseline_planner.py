"""Deliberately simple centreline extraction.

    road component -> centre of its run on each sampled row -> ground plane
    -> one quadratic right(forward) -> temporal blend -> back to pixels

No candidate scoring, no clearance channel, no Kalman prediction, no
straightening. It is the reference the advanced planner has to beat, and
the thing to fall back to whenever perception itself is in question.

Why the fit is done on the ground
---------------------------------
Far road occupies very few image rows. Rows are
sampled densely toward the top of the ROI (quadratic spacing), or
the lookahead region would hold one or two samples. And a quadratic in
*image* coordinates is the wrong model for a road seen in perspective - a
straight road is a line converging on the vanishing point and a curve bends
hyperbolically toward the horizon - so an image-space fit invents curvature
at the far end. On the ground a road is ``right = a + b*f + c*f^2`` to a
good approximation, so that is where it is fitted and smoothed.
"""

from __future__ import annotations

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PipelineConfig

_MAX_GAP_ROWS = 3
_OUTPUT_POINTS = 24
#: Bound the centreline's lateral step. A road widening can move its row
#: midpoint faster than this even when a continuous path remains inside it.
_MAX_LATERAL_SLOPE = 1.0
_SLACK_M = 0.25
#: Stones and segmentation specks must not split the road and move its
#: centre by half its width.
_MERGE_GAP_M = 0.6
_MIN_ON_ROAD = 0.9


class BaselinePlanner:
    def __init__(self, config: PipelineConfig, camera: CameraModel | None = None) -> None:
        self._n_samples = max(8, int(config.centerline_samples))
        self._min_points = max(4, int(round(self._n_samples * 0.3)))
        self._temporal_blend = float(np.clip(config.baseline_temporal_blend, 0.0, 1.0))
        self._max_shift_m = float(max(config.baseline_max_shift_m, 0.0))
        self._camera = camera or CameraModel(
            config.camera, config.preprocess_width, config.preprocess_height
        )
        self._prev: np.ndarray | None = None

    def centerline(self, component: np.ndarray, roi_top: int) -> np.ndarray:
        """``(N, 2)`` pixel points ordered far -> near, or empty."""
        h, w = component.shape[:2]
        bottom = min(h - 1, int(round(h * 0.95)))
        # Quadratic spacing: dense near the ROI top, where metres per row
        # explode, sparse near the bumper where they are tiny.
        s = np.linspace(0.0, 1.0, self._n_samples)
        rows = np.unique(np.round(roi_top + (bottom - roi_top) * s**2).astype(int))[::-1]

        x_prev = w / 2.0
        forward_pts: list[float] = []
        right_pts: list[float] = []
        gap = 0
        for y in rows:
            row_f, _, row_ok = self._camera.image_to_ground(
                np.array([[self._camera.cx, y]], dtype=np.float64)
            )
            if not row_ok[0]:
                break
            px_per_m = self._camera.focal_px / max(float(row_f[0]), 0.1)
            run = self._nearest_run(component[y], x_prev, _MERGE_GAP_M * px_per_m)
            accepted = run is not None
            if accepted:
                x = 0.5 * (run[0] + run[1])
                f, r, ok = self._camera.image_to_ground(np.array([[x, y]], dtype=np.float64))
                accepted = bool(ok[0])
                if accepted and forward_pts:
                    step_f = float(f[0]) - forward_pts[-1]
                    max_step = _MAX_LATERAL_SLOPE * step_f + _SLACK_M
                    accepted = abs(float(r[0]) - right_pts[-1]) <= max_step
                    if not accepted and step_f > 0:
                        # At a side opening the midpoint can swing across the
                        # road. Continue only if the previous ground lateral
                        # position is still visibly inside this same run.
                        continuation_x = float(
                            self._camera.ground_to_image(f, [right_pts[-1]])[0, 0]
                        )
                        continuation_col = int(round(continuation_x))
                        if run[0] <= continuation_col <= run[1] and component[y, continuation_col]:
                            bounded_r = np.clip(
                                float(r[0]), right_pts[-1] - max_step, right_pts[-1] + max_step
                            )
                            x = float(self._camera.ground_to_image(f, [bounded_r])[0, 0])
                            f, r, ok = self._camera.image_to_ground([[x, y]])
                            step_f = float(f[0]) - forward_pts[-1]
                            accepted = (
                                bool(ok[0])
                                and step_f > 0
                                and run[0] <= x <= run[1]
                                and component[y, int(round(x))]
                                and abs(float(r[0]) - right_pts[-1])
                                <= _MAX_LATERAL_SLOPE * step_f + _SLACK_M + 1e-6
                            )
            if not accepted:
                gap += 1
                if forward_pts and gap >= _MAX_GAP_ROWS:
                    break
                continue
            gap = 0
            x_prev = x
            forward_pts.append(float(f[0]))
            right_pts.append(float(r[0]))

        if len(forward_pts) < self._min_points:
            return np.empty((0, 2), dtype=np.float32)
        forward, right = np.asarray(forward_pts), np.asarray(right_pts)

        degree = 1
        if len(forward) >= 6:
            degree = 2
        coeffs = np.polyfit(forward, right, deg=degree)
        # Padded to [c, b, a] so the temporal blend can mix fits of either degree.
        coeffs = np.pad(coeffs, (3 - len(coeffs), 0))

        f_out = np.linspace(float(forward.min()), float(forward.max()), _OUTPUT_POINTS)
        r_out = np.polyval(coeffs, f_out)
        pixels = self._camera.ground_to_image(f_out, r_out)
        if self._temporal_blend > 0.0 and self._prev is not None:
            # Blend at equal distances ahead: keeps the road's curvature and
            # damps frame-to-frame jitter. The cap is in metres, so it is as
            # strict at 12 m as at 2 m - in pixels it was 30x looser far out.
            prev = np.polyval(self._prev, f_out)
            blended = (1.0 - self._temporal_blend) * r_out + self._temporal_blend * prev
            if self._max_shift_m > 0.0:
                blended = prev + np.clip(blended - prev, -self._max_shift_m, self._max_shift_m)
            blended_px = self._camera.ground_to_image(f_out, blended)
            # The previous path is held in the vehicle frame, so when the car
            # yaws it swings with it. Smoothing may soften the path but never
            # move it off the road seen now; if it would, take this frame as is.
            if self._on_road_fraction(component, blended_px) >= _MIN_ON_ROAD:
                r_out, pixels = blended, blended_px
                coeffs = np.polyfit(f_out, r_out, deg=2)
        self._prev = coeffs

        pixels[:, 0] = np.clip(pixels[:, 0], 0.0, w - 1.0)
        return pixels[::-1].copy()

    @staticmethod
    def _on_road_fraction(component: np.ndarray, pixels: np.ndarray) -> float:
        h, w = component.shape[:2]
        xs = np.round(pixels[:, 0]).astype(int)
        ys = np.round(pixels[:, 1]).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not inside.any():
            return 0.0
        return float(component[ys[inside], xs[inside]].mean()) * float(inside.mean())

    @staticmethod
    def _nearest_run(
        row: np.ndarray, x_ref: float, merge_gap_px: float = 1.0
    ) -> tuple[int, int] | None:
        """Following the nearest run keeps the walk on the same road instead
        of jumping to whichever patch is widest."""
        cols = np.flatnonzero(row)
        if len(cols) == 0:
            return None
        splits = np.flatnonzero(np.diff(cols) > max(1.0, merge_gap_px)) + 1
        runs = [(int(g[0]), int(g[-1])) for g in np.split(cols, splits)]

        def distance(run: tuple[int, int]) -> float:
            start, end = run
            if start <= x_ref <= end:
                return 0.0
            return min(abs(start - x_ref), abs(end - x_ref))

        return min(runs, key=distance)

    def reset(self) -> None:
        self._prev = None
