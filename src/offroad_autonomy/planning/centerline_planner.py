"""Centerline path planning with Kalman-filter fallback.

Extracts a drivable centerline from the binary traversable-road mask and
smooths it with a linear Kalman filter.  When the mask is degraded or
empty the filter predicts forward using a constant-curvature motion model,
providing graceful fallback instead of zero output.

The state vector and transition matrix are structured so an Extended
Kalman Filter (EKF) can replace the linear filter later without changing
the rest of the module's interface.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from offroad_autonomy.types import PathPlan, PipelineConfig, StabilizedResult

logger = logging.getLogger("offroad_autonomy.planning")

_REF_DT = 1.0 / 15.0  # seconds — Kalman dynamics tuned at 15 fps
_CURVATURE_DECAY_REF = 0.8  # per-step decay at reference rate


class _KalmanTracker:
    """Linear Kalman filter over [lateral_offset, heading, curvature].

    State vector
    ------------
    x[0]  lateral_offset   pixels from frame centre (positive = road right)
    x[1]  heading          road angle relative to forward (radians)
    x[2]  curvature        rate of change of heading (rad/step at ref rate)

    Transition model (constant curvature, time-scaled)
    --------------------------------------------------
    offset'    = offset + heading · dt_ratio
    heading'   = heading + curvature · dt_ratio
    curvature' = curvature · decay^dt_ratio
    """

    DIM_X = 3
    DIM_Z = 2

    def __init__(self, q: float, r: float) -> None:
        self.x = np.zeros(self.DIM_X)
        self.P = np.eye(self.DIM_X) * 100.0

        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self._q = q
        # Heading measurement is noisier than lateral — give it more slack
        self.R = np.diag([r, r * 5.0])

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def predict(self, dt_ratio: float = 1.0) -> np.ndarray:
        # Build F scaled by dt_ratio (1.0 = reference timestep)
        F = np.array([
            [1.0, dt_ratio, 0.0],
            [0.0, 1.0,      dt_ratio],
            [0.0, 0.0,      _CURVATURE_DECAY_REF ** dt_ratio],
        ])
        # Process noise scales with time step
        Q = np.eye(self.DIM_X) * self._q * dt_ratio

        self.x = F @ self.x
        self.x[1] = self._wrap_angle(self.x[1])
        self.P = F @ self.P @ F.T + Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z - self.H @ self.x
        y[1] = self._wrap_angle(y[1])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[1] = self._wrap_angle(self.x[1])
        self.P = (np.eye(self.DIM_X) - K @ self.H) @ self.P
        return self.x.copy()


class CenterlinePlanner:
    """Extract a centerline from the road mask and track it with a Kalman filter."""

    def __init__(self, config: PipelineConfig) -> None:
        self._n_samples = config.centerline_samples
        self._min_road_px = config.min_road_pixels
        self._max_misses = config.fallback_after_n_misses

        self._kf = _KalmanTracker(
            q=config.kalman_process_noise,
            r=config.kalman_measurement_noise,
        )
        self._consecutive_misses = 0
        self._last_time: float = 0.0
        # Per-row width history for known-width correction.
        # Stores recent unclipped widths bucketed by image row zone (near/mid/far)
        # so irregular road shapes are handled — each zone tracks its own width.
        self._width_history: dict[str, list[float]] = {"near": [], "mid": [], "far": []}
        self._width_history_max = 30  # max samples per zone

    def _dt_ratio(self) -> float:
        """Compute dt / reference_dt for time-scaling Kalman dynamics."""
        now = time.perf_counter()
        dt = now - self._last_time if self._last_time > 0 else _REF_DT
        dt = min(dt, 0.5)
        self._last_time = now
        return dt / _REF_DT

    def plan(self, stabilized: StabilizedResult) -> PathPlan:
        """Compute a centered path through the traversable region."""
        dt_ratio = self._dt_ratio()
        mask = stabilized.mask
        h, w = mask.shape[:2]
        road_px = int(mask.sum())

        if road_px < self._min_road_px:
            return self._fallback(h, w, dt_ratio)

        centerline, road_width, corrected_rows = self._extract_centerline(mask, h, w)

        if len(centerline) < 2:
            return self._fallback(h, w, dt_ratio)

        lateral_offset = centerline[-1, 0] - w / 2.0
        heading = self._estimate_heading(centerline)

        z = np.array([lateral_offset, heading])
        self._kf.predict(dt_ratio)
        state = self._kf.update(z)
        self._consecutive_misses = 0

        return PathPlan(
            centerline=centerline,
            heading_rad=float(state[1]),
            curvature=float(state[2]),
            road_width_px=road_width,
            kalman_active=False,
            raw_heading_rad=heading,
            raw_lateral_offset_px=lateral_offset,
            width_corrected_rows=corrected_rows,
        )

    def _fallback(self, h: int, w: int, dt_ratio: float = 1.0) -> PathPlan:
        """Use Kalman prediction when the mask is absent or degraded."""
        self._consecutive_misses += 1
        state = self._kf.predict(dt_ratio)

        if self._consecutive_misses > self._max_misses:
            logger.warning(
                "No valid road mask for %d frames — Kalman coasting",
                self._consecutive_misses,
            )

        cx = w / 2.0 + state[0]
        fallback_pts = np.array([
            [cx, h * 0.9],
            [cx, h * 0.5],
            [cx, h * 0.1],
        ])

        return PathPlan(
            centerline=fallback_pts,
            heading_rad=float(state[1]),
            curvature=float(state[2]),
            road_width_px=0.0,
            kalman_active=True,
        )

    def _extract_centerline(
        self,
        mask: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, float, int]:
        """Scan horizontal rows to find the road midpoint at each height.

        When one road edge is clipped by the camera frame (left==0 or
        right==W-1), the naive midpoint is biased toward the visible side.
        We correct this using a per-zone width history learned from frames
        where both edges are fully visible (known-width prior — see
        notes.txt for paper references).  Three zones (near/mid/far)
        handle irregular road shapes where width varies with distance.
        """
        row_indices = np.linspace(
            int(h * 0.1), int(h * 0.95), self._n_samples, dtype=int,
        )

        EDGE_MARGIN = 2  # pixels — treat edges within this margin as clipped
        # Zone boundaries as fractions of image height
        mid_boundary = h * 0.4
        far_boundary = h * 0.7

        points: list[tuple[float, float]] = []
        widths: list[float] = []
        corrected_count = 0

        for y in row_indices:
            row = mask[y, :]
            cols = np.where(row)[0]

            if len(cols) < 2:
                continue

            left = float(cols[0])
            right = float(cols[-1])
            raw_width = right - left

            left_clipped = left <= EDGE_MARGIN
            right_clipped = right >= (w - 1 - EDGE_MARGIN)

            # Determine which zone this row belongs to
            if y >= far_boundary:
                zone = "near"
            elif y >= mid_boundary:
                zone = "mid"
            else:
                zone = "far"

            # Track width only from rows where both edges are fully visible
            if not left_clipped and not right_clipped:
                hist = self._width_history[zone]
                hist.append(raw_width)
                if len(hist) > self._width_history_max:
                    hist.pop(0)

            # Get the median width estimate for this zone
            zone_width = 0.0
            if self._width_history[zone]:
                zone_width = float(np.median(self._width_history[zone]))

            # Correct centerline using known-width prior when an edge is clipped
            if zone_width > 0:
                if left_clipped and not right_clipped:
                    cx = right - zone_width / 2.0
                    corrected_count += 1
                elif right_clipped and not left_clipped:
                    cx = left + zone_width / 2.0
                    corrected_count += 1
                else:
                    cx = (left + right) / 2.0
            else:
                cx = (left + right) / 2.0

            points.append((cx, float(y)))
            widths.append(raw_width)

        if not points:
            return np.empty((0, 2)), 0.0, 0

        centerline = np.array(points)
        avg_width = float(np.mean(widths)) if widths else 0.0
        return centerline, avg_width, corrected_count

    @staticmethod
    def _estimate_heading(centerline: np.ndarray) -> float:
        """Compute heading as lookahead angle from ego to a point ~40% up.

        Using the bottom-third slope is extremely noisy — small pixel
        shifts at nearby points cause huge angle changes.  A lookahead
        angle to a distant point is much more stable because distant
        points shift less between frames.
        """
        n = len(centerline)
        if n < 2:
            return 0.0

        # centerline is ordered bottom-to-top: [-1] is near, [0] is far
        pt_near = centerline[-1]
        # Look ~40% of the way up the centerline
        lookahead_idx = max(0, n - 1 - int(n * 0.4))
        pt_far = centerline[lookahead_idx]

        dx = pt_far[0] - pt_near[0]
        dy = pt_near[1] - pt_far[1]

        if dy < 1e-6:
            return 0.0

        return float(math.atan2(dx, dy))

    def reset(self) -> None:
        """Clear Kalman state (e.g. on map reload)."""
        self._kf = _KalmanTracker(
            q=self._kf._q,
            r=self._kf.R[0, 0],
        )
        self._consecutive_misses = 0
        self._last_time = 0.0
        self._width_history = {"near": [], "mid": [], "far": []}
