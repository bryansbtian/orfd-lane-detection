"""Stanley lateral controller with proportional longitudinal control.

Converts a planned path into steering, throttle, and brake commands
suitable for sending to BeamNG.

Stanley law
-----------
    δ = heading_error + arctan(k · cross_track_error / (v + k_soft))

The cross-track error is computed from the ``PathPlan`` centerline
(lateral offset at the bottom of the image frame, normalised to [-1, 1]).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from offroad_autonomy.types import ControlCommand, PathPlan, PipelineConfig, VehicleState

logger = logging.getLogger("offroad_autonomy.control")


class StanleyController:
    """Image-space Stanley controller for BeamNG."""

    def __init__(self, config: PipelineConfig) -> None:
        self._k = config.stanley_gain_k
        self._k_soft = config.stanley_softening
        self._target_speed = config.target_speed_mps
        self._max_throttle = config.max_throttle
        self._max_brake = config.max_brake
        self._speed_kp = config.speed_kp
        self._frame_w: float = config.preprocess_width
        self._prev_steer: float = 0.0

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        """Compute actuator commands from the path plan and vehicle state."""
        steering = self._lateral_control(plan, state)
        throttle, brake = self._longitudinal_control(state, plan.kalman_active)

        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
        )

    def _lateral_control(self, plan: PathPlan, state: VehicleState) -> float:
        # When centerline is lost (mask dropout), hold last good steering
        # with gentle decay toward zero.  This keeps the car turning through
        # curves where perception briefly fails instead of snapping to 0.
        if len(plan.centerline) < 2:
            # Decay factor: retain 95% per frame (~15 fps → halves in ~1 s)
            self._prev_steer *= 0.95
            logger.debug("No centerline — holding steer %.3f", self._prev_steer)
            return self._prev_steer

        n = len(plan.centerline)
        ego_x = self._frame_w / 2.0

        # Lookahead target: ~50% up the centerline (far = early curve detection)
        look_idx = max(0, n - 1 - int(n * 0.5))
        target = plan.centerline[look_idx]
        target_x, target_y = target[0], target[1]

        bottom = plan.centerline[-1]
        bottom_y = bottom[1]

        dx = target_x - ego_x
        dy = bottom_y - target_y
        if dy < 1.0:
            self._prev_steer *= 0.95
            return self._prev_steer

        steer = math.atan2(dx, dy) * 2.0  # gain — far lookahead gives small angles
        self._prev_steer = steer
        return steer

    def _longitudinal_control(
        self, state: VehicleState, kalman_active: bool = False,
    ) -> tuple[float, float]:
        # When mask is lost (Kalman coasting), slow down — we're flying blind.
        # Also slow down proportionally to how hard we're steering (curves).
        target = self._target_speed

        if kalman_active:
            # Reduce target to 40% of normal — coast cautiously
            target *= 0.4

        # Curvature braking: steering near ±1 → target drops to 60%
        steer_mag = abs(self._prev_steer)
        curve_factor = 1.0 - 0.4 * min(steer_mag / 0.5, 1.0)
        target *= curve_factor

        speed_err = target - state.speed_mps

        if speed_err > 0:
            throttle = min(self._speed_kp * speed_err, self._max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self._speed_kp * abs(speed_err), self._max_brake)

        return throttle, brake
