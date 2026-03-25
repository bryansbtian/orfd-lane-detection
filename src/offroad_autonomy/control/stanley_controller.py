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

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        """Compute actuator commands from the path plan and vehicle state."""
        steering = self._lateral_control(plan, state)
        throttle, brake = self._longitudinal_control(state)

        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
        )

    def _lateral_control(self, plan: PathPlan, state: VehicleState) -> float:
        heading_err = plan.heading_rad

        if len(plan.centerline) > 0:
            cx_bottom = plan.centerline[-1, 0]
            cte = (cx_bottom - self._frame_w / 2.0) / (self._frame_w / 2.0)
        else:
            cte = 0.0

        speed = max(state.speed_mps, 0.1)
        stanley_term = math.atan2(self._k * cte, speed + self._k_soft)

        return heading_err + stanley_term

    def _longitudinal_control(self, state: VehicleState) -> tuple[float, float]:
        speed_err = self._target_speed - state.speed_mps

        if speed_err > 0:
            throttle = min(self._speed_kp * speed_err, self._max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self._speed_kp * abs(speed_err), self._max_brake)

        return throttle, brake
