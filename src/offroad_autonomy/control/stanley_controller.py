"""Lookahead lateral controller with proportional longitudinal control.

Pure pursuit to the centerline at mid-screen: compute the angle from
ego (bottom-center) to the road center at ~50% image height, steer
proportionally.  Hold last steering on mask dropout.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from offroad_autonomy.types import ControlCommand, PathPlan, PipelineConfig, VehicleState

logger = logging.getLogger("offroad_autonomy.control")

_REF_DT = 1.0 / 15.0
_STEER_HOLD_HALFLIFE = 1.0  # seconds


class StanleyController:
    """Pure-pursuit lateral controller for BeamNG."""

    def __init__(self, config: PipelineConfig) -> None:
        self._gain = config.steer_kp
        self._target_speed = config.target_speed_mps
        self._max_throttle = config.max_throttle
        self._max_brake = config.max_brake
        self._speed_kp = config.speed_kp
        self._frame_w: float = config.preprocess_width
        self._frame_h: float = config.preprocess_height
        self._prev_steer: float = 0.0
        self._last_time: float = 0.0

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        now = time.perf_counter()
        dt = now - self._last_time if self._last_time > 0 else _REF_DT
        dt = min(dt, 0.5)
        self._last_time = now

        steering = self._lateral_control(plan, dt)
        throttle, brake = self._longitudinal_control(state, plan.kalman_active)

        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
        )

    def _lateral_control(self, plan: PathPlan, dt: float) -> float:
        decay = 0.5 ** (dt / _STEER_HOLD_HALFLIFE)

        if len(plan.centerline) < 2:
            self._prev_steer *= decay
            return self._prev_steer

        # Find the centerline point closest to mid-screen height
        mid_y = self._frame_h * 0.5
        dists = np.abs(plan.centerline[:, 1] - mid_y)
        idx = int(np.argmin(dists))
        target_x = plan.centerline[idx, 0]

        # Ego = bottom-center of frame
        ego_x = self._frame_w / 2.0
        ego_y = self._frame_h

        dx = target_x - ego_x
        dy = ego_y - mid_y  # always positive (looking up)
        if dy < 1.0:
            self._prev_steer *= decay
            return self._prev_steer

        # Steer = gain * angle to target
        steer = self._gain * math.atan2(dx, dy)
        self._prev_steer = steer
        return steer

    def _longitudinal_control(
        self, state: VehicleState, kalman_active: bool = False,
    ) -> tuple[float, float]:
        target = self._target_speed

        if kalman_active:
            target *= 0.4

        # Curvature braking: slow down when steering hard
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
