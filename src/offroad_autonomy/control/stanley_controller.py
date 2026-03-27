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

_MPS_PER_MPH = 0.44704


class StanleyController:
    """Image-space Stanley controller for BeamNG."""

    def __init__(self, config: PipelineConfig) -> None:
        self._k = config.stanley_gain_k
        self._k_soft = config.stanley_softening
        self._heading_gain = config.stanley_heading_gain
        self._lookahead_fraction = float(np.clip(config.stanley_lookahead_fraction, 0.0, 0.9))
        self._near_path_weight = float(np.clip(config.stanley_near_path_weight, 0.0, 1.0))
        self._steering_alpha = float(np.clip(config.steering_ema_alpha, 0.0, 1.0))
        self._max_steering_delta = float(max(config.max_steering_delta, 0.0))
        self._steering_deadband = float(max(config.steering_deadband, 0.0))
        self._target_speed = min(config.target_speed_mph, config.speed_limit_mph) * _MPS_PER_MPH
        self._speed_limit = config.speed_limit_mph * _MPS_PER_MPH
        self._min_turn_speed = min(config.min_turn_speed_mph, config.speed_limit_mph) * _MPS_PER_MPH
        self._curve_speed_gain = float(max(config.curve_speed_gain, 0.0))
        self._heading_speed_gain = float(max(config.heading_speed_gain, 0.0))
        self._max_throttle = config.max_throttle
        self._max_brake = config.max_brake
        self._speed_kp = config.speed_kp
        self._frame_w: float = config.preprocess_width
        self._prev_steering = 0.0

        if config.target_speed_mph > config.speed_limit_mph:
            logger.warning(
                "Target speed %.1f mph exceeds limit %.1f mph - clamping to the speed limit",
                config.target_speed_mph,
                config.speed_limit_mph,
            )

    def compute(self, plan: PathPlan, state: VehicleState) -> ControlCommand:
        """Compute actuator commands from the path plan and vehicle state."""
        steering = self._smooth_steering(self._lateral_control(plan, state))
        throttle, brake = self._longitudinal_control(plan, state)

        return ControlCommand(
            steering=float(np.clip(steering, -1.0, 1.0)),
            throttle=throttle,
            brake=brake,
        )

    def _lateral_control(self, plan: PathPlan, state: VehicleState) -> float:
        if len(plan.centerline) < 2:
            cte = 0.0
            heading_err = self._heading_gain * plan.heading_rad
        else:
            target_idx = self._target_index(plan.centerline)
            x_near = float(plan.centerline[-1, 0])
            x_target = float(plan.centerline[target_idx, 0])
            reference_x = self._near_path_weight * x_near + (1.0 - self._near_path_weight) * x_target
            cte = (reference_x - self._frame_w / 2.0) / (self._frame_w / 2.0)
            cte = self._apply_deadband(cte, self._steering_deadband)
            heading_err = self._heading_gain * self._estimate_path_heading(plan, target_idx)

        speed = max(state.speed_mps, 0.1)
        stanley_term = math.atan2(self._k * cte, speed + self._k_soft)

        return heading_err + stanley_term

    def _smooth_steering(self, steering_cmd: float) -> float:
        """Low-pass and rate-limit steering to avoid oscillation."""
        blended = self._prev_steering + self._steering_alpha * (steering_cmd - self._prev_steering)
        delta = float(np.clip(
            blended - self._prev_steering,
            -self._max_steering_delta,
            self._max_steering_delta,
        ))
        self._prev_steering = float(np.clip(self._prev_steering + delta, -1.0, 1.0))
        return self._prev_steering

    @staticmethod
    def _apply_deadband(value: float, threshold: float) -> float:
        """Zero out tiny path offsets that only create chatter."""
        if abs(value) <= threshold:
            return 0.0
        return value - math.copysign(threshold, value)

    def _target_index(self, centerline: np.ndarray) -> int:
        count = len(centerline)
        lookahead_steps = max(1, int(round((count - 1) * self._lookahead_fraction)))
        return max(0, count - 1 - lookahead_steps)

    @staticmethod
    def _estimate_path_heading(plan: PathPlan, target_idx: int) -> float:
        centerline = plan.centerline
        if len(centerline) < 2:
            return plan.heading_rad

        idx_near = min(len(centerline) - 1, target_idx + 1)
        idx_far = max(0, target_idx - 1)
        if idx_near == idx_far:
            return plan.heading_rad

        pt_near = centerline[idx_near]
        pt_far = centerline[idx_far]
        dx = pt_far[0] - pt_near[0]
        dy = pt_near[1] - pt_far[1]
        if dy < 1e-6:
            return plan.heading_rad

        return float(math.atan2(dx, dy))

    def _longitudinal_control(self, plan: PathPlan, state: VehicleState) -> tuple[float, float]:
        if state.speed_mps > self._speed_limit:
            overspeed = state.speed_mps - self._speed_limit
            return 0.0, min(self._speed_kp * overspeed, self._max_brake)

        target_speed = self._target_speed
        turn_severity = max(
            self._curve_speed_gain * abs(plan.curvature),
            self._heading_speed_gain * abs(plan.heading_rad),
        )
        if turn_severity > 0.0:
            target_speed = max(self._min_turn_speed, self._target_speed / (1.0 + turn_severity))

        speed_err = target_speed - state.speed_mps

        if speed_err > 0:
            throttle = min(self._speed_kp * speed_err, self._max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(self._speed_kp * abs(speed_err), self._max_brake)

        return throttle, brake

    def reset(self) -> None:
        """Clear controller state on scene resets."""
        self._prev_steering = 0.0
