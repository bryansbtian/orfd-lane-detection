"""Planar wheel/gyro dead reckoning in an arbitrary local frame.

Inputs are signed wheel speed, body yaw rate (positive left), and sensor time.
There is no global pose, compass heading, or simulator velocity input. Wheel
slip and body tilt can cause drift; no ground-truth correction is applied.
"""

from __future__ import annotations

import math

from offroad_autonomy.types import VehicleState


class WheelGyroOdometry:
    def __init__(self, max_gap_s: float = 1.0) -> None:
        self.max_gap_s = max_gap_s
        self.reset()

    def reset(self) -> None:
        self.x = self.y = self.yaw = 0.0
        self._previous: tuple[float, float, float] | None = None

    def update(self, timestamp: float, speed: float, yaw_rate: float) -> VehicleState:
        if not all(math.isfinite(v) for v in (timestamp, speed, yaw_rate)):
            self.reset()
            return VehicleState(valid=False)
        if self._previous is not None:
            last_time, last_speed, last_rate = self._previous
            dt = timestamp - last_time
            if dt <= 0 or dt > self.max_gap_s:
                self.reset()
                return VehicleState(valid=False)
            # Trapezoidal sensor integration, exact arc for the mean turn rate.
            distance = 0.5 * (speed + last_speed) * dt
            angle = 0.5 * (yaw_rate + last_rate) * dt
            half = angle / 2
            scale = math.sin(half) / half if abs(half) > 1e-8 else 1.0
            self.x += distance * scale * math.cos(self.yaw + half)
            self.y += distance * scale * math.sin(self.yaw + half)
            self.yaw = math.remainder(self.yaw + angle, 2 * math.pi)
        self._previous = timestamp, speed, yaw_rate
        fx, fy = math.cos(self.yaw), math.sin(self.yaw)
        return VehicleState(
            position=(self.x, self.y, 0.0),
            rotation=(0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)),
            direction=(fx, fy, 0.0),
            velocity=(speed * fx, speed * fy, 0.0),
            speed_mps=abs(speed),
            heading_rad=self.yaw,
        )
