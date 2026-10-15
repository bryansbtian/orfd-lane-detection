import math
from unittest.mock import Mock

import pytest

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.simulation.odometry import WheelGyroOdometry
from offroad_autonomy.types import PipelineConfig


def test_straight_motion_starts_at_zero_and_uses_sensor_time():
    odom = WheelGyroOdometry()
    first = odom.update(1234, 2, 0)
    assert first.position == (0, 0, 0)
    assert first.direction == (1, 0, 0)
    state = odom.update(1234.5, 2, 0)
    assert state.position == pytest.approx((1, 0, 0))
    assert state.speed_mps == 2


@pytest.mark.parametrize("sign", [-1, 1])
def test_quarter_circle_and_turn_sign(sign):
    odom = WheelGyroOdometry()
    for i in range(101):
        state = odom.update(i * math.pi / 200, 2, sign)
    assert state.position == pytest.approx((2, sign * 2, 0), abs=1e-8)
    assert state.direction == pytest.approx((0, sign, 0), abs=1e-8)


def test_reverse_uses_wheel_rotation_not_gear():
    odom = WheelGyroOdometry()
    odom.update(0, -2, 0)
    state = odom.update(0.5, -2, 0)
    assert state.position == (-1, 0, 0)
    assert state.speed_mps == 2
    assert state.velocity == (-2, 0, 0)


@pytest.mark.parametrize(
    "time,speed,rate",
    [
        (0, 1, 0),
        (-1, 1, 0),
        (2, 1, 0),
        (0.1, float("nan"), 0),
        (0.1, 1, float("inf")),
    ],
)
def test_bad_or_stale_sample_invalidates_pose_and_restarts_locally(time, speed, rate):
    odom = WheelGyroOdometry()
    odom.update(0, 1, 0)
    assert not odom.update(time, speed, rate).valid
    fresh = odom.update(3, 1, 0)
    assert fresh.valid
    assert fresh.position == (0, 0, 0)


def test_client_only_requests_sensor_channels_never_global_state():
    class SensorOnlyVehicle:
        def queue_lua_command(self, code, response):
            assert response
            assert "getYawAngularVelocity" in code and "wheelspeed" in code
            assert all(word not in code for word in ("getPosition", "getDirection", "getVelocity"))
            return '{"time": 10, "wheel_speed": 2, "yaw_rate": 0.1}'

        @property
        def state(self):
            raise AssertionError("Global state must not be read")

        @property
        def sensors(self):
            raise AssertionError("State sensor must not be polled")

    client = BeamNGClient(PipelineConfig())
    client._vehicle = SensorOnlyVehicle()
    state = client.get_vehicle_state()
    assert state.valid
    assert state.position == (0, 0, 0)


def test_sensor_failure_has_no_truth_fallback():
    client = BeamNGClient(PipelineConfig())
    client._vehicle = Mock()
    client._vehicle.queue_lua_command.side_effect = RuntimeError("sensor disconnected")
    assert not client.get_vehicle_state().valid
    client._vehicle.sensors.poll.assert_not_called()
