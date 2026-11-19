"""Measure a parked vehicle's wheel response without asserting real-recording semantics."""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from offroad_autonomy.simulation.beamng_client import BeamNGClient

logger = logging.getLogger("offroad_autonomy.calibration")

# Wheel node axes give physical road-wheel angles independently of the steering input.
# The reference frame moves with the body, so spawn yaw and suspension pitch cancel.
_WHEEL_SAMPLE = """
local f = obj:getDirectionVector():normalized()
local u = obj:getDirectionVectorUp():normalized()
local r = f:cross(u):normalized()
local out = {wheels = {}, time = obj:getSimTime(), speed_mps=electrics.values.wheelspeed}
for _, w in pairs(wheels.wheels) do
  local p = obj:getNodePosition(w.node1)
  local q = obj:getNodePosition(w.node2)
  local axis = (q-p):normalized()
  if axis:dot(r) < 0 then
    axis = -axis
  end
  local mid = (p+q)*0.5
  table.insert(out.wheels, {name=w.name, position={mid.x,mid.y,mid.z},
    angle_left_rad=math.atan2(axis:dot(f),axis:dot(r))})
end
return jsonEncode(out)
"""


def summarize_sweep(samples, settings):
    """Reject incomplete, nonmonotonic or inconsistent measurements before making a table."""
    commands = sorted({float(row["command"]) for row in samples})
    if len(commands) != settings.calibration_points or 0.0 not in commands:
        raise ValueError("Calibration sweep must include every command and neutral")
    wheelbases = []
    grouped = {command: [] for command in commands}
    for row in samples:
        if (
            not math.isfinite(row["speed_mps"])
            or abs(row["speed_mps"]) > settings.calibration_stationary_speed_mps
        ):
            raise ValueError("Calibration vehicle must remain stationary")
        wheels = {wheel["name"]: wheel for wheel in row["wheels"]}
        if set(wheels) != {"FL", "FR", "RL", "RR"}:
            raise ValueError("Calibration requires four named road wheels")
        positions = {
            name: np.asarray(wheel["position"], dtype=float) for name, wheel in wheels.items()
        }
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in positions.values()):
            raise ValueError("Wheel positions must be finite 3D vectors")
        front = (positions["FL"] + positions["FR"]) / 2
        rear = (positions["RL"] + positions["RR"]) / 2
        wheelbases.append(float(np.linalg.norm(front - rear)))
        angles = np.array([wheels[name]["angle_left_rad"] for name in ("FL", "FR")])
        if not np.isfinite(angles).all():
            raise ValueError("Road-wheel angles must be finite")
        tangents = np.tan(angles)
        tangent = float(tangents.mean())
        # Ackermann wheels turn through different angles; their virtual centre wheel
        # has the harmonic mean tangent when both wheels turn to the same side.
        if tangents[0] * tangents[1] > 0:
            tangent = float(2 * tangents.prod() / tangents.sum())
        grouped[float(row["command"])].append(math.atan(tangent))
    angles = []
    repeatability = 0.0
    for command in commands:
        values = grouped[command]
        if len(values) < settings.calibration_samples * 2:
            raise ValueError("Calibration requires two sampled sweeps")
        repeatability = max(repeatability, float(np.ptp(values)))
        angles.append(float(np.median(values)))
    if repeatability > settings.calibration_repeatability_rad:
        raise ValueError("Steering repeats exceed calibration_repeatability_rad")
    neutral = angles[commands.index(0.0)]
    if abs(neutral) > settings.calibration_neutral_tolerance_rad:
        raise ValueError("Neutral steering exceeds calibration_neutral_tolerance_rad")
    angles = np.array(angles) - neutral
    if not (np.diff(angles) < 0).all() or not angles[-1] < 0 < angles[0]:
        raise ValueError("Measured positive-left angles must decrease with command")
    return {
        "simulator_wheelbase_m": float(np.median(wheelbases)),
        "wheel_angle_left_rad": angles[::-1].tolist(),
        "beamng_commands": commands[::-1],
        "neutral_bias_rad": neutral,
        "repeatability_rad": repeatability,
        "samples": len(samples),
    }


def measure_calibration(config, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    settings = cfg.imitation
    client = BeamNGClient(cfg)
    samples = []
    try:
        client.connect()
        client.park()
        vehicle = client._vehicle
        vehicle.sensors.poll("state")
        state = dict(vehicle.state)
        camera = client._cameras["dashcam"]
        camera_evidence = {
            "configured": asdict(cfg.camera),
            "world_position": camera.get_position(),
            "world_direction": camera.get_direction(),
            "vehicle_position": state["pos"],
            "vehicle_forward": state["dir"],
            "vehicle_up": state["up"],
            "source_camera_extrinsics_verified": False,
        }
        frame = client.capture_frame()
        if frame is not None:
            cv2.imwrite(str(output / "dashcam.jpg"), frame.image)
        commands = np.linspace(-1, 1, settings.calibration_points).tolist()
        for command in commands + commands[::-1]:
            logger.info("Measuring Parked Steering: %+.2f", command)
            vehicle.control(steering=command, throttle=0, brake=1, parkingbrake=1)
            time.sleep(settings.calibration_settle_s)
            for _ in range(settings.calibration_samples):
                sample = vehicle.queue_lua_command(_WHEEL_SAMPLE, response=True)
                if isinstance(sample, str):
                    sample = json.loads(sample)
                sample["command"] = command
                samples.append(sample)
                time.sleep(settings.calibration_sample_interval_s)
        table = summarize_sweep(samples, settings)
        report = {
            "version": 2,
            "simulation_only": True,
            "measured": True,
            "validation_mode": "simulation_experiment",
            "measurement_source": "Live BeamNG Wheel Node Axes; Two Parked Steering Sweeps",
            "simulator_vehicle": cfg.beamng_vehicle,
            "simulator_camera": asdict(cfg.camera),
            "camera_evidence": camera_evidence,
            "recorded_rwa_radians_positive_left_verified": False,
            "source_camera_extrinsics_verified": False,
            "recorded_units_status": "dynamics_supported_radians_hypothesis",
            "recorded_direction_status": "positive_yaw_hypothesis_not_publisher_verified",
            "real_wheelbase_m": settings.real_wheelbase_m,
            "real_wheelbase_status": "recorded_2815_assumed_millimetres",
            **table,
        }
        path = output / "measured_calibration.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        # Retain incomplete measurements for diagnosis, never as an actuation adapter.
        try:
            (output / "wheel_samples.json").write_text(
                json.dumps(samples, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        finally:
            client.disconnect()
