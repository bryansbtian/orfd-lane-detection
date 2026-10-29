"""Physics And Input-Authority Qualification For The Approved R3 Bends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from offroad_autonomy.simulation.r3_client import R3BeamNGClient
from offroad_autonomy.types import ControlCommand


class R3VerificationClient(R3BeamNGClient):
    def __init__(self, config) -> None:
        super().__init__(config)
        self._verifier_active = False
        self._physics_stream = None

    def start_physics_recording(self, raw_path: Path, envelope_margin_m: float) -> dict:
        calibration = self.probe_physics_geometry()
        forward = np.asarray(calibration["forward"])
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, calibration["up"])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        corners = np.asarray(list(calibration["ge_bbox"].values()))
        local = (corners - calibration["position"]) @ np.array([forward, right, up]).T
        bounds = np.column_stack((local.min(axis=0), local.max(axis=0))).ravel().tolist()
        source_path = Path(__file__).with_name("r3_vehicle_verifier.lua")
        source = source_path.read_text(encoding="utf-8")
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        args = {
            "body_bounds": bounds,
            "envelope_margin_m": envelope_margin_m,
            "max_buffered_samples": self._config.r3_max_buffered_samples,
        }
        encoded = json.dumps(json.dumps(args, allow_nan=False))
        chunk = (
            'extensions.unload("r3Verifier"); '
            f'package.loaded["r3Verifier"] = (function() {source} end)(); '
            'extensions.load("r3Verifier"); '
            f"return jsonEncode(extensions.r3Verifier.start(jsonDecode({encoded})))"
        )
        status = self._decode_lua(self._vehicle.queue_lua_command(chunk, response=True))
        self._verifier_active = True
        self._physics_stream = raw_path.open("x", encoding="utf-8", newline="")
        return {
            "native": status,
            "body_bounds": bounds,
            "calibration": calibration,
            "envelope_margin_m": envelope_margin_m,
            "source_sha256": source_sha256,
        }

    def native_status(self) -> dict:
        result = self._decode_lua(
            self._vehicle.queue_lua_command(
                "return jsonEncode(extensions.r3Verifier.packet())", response=True
            )
        )
        self._physics_stream.write(result.pop("csv"))
        self._physics_stream.flush()
        return result

    def native_command(self, values: dict) -> None:
        encoded = json.dumps(json.dumps(values, allow_nan=False))
        self._vehicle.queue_lua_command(f"extensions.r3Verifier.apply(jsonDecode({encoded}))")

    def send_controls(self, cmd: ControlCommand) -> None:
        if not self._verifier_active:
            super().send_controls(cmd)
            return
        self.native_command(
            {
                "steering": cmd.steering,
                "throttle": cmd.throttle,
                "brake": cmd.brake,
                "parkingbrake": cmd.parkingbrake,
            }
        )

    def park(self) -> None:
        if self._verifier_active:
            self.native_command({"steering": 0, "throttle": 0, "brake": 1, "parkingbrake": 1})
            return
        super().park()

    def stop_physics_recording(self) -> dict:
        result = self._decode_lua(
            self._vehicle.queue_lua_command(
                "return jsonEncode(extensions.r3Verifier.stop())", response=True
            )
        )
        self._verifier_active = False
        self._physics_stream.write(
            self._vehicle.queue_lua_command("return extensions.r3Verifier.drain()", response=True)
        )
        self._physics_stream.close()
        self._physics_stream = None
        self._vehicle.queue_lua_command('extensions.unload("r3Verifier")')
        return result

    def inject_validation_input(self, chunk: str) -> dict:
        self._vehicle.queue_lua_command(chunk)
        return self.native_status()

    def inject_external_control(self) -> None:
        # Qualification must exercise the normal client API outside verifier authority.
        self._vehicle.control(throttle=1, steering=1)

    def probe_physics_geometry(self) -> dict:
        self.park()
        self._bng.control.pause()
        try:
            result = self._decode_lua(
                self._vehicle.queue_lua_command(
                    """
local function vector(p) return {p.x, p.y, p.z} end
local checks = {}
local methods = {
  corner_0 = function() return vector(obj:getCornerPosition(0)) end,
  corner_1 = function() return vector(obj:getCornerPosition(1)) end,
  corner_2 = function() return vector(obj:getCornerPosition(2)) end,
  corner_3 = function() return vector(obj:getCornerPosition(3)) end,
  initial_width = function() return obj:getInitialWidth() end,
  initial_length = function() return obj:getInitialLength() end,
  initial_height = function() return obj:getInitialHeight() end,
  center = function() return vector(obj:getObjectCenterPosition(objectId)) end,
  spawn_oobb = function()
    local box = obj:getSpawnWorldOOBB()
    local points = {}
    for i = 0, 7 do points[#points+1] = vector(box:getPoint(i)) end
    return points
  end
}
for name, fn in pairs(methods) do
  local ok, value = pcall(fn)
  checks[name] = {available=ok, value=tostring(value)}
  if ok then
    checks[name].value = value
  end
end
return jsonEncode({checks=checks, physics_dt=obj:getPhysicsDt(), time=obj:getSimTime(),
  position=vector(obj:getPosition()), forward=vector(obj:getDirectionVector()),
  up=vector(obj:getDirectionVectorUp())})
""",
                    response=True,
                )
            )
            result["ge_bbox"] = self._vehicle.get_bbox()
            result["vehicle_parts"] = self._vehicle.get_part_config()
            result["camera_world_position"] = self._cameras["dashcam"].get_position()
            result["camera_world_direction"] = self._cameras["dashcam"].get_direction()
            return result
        finally:
            self._bng.control.resume()
