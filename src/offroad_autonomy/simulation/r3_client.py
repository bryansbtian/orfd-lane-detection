"""Evaluation-only ground truth. Never pass these channels to the driving stack."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import ControlCommand

logger = logging.getLogger("offroad_autonomy.simulation.r3")


class R3BeamNGClient(BeamNGClient):
    """Reuse normal startup and actuation, exposing independent measurement I/O."""

    def road_survey(self) -> dict:
        self.park()
        return {
            "roads": self._bng.scenario.get_road_network(include_edges=True, drivable_only=False),
            "time_of_day": self._bng.env.get_tod(),
            "system": self._bng.system.get_info(),
            "vehicle_parts": self._vehicle.get_part_config(),
        }

    def configure_environment(self, weather_preset: str, time_of_day: float) -> dict:
        self.park()
        logger.info("Setting weather preset: %s", weather_preset)
        # BeamNG interpolates using timer / transition duration; zero produces
        # invalid environment values instead of an instantaneous transition.
        self._bng.env.set_weather_preset(weather_preset, time=1)
        time.sleep(2)
        self._bng.env.set_tod(time_of_day, play=False)
        logger.info("Weather transition complete; daytime fixed")
        return self._bng.env.get_tod()

    def selected_road(self, persistent_id: str) -> dict:
        logger.info("Retrieving selected road geometry")
        roads = self._bng.scenario.get_road_network(include_edges=True, drivable_only=False)
        matches = [road for road in roads.values() if road.get("persistentId") == persistent_id]
        if len(matches) != 1:
            raise ValueError("Selected Persistent Road ID Must Match Exactly One Road")
        return matches[0]

    def session_info(self) -> dict:
        logger.info("Retrieving vehicle configuration and system information")
        return {
            "system": self._bng.system.get_info(),
            "beamng_version": self._bng.control.queue_lua_command(
                "return beamng_version", response=True
            ),
            "vehicle_parts": self._vehicle.get_part_config(),
            "environment_paths": self._bng.system.get_environment_paths(),
        }

    def start_recording(self) -> dict:
        logger.info("Starting native graphics-step recorder")
        source = Path(__file__).with_name("r3_recorder.lua").read_text(encoding="utf-8")
        # Loading an in-memory extension avoids writing into the simulator install
        # or user folder. The normal extension hook samples each graphics update.
        chunk = (
            'extensions.unload("r3Recorder"); '
            f'package.loaded["r3Recorder"] = (function() {source} end)(); '
            'extensions.load("r3Recorder"); '
            f"extensions.r3Recorder.start({json.dumps(self._vehicle.vid)}, "
            f"{self._config.r3_max_buffered_samples}); "
            "return jsonEncode({version=beamng_version, "
            "time_of_day=core_environment.getTimeOfDay()})"
        )
        return self._decode_lua(self._bng.control.queue_lua_command(chunk, response=True))

    def drain_recording(self) -> dict:
        return self._decode_lua(
            self._bng.control.queue_lua_command(
                "return jsonEncode(extensions.r3Recorder.drain())", response=True
            )
        )

    def stop_recording(self) -> None:
        if self._bng is not None:
            self._bng.control.queue_lua_command('extensions.unload("r3Recorder")')

    def send_controls(self, cmd: ControlCommand) -> None:
        super().send_controls(cmd)
        # These are requested inputs, not proof that no human overrode the inputs.
        command = json.dumps(
            {"steering": cmd.steering, "throttle": cmd.throttle, "brake": cmd.brake},
            allow_nan=False,
        )
        self._bng.control.queue_lua_command(
            "if extensions.r3Recorder then\n"
            f"  extensions.r3Recorder.command = jsonDecode({json.dumps(command)})\n"
            "end"
        )

    def reset_odometry(self) -> None:
        self._odometry.reset()

    @staticmethod
    def _decode_lua(value) -> dict:
        if isinstance(value, str):
            return json.loads(value)
        if isinstance(value, dict):
            return value
        raise RuntimeError(f"Unexpected R3 Recorder Response: {value!r}")
