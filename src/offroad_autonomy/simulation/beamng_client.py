"""BeamNG.tech simulator client.

Encapsulates all BeamNG-specific I/O: launching the simulator, loading
a scenario, capturing camera frames, polling vehicle state, and sending
control commands.  Nothing outside this module should import ``beamngpy``.
"""

from __future__ import annotations

import logging
import math
import time

import cv2
import numpy as np

from offroad_autonomy.types import ControlCommand, PipelineConfig, VehicleState

logger = logging.getLogger("offroad_autonomy.simulation")


class BeamNGClient:
    """Manages the BeamNG.tech session lifecycle and sensor I/O."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._bng = None
        self._vehicle = None
        self._camera = None

    def connect(self) -> None:
        """Launch BeamNG, load the configured map, spawn the vehicle, and attach a camera."""
        from beamngpy import BeamNGpy, Scenario, Vehicle
        from beamngpy.sensors.camera import Camera

        cfg = self._config

        logger.info("Launching BeamNG from %s", cfg.beamng_home)
        self._bng = BeamNGpy(cfg.beamng_host, cfg.beamng_port, home=cfg.beamng_home)
        self._bng.open(launch=True)

        spawn_pos, spawn_rot = self._resolve_spawn(cfg)

        logger.info("Loading map '%s'", cfg.beamng_map)
        scenario = Scenario(cfg.beamng_map, "offroad_autonomy")
        vehicle = Vehicle("ego", model=cfg.beamng_vehicle, licence="OFFROAD")
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

        scenario.make(self._bng)
        self._bng.scenario.load(scenario)
        self._bng.scenario.start()
        vehicle.connect(self._bng)

        logger.info("Waiting for physics to settle")
        time.sleep(3.0)

        logger.info("Attaching front camera")
        self._camera = Camera(
            name="front_cam",
            bng=self._bng,
            vehicle=vehicle,
            pos=tuple(cfg.camera_pos),
            dir=tuple(cfg.camera_dir),
            up=(0, 0, 1),
            resolution=(cfg.camera_width, cfg.camera_height),
            field_of_view_y=cfg.camera_fov,
            near_far_planes=(0.1, 500.0),
            requested_update_time=0.01,
            update_priority=1.0,
            is_render_colours=True,
            is_render_annotations=False,
            is_render_depth=False,
            is_using_shared_memory=True,
            is_streaming=True,
        )
        time.sleep(1.0)

        self._vehicle = vehicle
        logger.info("BeamNG session ready")

    def capture_frame(self) -> np.ndarray | None:
        """Grab the latest colour frame from the streaming camera."""
        if self._camera is None:
            return None
        try:
            images = self._camera.stream()
            colour = images.get("colour")
            if colour is None:
                return None
            return cv2.cvtColor(np.array(colour.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.debug("Frame capture failed: %s", exc)
            return None

    def get_vehicle_state(self) -> VehicleState:
        """Poll the latest vehicle telemetry."""
        if self._vehicle is None:
            return VehicleState()

        try:
            self._vehicle.sensors.poll()
            st = self._vehicle.state
            pos = tuple(st.get("pos", (0, 0, 0)))
            rot = tuple(st.get("rotation", (0, 0, 0, 1)))
            vel = tuple(st.get("vel", (0, 0, 0)))
            speed = math.sqrt(sum(v ** 2 for v in vel))

            _, _, yaw = self._quat_to_euler(rot)

            return VehicleState(
                position=pos,
                rotation=rot,
                velocity=vel,
                speed_mps=speed,
                heading_rad=yaw,
            )
        except Exception as exc:
            logger.debug("State poll failed: %s", exc)
            return VehicleState()

    def send_controls(self, cmd: ControlCommand) -> None:
        """Send steering, throttle, and brake to the vehicle."""
        if self._vehicle is None:
            return
        self._vehicle.control(
            steering=cmd.steering,
            throttle=cmd.throttle,
            brake=cmd.brake,
        )

    def set_beamng_ai(self, enabled: bool) -> None:
        """Toggle BeamNG's built-in AI driver."""
        if self._vehicle is None:
            return
        try:
            if enabled:
                self._vehicle.ai.set_mode("span")
                logger.info("BeamNG AI enabled (span mode)")
            else:
                self._vehicle.ai.set_mode("disabled")
                logger.info("BeamNG AI disabled")
        except Exception as exc:
            logger.warning("Failed to toggle BeamNG AI: %s", exc)

    def disconnect(self) -> None:
        """Tear down the session gracefully."""
        if self._camera is not None:
            try:
                self._camera.remove()
            except Exception:
                pass
            self._camera = None

        if self._bng is not None:
            try:
                self._bng.close()
            except Exception:
                pass
            self._bng = None

        self._vehicle = None
        logger.info("BeamNG session closed")

    def _resolve_spawn(self, cfg: PipelineConfig) -> tuple[tuple, tuple]:
        """Look up spawn position/rotation from the config map table."""
        map_cfg = cfg.map_spawns.get(cfg.beamng_map)
        if map_cfg and "spawns" in map_cfg:
            spawns = map_cfg["spawns"]
            idx = min(cfg.beamng_spawn_index, len(spawns) - 1)
            s = spawns[idx]
            pos = tuple(s.get("pos", [0, 0, 0]))
            rot = tuple(s.get("rot", [0, 0, 0, 1]))
            logger.info("Spawn #%d: pos=%s", idx, pos)
            return pos, rot

        logger.warning("No spawn data for map '%s' — using origin", cfg.beamng_map)
        return (0, 0, 0), (0, 0, 0, 1)

    @staticmethod
    def _quat_to_euler(q: tuple) -> tuple[float, float, float]:
        """Convert quaternion (x, y, z, w) to Euler (roll, pitch, yaw)."""
        x, y, z, w = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
