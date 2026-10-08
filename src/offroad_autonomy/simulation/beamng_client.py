"""BeamNG.tech session, sensor I/O and actuation.

Nothing outside this module imports ``beamngpy``, so the rest of the stack
can be tested and run without a simulator.
"""

from __future__ import annotations

import logging
import math
import time
import zlib

import cv2
import numpy as np

from offroad_autonomy.types import (
    CameraFrame,
    CameraSpec,
    ControlCommand,
    PipelineConfig,
    VehicleState,
)
from offroad_autonomy.utils.environment import LOCAL_HOSTS

logger = logging.getLogger("offroad_autonomy.simulation")

_PHYSICS_SETTLE_S = 3.0
_SENSOR_WARMUP_S = 1.0


class BeamNGConnectionError(RuntimeError):
    """Simulator startup failed, with instructions for the configured endpoint."""


class BeamNGClient:
    def __init__(self, config: PipelineConfig, orbit: bool = False) -> None:
        self._config = config
        # A flag rather than a config switch: the orbit camera costs capture
        # time in the loop, so only a run showing or recording a presentation pays it.
        self._orbit = orbit
        self._bng = None
        self._vehicle = None
        self._cameras: dict[str, object] = {}
        self._frame_id = 0
        self._last_signature: int | None = None
        self._shared_memory = config.beamng_camera_transport == "shared_memory"

    def connect(self) -> None:
        if not self._config.beamng_host:
            raise RuntimeError(
                "No BeamNG address for this machine: BeamNG.tech runs only on Windows, "
                "so set BEAMNG_HOST to the address of the Windows machine running it"
            )
        from beamngpy import BeamNGpy, Scenario, Vehicle
        from beamngpy.logging import BNGDisconnectedError
        from beamngpy.sensors.camera import Camera

        cfg = self._config

        if cfg.beamng_launch:
            logger.info(
                "Connecting to BeamNG at %s:%d (launching from %s if needed)",
                cfg.beamng_host,
                cfg.beamng_port,
                cfg.beamng_home,
            )
        else:
            logger.info("Connecting to running BeamNG at %s:%d", cfg.beamng_host, cfg.beamng_port)
        self._bng = BeamNGpy(cfg.beamng_host, cfg.beamng_port, home=cfg.beamng_home or None)
        try:
            self._bng.open(launch=cfg.beamng_launch)
        except (BNGDisconnectedError, OSError) as exc:
            local = cfg.beamng_host in LOCAL_HOSTS
            listen_ip = "127.0.0.1" if local else "*"
            remedy = (
                "Start BeamNG.tech on the simulator machine with "
                f'-tcom -tport {cfg.beamng_port} -tcom-listen-ip "{listen_ip}". '
                "Opening the game normally does not enable the BeamNGpy server. "
            )
            if cfg.beamng_launch:
                remedy += (
                    f"Automatic launch from {cfg.beamng_home!r} failed; "
                    "check the install path and the simulator startup log."
                )
            elif local:
                remedy += (
                    "To let this app start the simulator on Windows, set BEAMNG_HOME "
                    "to its install folder and BEAMNG_LAUNCH=true."
                )
            else:
                remedy += "Check BEAMNG_HOST, BEAMNG_PORT and the simulator host's firewall."
            raise BeamNGConnectionError(
                f"Cannot connect to BeamNG.tech at {cfg.beamng_host}:{cfg.beamng_port}. "
                f"{remedy} Original error: {exc}"
            ) from exc

        spawn_pos, spawn_rot = self._resolve_spawn(cfg)

        logger.info("Loading map '%s'", cfg.beamng_map)
        scenario = Scenario(cfg.beamng_map, "offroad_autonomy")
        vehicle = Vehicle("ego", model=cfg.beamng_vehicle, licence="OFFROAD")
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

        scenario.make(self._bng)
        self._bng.scenario.load(scenario)
        self._bng.scenario.start()
        vehicle.connect(self._bng)
        self._vehicle = vehicle
        # Hold the spawn while sensors settle. In arcade mode the service
        # brake doubles as reverse, so use only the parking brake here.
        vehicle.control(throttle=0, brake=0, parkingbrake=1)

        # Cameras attached while the vehicle is still dropping onto the terrain
        # render a bouncing horizon that the first plans would steer on.
        time.sleep(_PHYSICS_SETTLE_S)

        spec = cfg.camera
        self._cameras["dashcam"] = self._attach_camera(Camera, vehicle, spec)
        logger.info(
            "Attaching dashcam (%s): %s", cfg.beamng_camera_transport, spec.sensor.describe()
        )
        if self._orbit:
            orbit = cfg.orbit_camera
            self._cameras["orbit"] = self._attach_camera(Camera, vehicle, orbit)
            logger.info("Attaching orbit camera for the presentation: %s", orbit.sensor.describe())
        time.sleep(_SENSOR_WARMUP_S)

        self.release_park()
        logger.info("BeamNG session ready")

    def _attach_camera(self, camera_cls, vehicle, spec: CameraSpec):
        # beamngpy takes the vertical field of view; handing it the
        # horizontal one would silently widen the lens and skew path projection.
        # Streaming is only available over shared memory, which a remote
        # client cannot map, so socket transport polls instead.
        return camera_cls(
            name=spec.name,
            bng=self._bng,
            vehicle=vehicle,
            pos=tuple(spec.pos),
            dir=tuple(spec.dir),
            up=tuple(spec.up),
            resolution=(spec.width, spec.height),
            field_of_view_y=spec.fov_y_deg,
            near_far_planes=(0.1, 500.0),
            requested_update_time=spec.sensor.frame_interval_s,
            update_priority=1.0,
            is_render_colours=True,
            is_render_annotations=False,
            is_render_instance=False,
            is_render_depth=False,
            is_using_shared_memory=self._shared_memory,
            is_streaming=self._shared_memory,
        )

    def capture_frame(self) -> CameraFrame | None:
        # Copy shared memory before decoding so display and inference keep
        # the same immutable snapshot even when BeamNG renders again.
        buffer = self._live_buffer("dashcam")
        if buffer is None:
            return None
        timestamp = time.perf_counter()
        snapshot = bytes(buffer)
        image = self._decode(snapshot, self._config.camera)
        if image is None:
            return None
        signature = frame_signature(snapshot)
        is_new = signature != self._last_signature
        self._last_signature = signature
        self._frame_id += 1
        return CameraFrame(image=image, timestamp=timestamp, frame_id=self._frame_id, is_new=is_new)

    def capture_orbit(self) -> np.ndarray | None:
        """For the presentation view and video; no pipeline stage reads it.

        Called from the main loop, never the dashboard thread: the socket
        transport is not thread safe."""
        if "orbit" not in self._cameras:
            return None
        return self._decode(self._live_buffer("orbit"), self._config.orbit_camera)

    def _live_buffer(self, role: str):
        camera = self._cameras.get(role)
        if camera is None:
            return None
        try:
            if self._shared_memory:
                # Reading the mapped buffer directly skips beamngpy's PIL
                # conversion, which alone cost more than the loop budget.
                return camera.colour_shmem.read()
            raw = camera.poll_raw().get("colour")
            if isinstance(raw, str):
                return raw.encode()
            return raw
        except Exception as exc:
            logger.debug("Frame capture failed for %s camera: %s", role, exc)
            return None

    @staticmethod
    def _decode(buffer, spec: CameraSpec) -> np.ndarray | None:
        if buffer is None:
            return None
        pixels = np.frombuffer(buffer, dtype=np.uint8)
        expected = spec.width * spec.height
        if pixels.size == expected * 4:
            return cv2.cvtColor(pixels.reshape(spec.height, spec.width, 4), cv2.COLOR_RGBA2BGR)
        if pixels.size == expected * 3:
            return cv2.cvtColor(pixels.reshape(spec.height, spec.width, 3), cv2.COLOR_RGB2BGR)
        logger.debug("Unexpected %s buffer size %d", spec.name, pixels.size)
        return None

    def get_vehicle_state(self) -> VehicleState:
        if self._vehicle is None:
            return VehicleState(valid=False)

        try:
            self._vehicle.sensors.poll()
            st = self._vehicle.state
            if not all(key in st for key in ("pos", "vel", "rotation")):
                return VehicleState(valid=False)
            pos = tuple(st.get("pos", (0, 0, 0)))
            rot = tuple(st.get("rotation", (0, 0, 0, 1)))
            vel = tuple(st.get("vel", (0, 0, 0)))
            speed = math.sqrt(sum(v**2 for v in vel))
            _, _, yaw = self._quat_to_euler(rot)
            direction = None
            if "dir" in st:
                direction = tuple(st["dir"])
            return VehicleState(
                position=pos,
                rotation=rot,
                velocity=vel,
                speed_mps=speed,
                heading_rad=yaw,
                direction=direction,
            )
        except Exception as exc:
            # Do not let a failed poll masquerade as a stationary vehicle.
            logger.debug("State poll failed: %s", exc)
            return VehicleState(valid=False)

    def send_controls(self, cmd: ControlCommand) -> None:
        if self._vehicle is None:
            return
        self._vehicle.control(
            steering=cmd.steering,
            throttle=cmd.throttle,
            brake=cmd.brake,
            parkingbrake=cmd.parkingbrake,
        )

    def release_park(self) -> None:
        if self._vehicle is None:
            return
        # This stack only plans forward motion. Arcade shifting turns a held
        # brake into reverse throttle at standstill, including a gate stop.
        # Configure on every resume as well, since an operator may have used R.
        self._vehicle.control(throttle=0, brake=0, parkingbrake=1)
        self._vehicle.set_shift_mode("realistic_automatic")
        gearbox_type = self._vehicle.queue_lua_command(
            'return powertrain.getDevice("gearbox").type',
            response=True,
        )
        # These are shifter positions: 1 selects first on a manual but P on
        # an automatic; 2 selects D. Direct selection also works from R.
        if gearbox_type in ("manualGearbox", "sequentialGearbox"):
            forward_gear = 1
        elif gearbox_type in ("automaticGearbox", "dctGearbox", "cvtGearbox"):
            forward_gear = 2
        else:
            raise RuntimeError(f"Unsupported forward-drive gearbox: {gearbox_type!r}")
        self._vehicle.control(gear=forward_gear)
        self._vehicle.control(steering=0, throttle=0, brake=0, parkingbrake=0)
        # control() alone leaves the Lua input filter holding the parking
        # brake that park() set through it.
        try:
            self._vehicle.queue_lua_command('input.event("parkingbrake", 0, FILTER_DIRECT, 0)')
        except Exception as exc:
            logger.debug("Lua release park failed: %s", exc)

    def park(self) -> None:
        if self._vehicle is None:
            return
        self._vehicle.control(steering=0, throttle=0, brake=1, parkingbrake=1)
        # Apply the stop through the input filter too. The old self:setVelocity
        # call is invalid in vehicle Lua and prevented these inputs executing.
        try:
            self._vehicle.queue_lua_command(
                'input.event("throttle", 0, FILTER_DIRECT, 0); '
                'input.event("brake", 1, FILTER_DIRECT, 0); '
                'input.event("parkingbrake", 1, FILTER_DIRECT, 0)'
            )
        except Exception as exc:
            logger.debug("Lua park failed: %s", exc)

    def disconnect(self) -> None:
        # Park before letting go: BeamNG keeps applying the last command, so
        # a client that simply vanishes leaves the vehicle driving.
        try:
            self.park()
        except Exception:
            logger.debug("Failed to park before disconnecting")

        for role, camera in self._cameras.items():
            try:
                camera.remove()
            except Exception:
                logger.debug("Failed to remove %s camera", role)
        self._cameras.clear()

        if self._bng is not None:
            # close() also quits the simulator, which is only ours to quit
            # when this client launched it; a shared remote one stays up.
            try:
                if self._config.beamng_launch:
                    self._bng.close()
                else:
                    self._bng.disconnect()
            except Exception:
                logger.debug("Failed to disconnect from BeamNG")
            self._bng = None

        self._vehicle = None
        logger.info("BeamNG session closed")

    def _resolve_spawn(self, cfg: PipelineConfig) -> tuple[tuple, tuple]:
        map_cfg = cfg.map_spawns.get(cfg.beamng_map)
        if map_cfg and "spawns" in map_cfg:
            spawns = map_cfg["spawns"]
            idx = min(cfg.beamng_spawn_index, len(spawns) - 1)
            s = spawns[idx]
            pos = tuple(s.get("pos", [0, 0, 0]))
            rot = tuple(s.get("rot", [0, 0, 0, 1]))
            logger.info("Spawn #%d: pos=%s", idx, pos)
            return pos, rot

        logger.warning("No spawn data for map '%s' - using origin", cfg.beamng_map)
        return (0, 0, 0), (0, 0, 0, 1)

    @staticmethod
    def _quat_to_euler(q: tuple) -> tuple[float, float, float]:
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


def frame_signature(buffer) -> int | None:
    """CRC of every 997th byte.

    997 is prime, so the samples walk across rows and channels instead of
    landing on one column; a few microseconds still catches any new render.
    """
    if buffer is None:
        return None
    sample = np.frombuffer(buffer, dtype=np.uint8)[::997]
    return zlib.crc32(sample.tobytes())
