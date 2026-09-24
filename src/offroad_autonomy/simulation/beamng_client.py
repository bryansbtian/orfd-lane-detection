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
    CameraSpec,
    ControlCommand,
    PipelineConfig,
    StereoFramePair,
    VehicleState,
)

logger = logging.getLogger("offroad_autonomy.simulation")

_PHYSICS_SETTLE_S = 3.0
_SENSOR_WARMUP_S = 1.0


class BeamNGClient:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._bng = None
        self._vehicle = None
        self._cameras: dict[str, object] = {}
        self._frame_id = 0
        self._last_signatures: tuple[int | None, int | None] = (None, None)
        self._shared_memory = config.beamng_camera_transport == "shared_memory"

    def connect(self) -> None:
        from beamngpy import BeamNGpy, Scenario, Vehicle
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
        self._bng.open(launch=cfg.beamng_launch)

        spawn_pos, spawn_rot = self._resolve_spawn(cfg)

        logger.info("Loading map '%s'", cfg.beamng_map)
        scenario = Scenario(cfg.beamng_map, "offroad_autonomy")
        vehicle = Vehicle("ego", model=cfg.beamng_vehicle, licence="OFFROAD")
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

        scenario.make(self._bng)
        self._bng.scenario.load(scenario)
        self._bng.scenario.start()
        vehicle.connect(self._bng)

        # Cameras attached while the vehicle is still dropping onto the terrain
        # render a bouncing horizon that the first plans would steer on.
        time.sleep(_PHYSICS_SETTLE_S)

        sensor = cfg.left_camera.sensor
        rig = cfg.stereo_rig
        logger.info(
            "Attaching stereo pair (%s): %s  baseline=%.2f m  pitch=%.1f deg  toe-out=%.1f deg",
            cfg.beamng_camera_transport,
            sensor.describe(),
            rig.baseline_m,
            rig.pitch_deg,
            rig.toe_out_deg,
        )
        for role, spec in (("left", cfg.left_camera), ("right", cfg.right_camera)):
            self._cameras[role] = self._attach_camera(Camera, vehicle, spec)
            logger.info(
                "  %-7s %-12s pos=%s dir=%s",
                role,
                spec.name,
                tuple(round(v, 3) for v in spec.pos),
                tuple(round(v, 3) for v in spec.dir),
            )

        # Attached last and never read by capture_pair(), so nothing in the
        # perception path can reach it even by accident.
        if cfg.display_rig.enabled and not cfg.ui_headless:
            spec = cfg.display_camera
            self._cameras["display"] = self._attach_camera(Camera, vehicle, spec)
            logger.info("Attaching display-only camera: %s", spec.sensor.describe())
        time.sleep(_SENSOR_WARMUP_S)

        self._vehicle = vehicle
        logger.info("BeamNG session ready")

    def _attach_camera(self, camera_cls, vehicle, spec: CameraSpec):
        # beamngpy takes the vertical field of view; handing it the
        # horizontal one would silently widen the lens and skew every depth.
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
            is_render_depth=False,
            is_using_shared_memory=self._shared_memory,
            is_streaming=self._shared_memory,
        )

    def capture_pair(self) -> StereoFramePair | None:
        """One left/right capture, judged for synchronisation.

        BeamNG exposes no per-frame timestamp, so "same simulation step" is
        enforced from this side: both cameras share update time and priority,
        both buffers are read back to back, and a signature of each live
        buffer before and after the reads catches a render landing mid-read.
        Such a pair is re-read once, then flagged. A flagged pair still drives
        segmentation; only stereo skips it.
        """
        pair = self._read_pair()
        if pair is not None and pair.sync_note == "changed during read":
            pair = self._read_pair()
        return pair

    def capture_display(self) -> np.ndarray | None:
        """Called only from the dashboard thread, and deliberately separate
        from ``capture_pair`` so no code path can hand this frame to
        perception."""
        if "display" not in self._cameras:
            return None
        return self._decode(self._live_buffer("display"), self._config.display_camera)

    def _read_pair(self) -> StereoFramePair | None:
        cfg = self._config
        left_buf = self._live_buffer("left")
        right_buf = self._live_buffer("right")
        if left_buf is None and right_buf is None:
            return None

        t0 = time.perf_counter()
        sig_l0 = frame_signature(left_buf)
        sig_r0 = frame_signature(right_buf)
        left = self._decode(left_buf, cfg.left_camera)
        right = self._decode(right_buf, cfg.right_camera)
        sig_l1 = frame_signature(left_buf)
        sig_r1 = frame_signature(right_buf)
        t1 = time.perf_counter()

        self._frame_id += 1
        return judge_pair_sync(
            left,
            right,
            before=(sig_l0, sig_r0),
            after=(sig_l1, sig_r1),
            previous=self._last_signatures,
            read_skew_ms=(t1 - t0) * 1000.0,
            max_skew_ms=cfg.sync_max_read_skew_ms,
            require_both_new=cfg.sync_require_both_new,
            frame_id=self._frame_id,
            timestamp=t0,
            on_accept=self._accept_signatures,
        )

    def _accept_signatures(self, signatures: tuple[int | None, int | None]) -> None:
        self._last_signatures = signatures

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
            return VehicleState()

        try:
            self._vehicle.sensors.poll()
            st = self._vehicle.state
            pos = tuple(st.get("pos", (0, 0, 0)))
            rot = tuple(st.get("rotation", (0, 0, 0, 1)))
            vel = tuple(st.get("vel", (0, 0, 0)))
            speed = math.sqrt(sum(v**2 for v in vel))
            _, _, yaw = self._quat_to_euler(rot)
            return VehicleState(
                position=pos,
                rotation=rot,
                velocity=vel,
                speed_mps=speed,
                heading_rad=yaw,
            )
        except Exception as exc:
            # A zero state reads as "stopped", which makes the controller
            # cautious rather than letting one dropped poll end the session.
            logger.debug("State poll failed: %s", exc)
            return VehicleState()

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
        # Brakes alone let the vehicle roll on a slope for seconds; zeroing
        # the velocity makes a safe stop immediate.
        try:
            self._vehicle.queue_lua_command(
                "self:setVelocity(vec3(0,0,0)); "
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


def judge_pair_sync(
    left: np.ndarray | None,
    right: np.ndarray | None,
    before: tuple[int | None, int | None],
    after: tuple[int | None, int | None],
    previous: tuple[int | None, int | None],
    read_skew_ms: float,
    max_skew_ms: float,
    require_both_new: bool,
    frame_id: int,
    timestamp: float,
    on_accept=None,
) -> StereoFramePair:
    """Free of simulator objects so the sync rules are unit-testable."""
    new_left = before[0] != previous[0]
    new_right = before[1] != previous[1]

    synchronized = True
    note = ""
    if left is None or right is None:
        synchronized = False
        note = "camera missing"
    elif before != after:
        synchronized = False
        note = "changed during read"
    elif read_skew_ms > max_skew_ms:
        synchronized = False
        note = f"read skew {read_skew_ms:.1f} ms"
    elif require_both_new and new_left != new_right:
        synchronized = False
        if new_left:
            note = "only left updated"
        else:
            note = "only right updated"

    if synchronized and on_accept is not None:
        on_accept(before)

    return StereoFramePair(
        left=left,
        right=right,
        timestamp=timestamp,
        frame_id=frame_id,
        synchronized=synchronized,
        read_skew_ms=read_skew_ms,
        is_new=new_left or new_right,
        sync_note=note,
    )
