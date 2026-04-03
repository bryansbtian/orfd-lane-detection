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
        self._annotation_camera = None
        self._annotation_frame_saved = False
        self._road_nodes: np.ndarray | None = None  # (N, 3) world XY + half-width
        self._roads_sensor = None

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

        if cfg.debug_mode != "normal":
            self._cache_road_network()
            self._attach_roads_sensor(vehicle)

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

        if cfg.debug_mode != "normal":
            logger.info("Attaching annotation camera (debug_mode=%s)", cfg.debug_mode)

            # Query BeamNG for its annotation color map and find the road color.
            road_bgr = self._find_road_annotation_color()
            if road_bgr is not None:
                self._road_annotation_color = road_bgr
                logger.info(
                    "Road annotation color from BeamNG: BGR=(%d, %d, %d)",
                    *road_bgr,
                )
            else:
                self._road_annotation_color = None
                logger.warning("Could not resolve road annotation color from BeamNG")

            self._annotation_camera = Camera(
                name="annotation_cam",
                bng=self._bng,
                vehicle=vehicle,
                pos=tuple(cfg.camera_pos),
                dir=tuple(cfg.camera_dir),
                up=(0, 0, 1),
                resolution=(cfg.camera_width, cfg.camera_height),
                field_of_view_y=cfg.camera_fov,
                near_far_planes=(0.1, 500.0),
                requested_update_time=0.05,
                update_priority=1.0,
                is_render_colours=True,
                is_render_annotations=True,
                is_render_depth=False,
                is_using_shared_memory=False,
                is_streaming=False,
            )
            time.sleep(2.0)

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

    def capture_annotation_frame(self) -> np.ndarray | None:
        """Grab the latest annotation frame (semantic labels) from BeamNG."""
        if self._annotation_camera is None:
            return None
        try:
            data = self._annotation_camera.poll()
            annotation = data.get("annotation")
            if annotation is None:
                logger.warning("Annotation poll returned no 'annotation' key — keys: %s", list(data.keys()))
                return None
            bgr = cv2.cvtColor(np.array(annotation.convert("RGB")), cv2.COLOR_RGB2BGR)
            if not getattr(self, "_annotation_frame_saved", True):
                save_path = "debug_annotation_frame.png"
                cv2.imwrite(save_path, bgr)
                self._annotation_frame_saved = True
                logger.info("Saved first annotation frame to %s", save_path)
            return bgr
        except Exception as exc:
            logger.debug("Annotation frame capture failed: %s", exc)
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
        """Send steering, throttle, brake, and parking brake to the vehicle."""
        if self._vehicle is None:
            return
        self._vehicle.control(
            steering=cmd.steering,
            throttle=cmd.throttle,
            brake=cmd.brake,
            parkingbrake=cmd.parkingbrake,
        )

    def release_park(self) -> None:
        """Release parking brake so the autopilot can take over."""
        if self._vehicle is None:
            return
        self._vehicle.control(steering=0, throttle=0, brake=0, parkingbrake=0)
        try:
            self._vehicle.queue_lua_command(
                'input.event("parkingbrake", 0, FILTER_DIRECT, 0)'
            )
        except Exception as exc:
            logger.debug("Lua release park failed: %s", exc)

    def park(self) -> None:
        """Immediately zero vehicle velocity and lock it in place."""
        if self._vehicle is None:
            return
        self._vehicle.control(steering=0, throttle=0, brake=1, parkingbrake=1)
        try:
            self._vehicle.queue_lua_command(
                'self:setVelocity(vec3(0,0,0)); '
                'input.event("throttle", 0, FILTER_DIRECT, 0); '
                'input.event("brake", 1, FILTER_DIRECT, 0); '
                'input.event("parkingbrake", 1, FILTER_DIRECT, 0)'
            )
        except Exception as exc:
            logger.debug("Lua park failed: %s", exc)

    def disconnect(self) -> None:
        """Tear down the session gracefully."""
        if self._camera is not None:
            try:
                self._camera.remove()
            except Exception:
                pass
            self._camera = None

        if self._annotation_camera is not None:
            try:
                self._annotation_camera.remove()
            except Exception:
                pass
            self._annotation_camera = None

        if self._roads_sensor is not None:
            try:
                self._roads_sensor.remove()
            except Exception:
                pass
            self._roads_sensor = None

        if self._bng is not None:
            try:
                self._bng.close()
            except Exception:
                pass
            self._bng = None

        self._vehicle = None
        logger.info("BeamNG session closed")

    def _cache_road_network(self) -> None:
        """Cache 3D road edge geometry via get_road_edges() for projection-based masking."""
        try:
            raw = self._bng.get_roads()
        except Exception as exc:
            logger.warning("get_roads() failed: %s", exc)
            return

        road_ids = list(raw.keys()) if isinstance(raw, dict) else []
        logger.info("Fetching edges for %d road segments...", len(road_ids))

        left_segs: list[np.ndarray] = []
        right_segs: list[np.ndarray] = []

        for road_id in road_ids:
            try:
                edges = self._bng.get_road_edges(str(road_id))
                if not edges:
                    continue
                lpts, rpts = [], []
                for e in edges:
                    l = e.get("left") or e.get("leftEdge")
                    r = e.get("right") or e.get("rightEdge")
                    if l is not None and r is not None:
                        lpts.append([float(l[0]), float(l[1]), float(l[2])])
                        rpts.append([float(r[0]), float(r[1]), float(r[2])])
                if len(lpts) >= 2:
                    left_segs.append(np.array(lpts, dtype=np.float32))
                    right_segs.append(np.array(rpts, dtype=np.float32))
            except Exception:
                pass

        if left_segs:
            self._road_left_segs = left_segs
            self._road_right_segs = right_segs
            # Pre-compute mean XY position per segment for fast distance filtering
            self._road_seg_means = np.array(
                [((l + r) / 2.0).mean(axis=0)[:2] for l, r in zip(left_segs, right_segs)],
                dtype=np.float32,
            )
            logger.info("Road edge geometry cached: %d segments", len(left_segs))
        else:
            self._road_left_segs = None
            self._road_right_segs = None
            self._road_seg_means = None
            logger.warning("get_road_edges() returned no usable data")

    def _nodes_ahead(
        self,
        vehicle_state: "VehicleState",
        max_range_m: float,
        lateral_scale_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return (forward, lateral, half_width) arrays for road nodes ahead."""
        if self._road_nodes is None or len(self._road_nodes) == 0:
            return None

        pos = vehicle_state.position
        heading = vehicle_state.heading_rad
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        dx = self._road_nodes[:, 0] - pos[0]
        dy = self._road_nodes[:, 1] - pos[1]
        forward = dx * cos_h + dy * sin_h
        lateral = dx * sin_h - dy * cos_h

        mask = (forward > 0.5) & (forward < max_range_m) & (np.abs(lateral) < lateral_scale_m * 1.5)
        if int(mask.sum()) < 2:
            return None

        fwd = forward[mask]
        lat = lateral[mask]
        hw = self._road_nodes[mask, 2]

        order = np.argsort(fwd)
        return fwd[order], lat[order], hw[order]

    def _attach_roads_sensor(self, vehicle) -> None:
        try:
            from beamngpy.sensors.roads_sensor import RoadsSensor
            self._roads_sensor = RoadsSensor(
                name="roads", bng=self._bng, vehicle=vehicle,
                gfx_update_time=0.01, physics_update_time=0.01,
                is_send_immediately=False, is_visualised=False,
            )
            time.sleep(0.5)
            self._roads_sensor.poll()
            logger.info("RoadsSensor attached")
        except Exception as exc:
            logger.warning("RoadsSensor unavailable: %s", exc)
            self._roads_sensor = None

    def get_road_gt_plan(
        self,
        vehicle_state: "VehicleState",
    ) -> "PathPlan | None":
        """GT plan from RoadsSensor — dist2CL + headingAngle direct from BeamNG."""
        if self._roads_sensor is None:
            return None

        from offroad_autonomy.types import PathPlan

        try:
            rd = self._roads_sensor.poll()
            if isinstance(rd, dict) and rd:
                rd = rd[max(rd.keys(), key=float)]
            if not isinstance(rd, dict) or not rd:
                return None

            lat_m = float(rd.get("dist2CL", 0.0))        # + = right of center
            heading_err = float(rd.get("headingAngle", 0.0))  # rad, relative to road
            half_w = float(rd.get("halfWidth", 3.5))
            if half_w <= 0:
                half_w = 3.5

        except Exception as exc:
            logger.debug("RoadsSensor poll failed: %s", exc)
            return None

        # Use projected road centerline for both visualization and CTE
        centerline = self.get_road_centerline_image(
            vehicle_state,
            img_w=self._config.preprocess_width,
            img_h=self._config.preprocess_height,
        )
        if centerline is None or len(centerline) < 2:
            frame_w = float(self._config.preprocess_width)
            frame_h = float(self._config.preprocess_height)
            cte_norm = float(np.clip(-lat_m / half_w, -1.0, 1.0))
            cx = frame_w / 2.0 + cte_norm * (frame_w / 2.0)
            centerline = np.array([[cx, frame_h * 0.1], [cx, frame_h * 0.9]], dtype=np.float32)

        return PathPlan(
            centerline=centerline,
            heading_rad=float(np.clip(-heading_err, -1.0, 1.0)),
            kalman_active=False,
        )

    def get_road_centerline_image(
        self,
        vehicle_state: "VehicleState",
        img_w: int,
        img_h: int,
        n_points: int = 20,
        max_range_m: float = 40.0,
        lateral_scale_m: float = 8.0,
    ) -> np.ndarray | None:
        """Return GT road centerline as image-space (x,y) array, far-to-near order.

        Centerline = midpoint of left+right road edges from get_road_edges().
        Uses same vehicle-relative coordinate transform as get_road_mask_image.
        """
        if not getattr(self, "_road_left_segs", None):
            return None

        pos = vehicle_state.position
        heading = vehicle_state.heading_rad
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        veh_xy = np.array([pos[0], pos[1]], dtype=np.float32)

        # Gather midpoints from all nearby segments
        fwd_all, lat_all = [], []
        for i in np.where(np.linalg.norm(self._road_seg_means - veh_xy, axis=1) < max_range_m * 1.5)[0]:
            mid = (self._road_left_segs[i] + self._road_right_segs[i]) / 2.0
            dx = mid[:, 0] - pos[0]
            dy = mid[:, 1] - pos[1]
            fwd = dx * cos_h + dy * sin_h
            lat = dx * sin_h - dy * cos_h
            ahead = (fwd > 0.5) & (fwd < max_range_m)
            fwd_all.extend(fwd[ahead].tolist())
            lat_all.extend(lat[ahead].tolist())

        if len(fwd_all) < 2:
            return None

        fwd_arr = np.array(fwd_all, dtype=np.float32)
        lat_arr = np.array(lat_all, dtype=np.float32)

        # Sort by forward distance (near → far) then resample to n_points far→near
        order = np.argsort(fwd_arr)
        fwd_arr = fwd_arr[order]
        lat_arr = lat_arr[order]

        t_in = np.linspace(0.0, 1.0, len(fwd_arr))
        t_out = np.linspace(1.0, 0.0, n_points)  # far first, near last (pts[-1]=nearest)
        fwd_r = np.interp(t_out, t_in, fwd_arr)
        lat_r = np.interp(t_out, t_in, lat_arr)

        y_img = np.clip(img_h * (1.0 - fwd_r / max_range_m), 0.0, float(img_h - 1))
        x_img = np.clip(img_w / 2.0 + lat_r * (img_w / 2.0) / lateral_scale_m, 0.0, float(img_w - 1))
        return np.stack([x_img, y_img], axis=1).astype(np.float32)

    def get_road_mask_image(
        self,
        vehicle_state: "VehicleState",
        img_w: int,
        img_h: int,
        max_range_m: float = 40.0,
        lateral_scale_m: float = 8.0,
    ) -> np.ndarray | None:
        """Project road edge geometry into image space using vehicle-relative coords.

        Same coordinate transform as get_road_centerline_image — forward maps to
        y (far=top, near=bottom), lateral maps to x (center=straight ahead).
        Only processes segments within max_range_m for speed.
        """
        if not getattr(self, "_road_left_segs", None):
            return None

        pos = vehicle_state.position
        heading = vehicle_state.heading_rad
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        veh_xy = np.array([pos[0], pos[1]], dtype=np.float32)

        # Distance-filter: only process segments whose mean XY is within range
        seg_means = self._road_seg_means
        d = np.linalg.norm(seg_means - veh_xy, axis=1)
        nearby = np.where(d < max_range_m * 1.5)[0]
        if len(nearby) == 0:
            return None

        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        def _to_image(pts3d: np.ndarray) -> np.ndarray:
            """World XY pts (N,3) → image (x,y) px using vehicle-relative transform."""
            dx = pts3d[:, 0] - pos[0]
            dy = pts3d[:, 1] - pos[1]
            forward = dx * cos_h + dy * sin_h
            lateral = dx * sin_h - dy * cos_h
            # forward=0 → bottom (near), forward=max_range → top (far)
            y_img = img_h * (1.0 - forward / max_range_m)
            x_img = img_w / 2.0 + lateral * (img_w / 2.0) / lateral_scale_m
            return np.stack([x_img, y_img], axis=1).astype(np.float32)

        for i in nearby:
            left_pts = self._road_left_segs[i]
            right_pts = self._road_right_segs[i]

            l_px = _to_image(left_pts)
            r_px = _to_image(right_pts)

            # Keep only points that are ahead and within lateral bounds
            l_fwd = left_pts[:, 0] * cos_h + left_pts[:, 1] * sin_h - (pos[0] * cos_h + pos[1] * sin_h)
            visible = (l_fwd > 0.5) & (l_fwd < max_range_m)
            if visible.sum() < 2:
                continue

            l_vis = l_px[visible]
            r_vis = r_px[visible]

            poly = np.vstack([l_vis, r_vis[::-1]]).astype(np.int32)
            poly[:, 0] = np.clip(poly[:, 0], 0, img_w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, img_h - 1)
            cv2.fillPoly(mask, [poly], 255)

        return (mask > 0) if mask.any() else None

    def get_annotation_map(self) -> dict:
        """Return the raw get_annotations() color map, or empty dict."""
        return getattr(self, "_raw_annotation_map", {})

    def get_road_annotation_color(self) -> np.ndarray | None:
        """Return the BGR annotation color for the road, or None if unknown."""
        return getattr(self, "_road_annotation_color", None)

    def _find_road_annotation_color(self) -> np.ndarray | None:
        """Query BeamNG's annotation color map and return the road BGR color."""
        _ROAD_KEYWORDS = ("road", "asphalt", "track", "pavement", "tarmac", "lane")
        try:
            color_map = self._bng.get_annotations()
            self._raw_annotation_map = color_map
            logger.info("=== BeamNG annotation classes ===")
            for name, color in sorted(color_map.items()):
                logger.info("  %-40s  %s", name, color)
            # Find the first entry whose name contains a road-related keyword
            for name, color in color_map.items():
                if any(kw in name.lower() for kw in _ROAD_KEYWORDS):
                    # color may be [r, g, b] or (r, g, b) — convert to BGR
                    r, g, b = int(color[0]), int(color[1]), int(color[2])
                    logger.info("Matched road class '%s' -> RGB=(%d,%d,%d)", name, r, g, b)
                    return np.array([b, g, r], dtype=np.int16)
            logger.warning("No road-related class found in annotation map")
        except Exception as exc:
            logger.warning("get_annotations() failed: %s", exc)
        return None

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
