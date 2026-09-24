"""Operator dashboard.

Layout::

    +---------------------------------------------+-----------------+
    | main view: DISPLAY camera + mask + path     | VEHICLE         |
    |   (or a debug view, keys 0-8)               | PERCEPTION      |
    +---------------+---------------+-------------+ STEREO / DEPTH  |
    | LEFT RECTIFIED| RIGHT RECTIFIED| STEREO DEPTH| RUNTIME         |
    +---------------+---------------+-------------+-----------------+

The bottom strip shows the bumper cameras, which produce everything the
vehicle computes. The main panel shows the display camera, which computes
nothing; its label says DISPLAY ONLY so a screenshot cannot be misread as
evidence that the stack is looking through it.

Everything is drawn at the working resolution and only then fitted to the
window, so the dashboard costs a few milliseconds rather than tens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from offroad_autonomy.perception.stereo_rectification import draw_epipolar_pair
from offroad_autonomy.visualization.path_projector import GroundProjector
from offroad_autonomy.types import (
    DEBUG_VIEWS,
    DEFAULT_DASHBOARD_COLORS,
    GMSL2_CAPTURE_SENSOR,
    CameraSensor,
    DepthResult,
    PathPlan,
    PipelineStepResult,
    TerrainAnalysis,
)

_VIEW_TITLES = {
    "default": "PATH VISUALIZATION",
    "raw": "RAW LEFT | RIGHT",
    "rectified": "RECTIFIED LEFT | RIGHT",
    "disparity": "DISPARITY (rectified left)",
    "depth": "DEPTH (segmentation view)",
    "stitched": "STITCHED WIDE VIEW",
    "mask": "SEGMENTATION MASK",
    "fused": "FUSED MASK + DEPTH",
    "roi": "DEPTH ROI",
    "pipeline": "PIPELINE DEBUG  raw | mask | planner input | path",
}


@dataclass
class DashboardTelemetry:
    speed_mph: float
    steering: float
    throttle: float
    brake: float
    perception_confidence: float
    stability_score: float
    kalman_active: bool
    fps: float
    latency_ms: float
    latency_p95_ms: float = 0.0
    autopilot_active: bool = True
    road_fraction: float = 0.0
    ego_coverage: float = 0.0
    segmentation_mode: str = "left"
    fallback_state: str = "NONE"
    depth_active: bool = False
    depth_state: str = "OFF"
    depth_coverage: float = 0.0
    valid_disparity_fraction: float = 0.0
    median_forward_depth_m: float = float("nan")
    min_corridor_depth_m: float = float("nan")
    min_clearance_m: float = float("inf")
    depth_age_ms: float = float("inf")
    stereo_fps: float = 0.0
    stereo_latency_ms: float = 0.0
    stereo_latency_p95_ms: float = 0.0
    #: Never an autonomy number: it can drop to single figures without the
    #: vehicle noticing.
    dashboard_fps: float = 0.0
    sync_ok: bool = True
    timing_lines: list[str] = field(default_factory=list)


class AutonomyDashboard:
    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        colors: dict[str, tuple[int, int, int]] | None = None,
        sensor: CameraSensor | None = None,
        projector: GroundProjector | None = None,
        display_clip: np.ndarray | None = None,
    ) -> None:
        self.width = width
        self.height = height
        if sensor is None:
            sensor = GMSL2_CAPTURE_SENSOR
        self.sensor = sensor
        # Without a projector the main panel falls back to the perception
        # view, so the dashboard still works on a rig with no display camera.
        self.projector = projector
        self.display_clip = display_clip
        self._pad = 24
        self._gap = 20
        self._header_h = 52
        self._sidebar_w = 420
        self._strip_fraction = 0.28
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        # Distinct from MASK_FILL and from BAD so the excluded region cannot be
        # mistaken for traversable road or for a detected obstacle.
        self._colors.setdefault("EGO_EXCLUDED", (168, 96, 220))
        if colors is not None:
            self._colors.update(colors)
        self._ego_cache: dict[tuple, tuple[np.ndarray, list]] = {}
        # Filling a large array with a colour costs ~5 ms at 1600x900;
        # copying a cached blank costs ~1 ms, so blanks are made once.
        self._blanks: dict[tuple, np.ndarray] = {}
        # Depth updates at the stereo rate, the dashboard at the loop rate,
        # so the colour-mapped view is reused until a new result arrives.
        self._depth_view_src: tuple = ()
        self._depth_view: np.ndarray | None = None

    def _blank(self, height: int, width: int, fill: tuple[int, int, int]) -> np.ndarray:
        key = (height, width, tuple(fill))
        blank = self._blanks.get(key)
        if blank is None:
            if len(self._blanks) > 16:
                self._blanks.clear()
            blank = np.full((height, width, 3), fill, dtype=np.uint8)
            self._blanks[key] = blank
        return blank.copy()

    def render(
        self,
        result: PipelineStepResult,
        telemetry: DashboardTelemetry,
        plan: PathPlan | None = None,
        valid_roi: np.ndarray | None = None,
        debug_view: str = "default",
        timing_overlay: bool = False,
        stitched: np.ndarray | None = None,
        depth_roi: np.ndarray | None = None,
        display_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """``plan`` is passed separately because it is withheld under safe
        stop: perception keeps running, but nothing should suggest the stack
        is steering while it is not."""
        if debug_view not in DEBUG_VIEWS:
            debug_view = "default"

        canvas = self._blank(self.height, self.width, self._colors["BG"])
        self._draw_header(canvas, telemetry.autopilot_active)

        content_top = self._pad + self._header_h
        content_h = self.height - content_top - self._pad
        main_w = self.width - (self._pad * 2) - self._gap - self._sidebar_w

        main_rect = (self._pad, content_top, main_w, content_h)
        side_rect = (self._pad + main_w + self._gap, content_top, self._sidebar_w, content_h)
        self._draw_panel(canvas, main_rect)
        self._draw_panel(canvas, side_rect)

        inner_pad = 16
        inner_x = main_rect[0] + inner_pad
        inner_y = main_rect[1] + inner_pad
        inner_w = main_rect[2] - inner_pad * 2
        inner_h = main_rect[3] - inner_pad * 2

        strip_h = int(round(inner_h * self._strip_fraction))
        viewport_rect = (inner_x, inner_y, inner_w, inner_h - strip_h - self._gap)
        strip_rect = (inner_x, inner_y + inner_h - strip_h, inner_w, strip_h)

        main_image = self._main_view(
            debug_view, result, plan, valid_roi, stitched, depth_roi, display_frame
        )
        on_display_camera = debug_view == "default" and self._can_reproject(display_frame)
        viewport = self._fit_image(
            main_image, viewport_rect[2], viewport_rect[3], fill=self._colors["PANEL_BG"]
        )
        self._blit(canvas, viewport, viewport_rect[0], viewport_rect[1])
        self._draw_viewport_labels(canvas, viewport_rect, telemetry, debug_view, on_display_camera)
        if debug_view == "default":
            self._draw_perception_legend(
                canvas,
                viewport_rect,
                has_ego=valid_roi is not None and not bool(valid_roi.all()),
                reprojected=on_display_camera,
            )
        if timing_overlay and telemetry.timing_lines:
            self._draw_timing_overlay(canvas, viewport_rect, telemetry.timing_lines)
        if not telemetry.autopilot_active:
            self._draw_safe_stop_overlay(canvas, viewport_rect)

        self._draw_camera_strip(canvas, strip_rect, result.depth, result.terrain, telemetry)
        self._draw_sidebar(canvas, side_rect, telemetry)
        return canvas

    def _main_view(
        self,
        view: str,
        result: PipelineStepResult,
        plan: PathPlan | None,
        valid_roi: np.ndarray | None,
        stitched: np.ndarray | None,
        depth_roi: np.ndarray | None,
        display_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        frame = result.frame.preprocessed
        depth = result.depth
        pair = result.frames

        if view == "raw":
            if pair is not None and pair.left is not None and pair.right is not None:
                return np.hstack([self._ensure_bgr(pair.left), self._ensure_bgr(pair.right)])
            return self._placeholder(frame, "PAIR INCOMPLETE")
        if view == "rectified":
            if depth is not None and depth.rectified_left is not None:
                return draw_epipolar_pair(depth.rectified_left, depth.rectified_right)
            return self._placeholder(frame, "NO STEREO RESULT")
        if view == "disparity":
            image = self._render_disparity(depth)
            if image is None:
                return self._placeholder(frame, "NO STEREO RESULT")
            return image
        if view == "depth":
            image = self._render_depth_view(depth, result.terrain)
            if image is None:
                return self._placeholder(frame, "NO DEPTH")
            return image
        if view == "stitched":
            if stitched is not None:
                return stitched
            return self._placeholder(frame, "STITCHING DISABLED (stitching.enabled)")
        if view == "mask":
            return self._render_mask_view(result)
        if view == "fused":
            return self._render_fused_view(result)
        if view == "roi":
            return self._render_roi_view(frame, depth_roi, depth)
        if view == "pipeline":
            return self._render_pipeline_view(result)

        if self._can_reproject(display_frame):
            return self._build_display_overlay(display_frame, result, plan)

        mask = self._ensure_mask(result.stabilized.mask, frame.shape[:2])
        ego = None
        if valid_roi is not None and not bool(valid_roi.all()):
            ego = ~self._ensure_mask(valid_roi, frame.shape[:2])
        return self._build_overlay(frame, mask, plan, result.stabilized.mask.shape[:2], ego)

    def _can_reproject(self, display_frame: np.ndarray | None) -> bool:
        return display_frame is not None and self.projector is not None

    def _build_display_overlay(
        self,
        display_frame: np.ndarray,
        result: PipelineStepResult,
        plan: PathPlan | None,
    ) -> np.ndarray:
        """Only a redraw of results that already exist, so the large view
        costs the same whether or not anyone is watching it."""
        assert self.projector is not None
        target = self._ensure_bgr(display_frame)
        projector = self.projector
        if target.shape[:2] != (projector.target.height, projector.target.width):
            target = cv2.resize(
                target,
                (projector.target.width, projector.target.height),
                interpolation=cv2.INTER_AREA,
            )

        mask = projector.warp_mask(result.stabilized.mask)
        if self.display_clip is not None and self.display_clip.shape == mask.shape:
            mask = mask & ~self.display_clip

        projected = None
        if plan is not None and len(plan.centerline) >= 2:
            points = projector.project_points(plan.centerline)
            if len(points) >= 2:
                projected = replace(plan, centerline=points)

        return self._build_overlay(target, mask, projected, target.shape[:2], ego=None)

    def _placeholder(self, like: np.ndarray, text: str) -> np.ndarray:
        image = np.full(like.shape[:2] + (3,), self._colors["CARD_BG"], dtype=np.uint8)
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(
            image,
            text,
            ((image.shape[1] - size[0]) // 2, image.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )
        return image

    def _render_disparity(self, depth: DepthResult | None) -> np.ndarray | None:
        if depth is None or depth.disparity is None:
            return None
        disparity = depth.disparity
        valid = disparity > 0.0
        if not valid.any():
            return None
        top = max(float(np.percentile(disparity[valid], 99)), 1.0)
        indexed = (np.clip(disparity / top, 0.0, 1.0) * 255.0).astype(np.uint8)
        coloured = cv2.applyColorMap(indexed, cv2.COLORMAP_TURBO)
        coloured[~valid] = self._colors["CARD_BG"]
        return coloured

    def _render_mask_view(self, result: PipelineStepResult) -> np.ndarray:
        stable = result.stabilized.mask
        raw = _model_mask(result)
        image = np.full(stable.shape + (3,), self._colors["CARD_BG"], dtype=np.uint8)
        image[raw.astype(bool)] = (60, 100, 70)
        image[stable] = self._colors["MASK_FILL"]
        contours, _ = cv2.findContours(
            stable.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, self._colors["MASK_EDGE"], 2, lineType=cv2.LINE_AA)
        return image

    def _render_pipeline_view(self, result: PipelineStepResult) -> np.ndarray:
        """Uses ``result.plan``, which is kept even under safe stop, so this
        view always shows what the planner decided."""
        frame = result.frame.preprocessed
        h, w = frame.shape[:2]
        raw = cv2.resize(self._ensure_bgr(result.frame.raw), (w, h), interpolation=cv2.INTER_AREA)
        perception = result.perception
        plan = result.plan

        def tint(image, mask, color, alpha=0.45):
            out = image.copy()
            mask = self._ensure_mask(mask, (h, w))
            out[mask] = (out[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
            return out

        def caption(image, lines):
            for i, line in enumerate(lines):
                y = 20 + 18 * i
                cv2.putText(
                    image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA
                )
                cv2.putText(
                    image,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            return image

        confs = perception.confidences
        seg = caption(
            tint(frame, _model_mask(result), (0, 200, 255)),
            [
                "2 SEGMENTATION (model)",
                f"{len(confs)} det  max conf {max(confs, default=0.0):.2f}",
                f"road {perception.road_fraction:.1%} of valid px",
            ],
        )

        processed = (frame * 0.4).astype(np.uint8)
        processed = tint(processed, result.stabilized.mask, (90, 90, 90), 0.6)
        if plan.planner_mask is not None:
            processed = tint(processed, plan.planner_mask, (0, 255, 0), 0.7)
        cv2.line(processed, (0, plan.roi_top), (w - 1, plan.roi_top), (255, 255, 0), 1)
        processed = caption(
            processed,
            ["3 PLANNER INPUT", "green = road component, grey = rest", "cyan line = ROI top"],
        )

        planner_mask = plan.planner_mask
        if planner_mask is None:
            planner_mask = np.zeros((h, w), bool)
        traj = tint(frame, planner_mask, (0, 255, 0), 0.25)
        status = "perceived path"
        path_color = (255, 200, 0)
        if plan.fallback_active:
            status = plan.fallback_reason
            path_color = (0, 165, 255)
        if len(plan.centerline) >= 2:
            pts = np.round(plan.centerline).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(traj, [pts], False, path_color, 4, cv2.LINE_AA)
        cv2.line(traj, (w // 2, h - 1), (w // 2, h - 30), (255, 255, 255), 2)
        lines = [
            "4 TRAJECTORY + CONTROL",
            status[:48],
            f"thr {result.command.throttle:.2f}  brk {result.command.brake:.2f}",
        ]
        dbg = result.command.debug
        if dbg is not None:
            edge_color = _level_color(dbg.edge_risk, warn=0.0, bad=0.66, ok=(0, 255, 255))
            for (ul, vl), (ur, vr) in dbg.boundary_px:
                for u, v in ((ul, vl), (ur, vr)):
                    if 3 < u < w - 4:
                        cv2.drawMarker(
                            traj, (int(u), int(v)), edge_color, cv2.MARKER_TILTED_CROSS, 9, 2
                        )
            if dbg.lookahead_px is not None:
                lx, ly = (int(round(v)) for v in dbg.lookahead_px)
                cv2.circle(traj, (lx, ly), 9, (0, 0, 0), 3, cv2.LINE_AA)
                marker = _level_color(dbg.saturation, warn=0.5, bad=0.85, ok=(255, 0, 255))
                cv2.circle(traj, (lx, ly), 9, marker, 2, cv2.LINE_AA)
                cv2.line(traj, (w // 2, h - 1), (lx, ly), (255, 0, 255), 1, cv2.LINE_AA)

            def edge(d):
                if not math.isfinite(d):
                    return "n/a"
                return f"{d:.2f}"

            curvature_line = f"curvature {dbg.curvature:+.3f} /m (straight)"
            if abs(dbg.curvature) > 1e-3:
                curvature_line = (
                    f"curvature {dbg.curvature:+.3f} /m (R {abs(1.0 / dbg.curvature):.0f} m)"
                )
            lookahead_line = f"lookahead {dbg.lookahead_m:.1f} m  rejoin {dbg.rejoin_m:.1f} m"
            if dbg.lookahead_px is None:
                lookahead_line += " (no point)"

            lines += [
                f"cross-track {dbg.cross_track_m:+.2f} m  ({dbg.cross_track_rate_mps:+.2f} m/s)",
                f"trail edge L {edge(dbg.left_boundary_m)} m  R {edge(dbg.right_boundary_m)} m",
                f"edge risk {dbg.edge_risk * 100:.0f}%  clear {edge(dbg.edge_clearance_m)} m",
                f"lat accel {dbg.lateral_accel_mps2:.2f} m/s2",
                f"heading err {math.degrees(dbg.heading_error_rad):+.1f} deg",
                curvature_line,
                f"steer desired {dbg.desired_steering:+.3f}  ff {dbg.feedforward_steering:+.2f}",
                f"steer final   {dbg.final_steering:+.3f}",
                f"saturation {dbg.saturation * 100:.0f}%  (sharpest ahead {dbg.max_curvature_ahead:.2f} /m)",
                f"target {dbg.target_speed_mps:.1f} m/s ({dbg.speed_reason})",
                lookahead_line,
            ]
        traj = caption(traj, lines)

        top = np.hstack([caption(raw, ["1 RAW RGB"]), seg])
        bottom = np.hstack([processed, traj])
        grid = np.vstack([top, bottom])
        cv2.line(grid, (w, 0), (w, 2 * h), (255, 255, 255), 2)
        cv2.line(grid, (0, h), (2 * w, h), (255, 255, 255), 2)
        # The view-title chip gets its own band so it covers no caption.
        header = np.full((80, grid.shape[1], 3), self._colors["PANEL_BG"], dtype=np.uint8)
        return np.vstack([header, grid])

    def _render_fused_view(self, result: PipelineStepResult) -> np.ndarray:
        frame = result.frame.preprocessed.copy()
        trav = result.traversability
        if trav is None:
            return self._placeholder(frame, "NO FUSED DEPTH (appearance only)")
        conf = np.clip(trav.confidence_map, 0.0, 1.0)
        coloured = cv2.applyColorMap((conf * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        road = trav.binary_mask
        frame[road] = cv2.addWeighted(frame, 0.35, coloured, 0.65, 0)[road]
        frame[trav.obstacle_mask] = self._colors["BAD"]
        edges = cv2.Canny(trav.valid_depth_mask.astype(np.uint8) * 255, 50, 150) > 0
        frame[edges] = self._colors["WARN"]
        return frame

    def _render_roi_view(
        self,
        frame: np.ndarray,
        depth_roi: np.ndarray | None,
        depth: DepthResult | None,
    ) -> np.ndarray:
        if depth_roi is None or depth_roi.shape != frame.shape[:2]:
            return self._placeholder(frame, "DEPTH ROI DISABLED")
        image = (frame * 0.25).astype(np.uint8)
        image[depth_roi] = frame[depth_roi]
        if depth is not None and depth.valid.shape == depth_roi.shape:
            hits = depth.valid & depth_roi
            image[hits] = cv2.addWeighted(
                image, 0.5, np.full_like(image, self._colors["WARN"]), 0.5, 0
            )[hits]
        return image

    def _draw_camera_strip(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        depth: DepthResult | None,
        terrain: TerrainAnalysis | None,
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        gap = 12
        tile_w = (w - gap * 2) // 3

        age = ""
        if math.isfinite(telemetry.depth_age_ms):
            age = f"  {telemetry.depth_age_ms:.0f} ms old"
        rect_left = None
        rect_right = None
        if depth is not None:
            rect_left = depth.rectified_left
            rect_right = depth.rectified_right
        tiles = (
            ("LEFT RECTIFIED", rect_left),
            ("RIGHT RECTIFIED", rect_right),
            ("STEREO DEPTH" + age, self._render_depth_view(depth, terrain)),
        )
        placeholder = telemetry.depth_state
        if telemetry.depth_state == "OFF":
            placeholder = "NO SIGNAL"

        for index, (title, image) in enumerate(tiles):
            tile_x = x + index * (tile_w + gap)
            self._draw_card(canvas, (tile_x, y, tile_w, h), title)

            body_rect = (tile_x + 8, y + 24, tile_w - 16, h - 32)
            if body_rect[2] <= 0 or body_rect[3] <= 0:
                continue
            if image is None:
                self._draw_tile_placeholder(canvas, body_rect, placeholder)
                continue
            fitted = self._fit_image(
                self._ensure_bgr(image), body_rect[2], body_rect[3], fill=self._colors["CARD_BG"]
            )
            self._blit(canvas, fitted, body_rect[0], body_rect[1])

    def _draw_tile_placeholder(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        text: str,
    ) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BG"], -1)
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            canvas,
            text,
            (x + (w - size[0]) // 2, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

    def _render_depth_view(
        self,
        depth: DepthResult | None,
        terrain: TerrainAnalysis | None,
    ) -> np.ndarray | None:
        """Inferred pixels are dimmed so it stays obvious at a glance how much
        of the frame is measurement rather than an assumption about the
        ground."""
        if depth is None:
            return None
        # Keyed on array identity: the pipeline passes an age-corrected copy
        # of terrain every frame, but the maps inside it are shared until
        # stereo publishes a new result.
        range_m = None
        if terrain is not None:
            range_m = terrain.range_m
        source = (depth.depth_m, range_m)
        cached = self._depth_view_src
        if cached and all(a is b for a, b in zip(source, cached)):
            return self._depth_view
        view = self._colour_depth(depth, terrain)
        self._depth_view_src, self._depth_view = source, view
        return view

    def _colour_depth(
        self,
        depth: DepthResult,
        terrain: TerrainAnalysis | None,
    ) -> np.ndarray | None:
        # Prefer the terrain range map: it covers the ground-plane fill too.
        if terrain is not None and terrain.range_m is not None:
            depth_m, coverage = terrain.range_m, terrain.valid
        else:
            depth_m, coverage = depth.depth_m, depth.valid

        finite = coverage & np.isfinite(depth_m) & (depth_m > 0.0)
        if not finite.any():
            return None

        # Scale by inverse depth, not depth: a linear ramp spends its range on
        # ground the vehicle will not reach for seconds and crushes the near
        # field into one hue. Inverse depth is also what stereo measures.
        near, far = 2.0, 40.0
        inverse = np.divide(1.0, depth_m, out=np.zeros_like(depth_m), where=depth_m > 1e-3)
        span = (1.0 / near) - (1.0 / far)
        normalised = np.clip((inverse - 1.0 / far) / span, 0.0, 1.0)
        indexed = (normalised * 255.0).astype(np.uint8)
        coloured = cv2.applyColorMap(indexed, cv2.COLORMAP_TURBO)
        coloured[~finite] = self._colors["CARD_BG"]

        if terrain is not None:
            if terrain.inferred is not None and terrain.inferred.any():
                inferred = terrain.inferred & finite
                coloured[inferred] = (coloured[inferred] * 0.45).astype(np.uint8)
            if terrain.obstacle_mask.any():
                coloured[terrain.obstacle_mask] = self._colors["BAD"]

        return coloured

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def _ensure_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if mask.shape[:2] == shape and mask.dtype == bool:
            return mask
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.shape[:2] != shape:
            mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_u8.astype(bool)

    def _build_overlay(
        self,
        camera: np.ndarray,
        mask: np.ndarray,
        plan: PathPlan | None,
        plan_shape: tuple[int, int],
        ego: np.ndarray | None = None,
    ) -> np.ndarray:
        display = camera.copy()

        if ego is not None and ego.any():
            self._draw_ego_exclusion(display, ego)

        if mask.any():
            tinted = display.copy()
            tinted[mask] = self._colors["MASK_FILL"]
            cv2.addWeighted(tinted, 0.28, display, 0.72, 0, display)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                display, contours, -1, self._colors["MASK_EDGE"], 2, lineType=cv2.LINE_AA
            )

        if plan is not None and len(plan.centerline) >= 2:
            pts = plan.centerline.astype(np.float32).copy()
            src_h, src_w = plan_shape
            dst_h, dst_w = camera.shape[:2]
            if src_h > 0 and src_w > 0 and (src_h, src_w) != (dst_h, dst_w):
                pts[:, 0] *= dst_w / src_w
                pts[:, 1] *= dst_h / src_h

            pts = self._smooth_points(pts)
            pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

            glow = display.copy()
            cv2.polylines(glow, [pts_i], False, self._colors["PATH_GLOW"], 12, lineType=cv2.LINE_AA)
            cv2.addWeighted(glow, 0.22, display, 0.78, 0, display)
            cv2.polylines(
                display, [pts_i], False, self._colors["PATH_CORE"], 4, lineType=cv2.LINE_AA
            )
            cv2.polylines(
                display, [pts_i], False, self._colors["PATH_HIGHLIGHT"], 1, lineType=cv2.LINE_AA
            )

            target = tuple(np.round(pts[max(0, len(pts) // 3)]).astype(int))
            cv2.circle(display, target, 6, self._colors["PATH_CORE"], -1, lineType=cv2.LINE_AA)
            cv2.circle(display, target, 6, self._colors["PATH_HIGHLIGHT"], 1, lineType=cv2.LINE_AA)

        return display

    def _ego_layer(self, ego: np.ndarray) -> tuple[np.ndarray, list]:
        """Cached because the exclusion never changes during a run."""
        key = (ego.shape, int(ego.sum()))
        cached = self._ego_cache.get(key)
        if cached is None:
            h, w = ego.shape
            hatch = np.zeros((h, w), dtype=np.uint8)
            for offset in range(-h, w, 14):
                cv2.line(hatch, (offset, 0), (offset + h, h), 1, 1)
            stripes = ego & hatch.astype(bool)
            contours, _ = cv2.findContours(
                ego.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cached = (stripes, list(contours))
            self._ego_cache = {key: cached}
        return cached

    def _draw_ego_exclusion(self, display: np.ndarray, ego: np.ndarray) -> None:
        """Hatched so it reads as excluded, not as terrain."""
        colour = self._colors["EGO_EXCLUDED"]
        stripes, contours = self._ego_layer(ego)

        tinted = display.copy()
        tinted[ego] = colour
        tinted[stripes] = colour
        cv2.addWeighted(tinted, 0.3, display, 0.7, 0, display)
        cv2.drawContours(display, contours, -1, colour, 1, lineType=cv2.LINE_AA)

        if contours:
            biggest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(biggest)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                text = "EGO - EXCLUDED"
                size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.putText(
                    display,
                    text,
                    (cx - size[0] // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (245, 240, 250),
                    1,
                    cv2.LINE_AA,
                )

    def _draw_perception_legend(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        has_ego: bool,
        reprojected: bool = False,
    ) -> None:
        x, y, w, h = rect
        suffix = ""
        box_w = 214
        if reprojected:
            suffix = " (from bumper pair)"
            box_w = 320
        entries = [
            (f"Traversable mask{suffix}", self._colors["MASK_FILL"]),
            (f"Planned path{suffix}", self._colors["PATH_CORE"]),
        ]
        if has_ego:
            entries.append(("Ego vehicle (excluded)", self._colors["EGO_EXCLUDED"]))

        pad = 10
        row_h = 20
        box_h = pad * 2 + row_h * len(entries)
        bx = x + w - box_w - 14
        by = y + h - box_h - 14

        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), (22, 26, 30), -1)
        cv2.rectangle(canvas, (bx, by), (bx + box_w, by + box_h), self._colors["CARD_BORDER"], 1)
        for index, (label, colour) in enumerate(entries):
            row_y = by + pad + row_h * index + 13
            cv2.rectangle(canvas, (bx + pad, row_y - 9), (bx + pad + 14, row_y + 2), colour, -1)
            cv2.putText(
                canvas,
                label,
                (bx + pad + 22, row_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                self._colors["TEXT_PRIMARY"],
                1,
                cv2.LINE_AA,
            )

    def _draw_timing_overlay(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        lines: list[str],
    ) -> None:
        x, y, _, h = rect
        row_h = 17
        box_w = 360
        box_h = min(h - 60, 16 + row_h * len(lines))
        bx, by = x + 14, y + 52
        region = canvas[by : by + box_h, bx : bx + box_w]
        region[:] = (region * 0.25).astype(np.uint8)
        for index, line in enumerate(lines):
            row_y = by + 16 + row_h * index
            if row_y > by + box_h - 4:
                break
            cv2.putText(
                canvas,
                line,
                (bx + 10, row_y),
                cv2.FONT_HERSHEY_PLAIN,
                1.0,
                self._colors["TEXT_PRIMARY"],
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _smooth_points(points: np.ndarray, samples: int = 120) -> np.ndarray:
        if len(points) < 2:
            return points
        diffs = np.diff(points, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        arc = np.concatenate(([0.0], np.cumsum(lengths)))
        total = float(arc[-1])
        if total <= 1e-6:
            return points
        q = np.linspace(0.0, total, max(samples, len(points)))
        xs = np.interp(q, arc, points[:, 0])
        ys = np.interp(q, arc, points[:, 1])
        return np.stack([xs, ys], axis=1)

    def _fit_image(
        self, image: np.ndarray, target_w: int, target_h: int, fill: tuple[int, int, int]
    ) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        out = self._blank(target_h, target_w, fill)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        out[off_y : off_y + new_h, off_x : off_x + new_w] = resized
        return out

    @staticmethod
    def _blit(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
        h, w = image.shape[:2]
        canvas[y : y + h, x : x + w] = image

    def _draw_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["PANEL_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)

    def _draw_header(self, canvas: np.ndarray, autopilot_active: bool = True) -> None:
        x = self._pad
        y = self._pad
        w = self.width - self._pad * 2
        h = self._header_h

        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["PANEL_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)
        cv2.putText(
            canvas,
            "OFF-ROAD AUTONOMY  |  FRONT PATH VIEW",
            (x + 20, y + 34),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )

        if autopilot_active:
            chip_text, chip_color = "AUTONOMY ACTIVE", self._colors["GOOD"]
        else:
            chip_text, chip_color = "SAFE STOP  |  MANUAL CONTROL", self._colors["BAD"]
        chip_size, _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        self._draw_chip(canvas, x + w - chip_size[0] - 44, y + 14, chip_text, chip_color)

    def _draw_viewport_labels(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
        debug_view: str,
        on_display_camera: bool = False,
    ) -> None:
        x, y, _, _ = rect
        source = telemetry.segmentation_mode.upper()
        # The label has to answer "is the stack looking through this?" at a
        # glance; on the display camera the answer is always no.
        if on_display_camera:
            label = "PATH VISUALIZATION  -  DISPLAY ONLY"
        elif source == "STITCHED":
            label = "STITCHED WIDE VIEW  (PERCEPTION)"
        else:
            label = f"{source} {self.sensor.model}  {self.sensor.fov_x_deg:.0f} deg  (PERCEPTION)"
        if debug_view == "default":
            self._draw_chip(canvas, x + 18, y + 18, label, self._colors["WARN"])
        else:
            index = DEBUG_VIEWS.index(debug_view)
            self._draw_chip(
                canvas,
                x + 18,
                y + 18,
                f"[{index}] {_VIEW_TITLES[debug_view]}   (0 = back)",
                self._colors["PATH_CORE"],
            )
        if not telemetry.sync_ok:
            self._draw_chip(
                canvas,
                x + 18,
                y + 48,
                "PAIR NOT SYNCHRONISED - STEREO SKIPPED",
                self._colors["BAD"],
            )

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        inner_x = x + 16
        inner_y = y + 16
        inner_w = w - 32
        gap = 14

        available_h = h - 32 - gap * 3
        vehicle_h = int(round(available_h * 0.23))
        perception_h = int(round(available_h * 0.23))
        runtime_h = int(round(available_h * 0.21))
        stereo_h = available_h - vehicle_h - perception_h - runtime_h

        self._draw_vehicle_card(canvas, (inner_x, inner_y, inner_w, vehicle_h), telemetry)
        inner_y += vehicle_h + gap
        self._draw_perception_card(canvas, (inner_x, inner_y, inner_w, perception_h), telemetry)
        inner_y += perception_h + gap
        self._draw_stereo_card(canvas, (inner_x, inner_y, inner_w, stereo_h), telemetry)
        inner_y += stereo_h + gap
        self._draw_runtime_card(canvas, (inner_x, inner_y, inner_w, runtime_h), telemetry)

    def _draw_vehicle_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "VEHICLE")
        self._draw_value_row(
            canvas, x + 18, y + 44, w - 36, "Vehicle speed", f"{telemetry.speed_mph:4.1f} mph"
        )
        self._draw_value_row(
            canvas, x + 18, y + h - 78, w - 36, "Steering", f"{telemetry.steering:+.2f}"
        )
        self._draw_progress_row(
            canvas, x + 18, y + h - 52, w - 36, "Throttle", telemetry.throttle, self._colors["GOOD"]
        )
        self._draw_progress_row(
            canvas, x + 18, y + h - 24, w - 36, "Brake", telemetry.brake, self._colors["BAD"]
        )

    def _draw_perception_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "PERCEPTION")
        self._draw_progress_row(
            canvas,
            x + 18,
            y + min(46, h - 74),
            w - 36,
            "Segmentation confidence",
            telemetry.perception_confidence,
            self._colors["WARN"],
        )
        # Road fraction, not confidence, is what the safe stop watches.
        self._draw_progress_row(
            canvas,
            x + 18,
            y + min(84, h - 46),
            w - 36,
            "Road / valid px",
            telemetry.road_fraction,
            self._colors["GOOD"],
        )
        self._draw_value_row(
            canvas,
            x + 18,
            y + h - 40,
            w - 36,
            "Valid ROI / ego",
            f"{(1.0 - telemetry.ego_coverage) * 100:.0f}%  /  {telemetry.ego_coverage * 100:.0f}%",
        )
        fallback = telemetry.fallback_state
        fallback_color = self._colors["BAD"]
        if fallback == "NONE":
            fallback_color = self._colors["GOOD"]
        self._draw_value_row(
            canvas, x + 18, y + h - 16, w - 36, "Fallback", fallback, value_color=fallback_color
        )

    def _draw_stereo_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "STEREO / DEPTH")
        state_color = {
            "LIVE": self._colors["GOOD"],
            "WARMING UP": self._colors["WARN"],
        }.get(telemetry.depth_state, self._colors["BAD"])
        self._draw_value_row(
            canvas, x + 18, y + 42, w - 36, "Depth state", telemetry.depth_state, state_color
        )
        if not telemetry.depth_active:
            return

        rows = np.linspace(y + 76, y + h - 14, 6).astype(int)
        self._draw_progress_row(
            canvas,
            x + 18,
            int(rows[0]),
            w - 36,
            "Depth coverage",
            telemetry.depth_coverage,
            self._colors["WARN"],
        )
        self._draw_progress_row(
            canvas,
            x + 18,
            int(rows[1]),
            w - 36,
            "Valid disparity",
            telemetry.valid_disparity_fraction,
            self._colors["GOOD"],
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[2]),
            w - 36,
            "Median forward depth",
            _metres(telemetry.median_forward_depth_m),
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[3]),
            w - 36,
            "Min corridor depth",
            _metres(telemetry.min_corridor_depth_m),
        )
        clear = telemetry.min_clearance_m
        clear_text = "CLEAR"
        clear_color = self._colors["GOOD"]
        if math.isfinite(clear):
            clear_text = f"{clear:.1f} m"
            if clear < 2.5:
                clear_color = self._colors["BAD"]
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[4]),
            w - 36,
            "Obstacle clearance",
            clear_text,
            clear_color,
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[5]),
            w - 36,
            "Stereo update rate",
            f"{telemetry.stereo_fps:.1f} Hz",
        )

    def _draw_runtime_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "RUNTIME")
        # Separate rows for separate threads: the autonomy numbers contain no
        # drawing or GUI time, so a slow window cannot pose as a slow vehicle.
        rows = np.linspace(y + 42, y + h - 14, 5).astype(int)
        fps_color = self._colors["BAD"]
        if telemetry.fps >= 20.0:
            fps_color = self._colors["GOOD"]
        latency_color = self._colors["BAD"]
        if telemetry.latency_p95_ms <= 50.0:
            latency_color = self._colors["GOOD"]
        self._draw_value_row(
            canvas, x + 18, int(rows[0]), w - 36, "Autonomy FPS", f"{telemetry.fps:.1f}", fps_color
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[1]),
            w - 36,
            "Autonomy latency (mean/p95)",
            f"{telemetry.latency_ms:.0f} / {telemetry.latency_p95_ms:.0f} ms",
            latency_color,
        )
        self._draw_value_row(
            canvas, x + 18, int(rows[2]), w - 36, "Stereo FPS", f"{telemetry.stereo_fps:.1f}"
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[3]),
            w - 36,
            "Stereo latency (mean/p95)",
            f"{telemetry.stereo_latency_ms:.0f} / {telemetry.stereo_latency_p95_ms:.0f} ms",
        )
        self._draw_value_row(
            canvas,
            x + 18,
            int(rows[4]),
            w - 36,
            "Dashboard FPS",
            f"{telemetry.dashboard_fps:.1f}",
            self._colors["TEXT_SECONDARY"],
        )

    def _draw_card(self, canvas: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)
        cv2.putText(
            canvas,
            title,
            (x + 14, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

    def _draw_value_row(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        width: int,
        label: str,
        value: str,
        value_color: tuple[int, int, int] | None = None,
    ) -> None:
        if value_color is None:
            value_color = self._colors["TEXT_PRIMARY"]
        cv2.putText(
            canvas,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )
        text_size, _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            canvas,
            value,
            (x + width - text_size[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            value_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_progress_row(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        color: tuple[int, int, int],
    ) -> None:
        value_clamped = 0.0
        if math.isfinite(value):
            value_clamped = float(np.clip(value, 0.0, 1.0))
        val_str = f"{value_clamped:.2f}"
        val_size, _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bar_w = width - (val_size[0] + 10)

        cv2.putText(
            canvas,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            val_str,
            (x + width - val_size[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        bar_y = y + 8
        cv2.rectangle(canvas, (x, bar_y), (x + bar_w, bar_y + 8), self._colors["MUTED_LINE"], -1)
        fill_w = int(round(bar_w * value_clamped))
        if fill_w > 0:
            cv2.rectangle(canvas, (x, bar_y), (x + fill_w, bar_y + 8), color, -1)

    def _draw_safe_stop_overlay(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        region = canvas[y : y + h, x : x + w]
        red = np.full_like(region, (30, 30, 160))
        cv2.addWeighted(red, 0.35, region, 0.65, 0, region)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 220), 3)

        banner_h = 72
        banner_y = y + h // 2 - banner_h // 2
        cv2.rectangle(canvas, (x, banner_y), (x + w, banner_y + banner_h), (20, 20, 100), -1)
        cv2.rectangle(canvas, (x, banner_y), (x + w, banner_y + banner_h), (60, 60, 220), 2)

        line1 = "SAFE STOP  -  MANUAL CONTROL REQUIRED"
        sz1, _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_DUPLEX, 0.78, 1)
        cv2.putText(
            canvas,
            line1,
            (x + (w - sz1[0]) // 2, banner_y + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.78,
            (180, 180, 255),
            1,
            cv2.LINE_AA,
        )
        line2 = "W/A/S/D = drive  |  SPACE = brake  |  P = resume autopilot"
        sz2, _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(
            canvas,
            line2,
            (x + (w - sz2[0]) // 2, banner_y + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (155, 163, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_chip(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        text: str,
        accent: tuple[int, int, int],
    ) -> None:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        w = text_size[0] + 26
        h = 24
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (36, 42, 48), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), accent, 1)
        cv2.rectangle(canvas, (x + 8, y + 7), (x + 14, y + 13), accent, -1)
        cv2.putText(
            canvas,
            text,
            (x + 20, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )


def _metres(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.1f} m"


def _model_mask(result: PipelineStepResult) -> np.ndarray:
    """The segmenter's own mask, before depth carved anything out of it."""
    if result.perception.rgb_mask is not None:
        return result.perception.rgb_mask
    return result.perception.mask


def _level_color(
    value: float,
    warn: float,
    bad: float,
    ok: tuple[int, int, int],
) -> tuple[int, int, int]:
    if value > bad:
        return (0, 0, 255)
    if value > warn:
        return (0, 165, 255)
    return ok
