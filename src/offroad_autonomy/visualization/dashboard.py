"""Dashcam imagery, segmentation and planned path from the same capture."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from offroad_autonomy.types import (
    DEBUG_VIEW_KEYS,
    DEBUG_VIEWS,
    DEFAULT_DASHBOARD_COLORS,
    GMSL2_CAPTURE_SENSOR,
    CameraSensor,
    DashboardThresholds,
    PathPlan,
    PipelineStepResult,
)
from offroad_autonomy.visualization import bands
from offroad_autonomy.visualization.layout import Rect, build_layout
from offroad_autonomy.visualization.text import (
    BODY,
    BODY_STRONG,
    CAPTION,
    CAPTION_MONO,
    DISPLAY,
    HERO,
    HERO_SANS,
    LABEL,
    LARGE,
    LARGE_SANS,
    TITLE,
    VALUE,
    TextRenderer,
    TextStyle,
)
from offroad_autonomy.visualization.widgets import (
    CoverTransform,
    blend_rect,
    blit_cover,
    chip_height,
    chip_width,
    cover_point,
    draw_bar,
    draw_centered_bar,
    draw_chip,
    fill_rect,
    outline_rect,
    wrap_text,
)

_VIEW_TITLES = {
    "default": "Path Visualization",
    "raw": "Raw Dashcam",
    "mask": "Segmentation Mask",
    "pipeline": "Pipeline Debug",
}

_ALL_STYLES: tuple[TextStyle, ...] = (
    DISPLAY,
    HERO,
    HERO_SANS,
    LARGE,
    LARGE_SANS,
    TITLE,
    VALUE,
    BODY,
    BODY_STRONG,
    LABEL,
    CAPTION,
    CAPTION_MONO,
)

# Row positions inside the cards, measured down from the card's top edge. The
# static layer and the per-frame values read the same numbers, so a label and
# its value cannot drift apart.
_CARD_TITLE_BASELINE = 30
_CARD_INSET = 20
_SPEED_BASELINE = 128
_STEER_BASELINE = 172
_STEER_BAR_TOP = 184
_PEDAL_BASELINE = 246
_PEDAL_BAR_TOP = 258
_ROAD_BASELINE = 64
_ROAD_BAR_TOP = 76
_ROAD_CAPTION_BASELINE = 108
_CONF_BASELINE = 146
_CONF_BAR_TOP = 158
_CONF_CAPTION_BASELINE = 190
_ROI_BASELINE = 230
_FALLBACK_BASELINE = 270
_REASON_LINE_HEIGHT = 22
_CARD_BOTTOM_MARGIN = 14

_TILE_TITLE_BASELINE = 26
_TILE_VALUE_BASELINE = 76
_TILE_BAR_TOP = 90
_TILE_BAR_H = 8
_DASHBOARD_TILE_VALUE_BASELINE = 68
_DASHBOARD_TILE_NOTE_BASELINE = 94

# Frame pixels below the hood's top edge.
_EGO_LABEL_DROP_PX = 22
EGO_LABEL = "EGO - EXCLUDED"

_PIPELINE_READOUT_W = 330
_READOUT_ROW_H = 22


class DashcamOverlay(NamedTuple):
    """The dashcam frame with perception and plan drawn on, before scaling.

    ``ego_label_px`` is where the hood label goes, in frame pixels. The label
    is left to whoever scales the image, since text drawn here would be
    scaled with it.
    """

    image: np.ndarray
    ego_label_px: tuple[int, int] | None


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
    segmentation_mode: str = "dashcam"
    fallback_state: str = "NONE"
    fallback_reason: str = ""
    #: Never an autonomy number: it can drop to single figures without the
    #: vehicle noticing.
    dashboard_fps: float = 0.0
    timing_lines: list[str] = field(default_factory=list)


class AutonomyDashboard:
    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        colors: dict[str, tuple[int, int, int]] | None = None,
        sensor: CameraSensor | None = None,
        thresholds: DashboardThresholds | None = None,
    ) -> None:
        self.width = width
        self.height = height
        if sensor is None:
            sensor = GMSL2_CAPTURE_SENSOR
        self.sensor = sensor
        if thresholds is None:
            thresholds = DashboardThresholds()
        self._thresholds = thresholds
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        if colors is not None:
            self._colors.update(colors)
        self._layout = build_layout(width, height)
        # Loaded here so a missing font stops startup, not the first frame.
        self._text = TextRenderer()
        self._text.preload(_ALL_STYLES)
        self._fallback_label_w = self._text.width("Fallback", BODY)
        self._ego_cache: dict[tuple, tuple[np.ndarray, list, tuple[int, int] | None]] = {}
        #: Centre of the hood exclusion in frame pixels, set while the default
        #: overlay is built so its label can be drawn after the frame is scaled.
        self._ego_label_px: tuple[int, int] | None = None
        # Filling a large array with a colour costs ~5 ms at 1600x900;
        # copying a cached blank costs ~1 ms, so everything that never changes
        # between frames is drawn once into one image and copied.
        self._static = self._build_static()

    # ------------------------------------------------------------------ frame

    def render(
        self,
        result: PipelineStepResult,
        telemetry: DashboardTelemetry,
        plan: PathPlan | None = None,
        valid_roi: np.ndarray | None = None,
        debug_view: str = "default",
        timing_overlay: bool = False,
    ) -> np.ndarray:
        """``plan`` is passed separately because it is withheld under safe
        stop: perception keeps running, but nothing should suggest the stack
        is steering while it is not."""
        if debug_view not in DEBUG_VIEWS:
            debug_view = "default"

        canvas = self._static.copy()
        self._draw_status_chip(canvas, telemetry.autopilot_active)

        viewport = self._layout.viewport
        has_ego = valid_roi is not None and not bool(valid_roi.all())
        self._ego_label_px = None
        if debug_view == "pipeline":
            self._draw_pipeline_view(canvas, viewport, result)
        else:
            image = self._main_view(debug_view, result, plan, valid_roi)
            transform = blit_cover(canvas, image, viewport)
            if self._ego_label_px is not None:
                self._draw_ego_label(canvas, transform, self._ego_label_px)
        self._draw_viewport_labels(canvas, viewport, debug_view)
        if debug_view in ("default", "raw"):
            self._draw_mask_inset(canvas, viewport, result)
        if debug_view == "default":
            self._draw_perception_legend(canvas, viewport, plan, has_ego)
            if plan is not None and plan.fallback_active:
                self._draw_fallback_chip(canvas, viewport)
        if timing_overlay and telemetry.timing_lines:
            self._draw_timing_overlay(canvas, viewport, telemetry.timing_lines)
        if not telemetry.autopilot_active:
            self._draw_safe_stop_overlay(canvas, viewport)

        self._draw_vehicle_card(canvas, telemetry)
        self._draw_perception_card(canvas, telemetry)
        self._draw_runtime_strip(canvas, telemetry)
        return canvas

    def _main_view(
        self,
        view: str,
        result: PipelineStepResult,
        plan: PathPlan | None,
        valid_roi: np.ndarray | None,
    ) -> np.ndarray:
        frame = result.frame.preprocessed
        if view == "raw":
            return self._ensure_bgr(result.frame.raw)
        if view == "mask":
            return self._render_mask_view(result)

        mask = self._ensure_mask(result.stabilized.mask, frame.shape[:2])
        ego = None
        if valid_roi is not None and not bool(valid_roi.all()):
            ego = ~self._ensure_mask(valid_roi, frame.shape[:2])
        return self._build_overlay(frame, mask, plan, result.stabilized.mask.shape[:2], ego)

    def dashcam_overlay(
        self,
        result: PipelineStepResult,
        plan: PathPlan | None,
        valid_roi: np.ndarray | None,
    ) -> DashcamOverlay:
        """The image the default view shows, for other layouts to scale.

        ``plan`` is None under safe stop, which leaves the path and its
        lookahead dot off, as on the dashboard."""
        self._ego_label_px = None
        image = self._main_view("default", result, plan, valid_roi)
        return DashcamOverlay(image, self._ego_label_px)

    # ----------------------------------------------------------- static layer

    def _build_static(self) -> np.ndarray:
        colors = self._colors
        layout = self._layout
        canvas = np.full((self.height, self.width, 3), colors["BG"], dtype=np.uint8)

        header = layout.header
        fill_rect(canvas, header, colors["PANEL_BG"])
        outline_rect(canvas, header, colors["CARD_BORDER"])
        title_w = self._text.draw(
            canvas,
            "OFF-ROAD AUTONOMY",
            header.x + 20,
            header.y + 36,
            TITLE,
            colors["TEXT_PRIMARY"],
        )
        self._text.draw(
            canvas,
            "Front Path View",
            header.x + 20 + title_w + 16,
            header.y + 35,
            BODY,
            colors["TEXT_SECONDARY"],
        )

        self._static_card(canvas, layout.vehicle, "Vehicle")
        vehicle = layout.vehicle
        self._static_label(canvas, vehicle, "Steering", _STEER_BASELINE)
        half_gap = 20
        column_w = (vehicle.w - 2 * _CARD_INSET - half_gap) // 2
        self._text.draw(
            canvas,
            "Throttle",
            vehicle.x + _CARD_INSET,
            vehicle.y + _PEDAL_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )
        self._text.draw(
            canvas,
            "Brake",
            vehicle.x + _CARD_INSET + column_w + half_gap,
            vehicle.y + _PEDAL_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )

        self._static_card(canvas, layout.perception, "Perception")
        perception = layout.perception
        thresholds = self._thresholds
        self._static_label(canvas, perception, "Road / Valid Px", _ROAD_BASELINE)
        self._static_caption(
            canvas,
            perception,
            f"Tick: Safe Stop Threshold {thresholds.road_floor:g}",
            _ROAD_CAPTION_BASELINE,
        )
        self._static_label(canvas, perception, "Segmentation Confidence", _CONF_BASELINE)
        self._static_caption(
            canvas,
            perception,
            f"Tick: Gate Minimum {thresholds.confidence_floor:g}",
            _CONF_CAPTION_BASELINE,
        )
        self._static_label(canvas, perception, "Valid ROI / Ego", _ROI_BASELINE)
        self._static_label(canvas, perception, "Fallback", _FALLBACK_BASELINE)

        for rect, title in (
            (layout.autonomy_fps, "Autonomy FPS"),
            (layout.latency, "Autonomy Latency Mean / P95"),
            (layout.dashboard_fps, "Dashboard FPS"),
        ):
            fill_rect(canvas, rect, colors["CARD_BG"])
            outline_rect(canvas, rect, colors["CARD_BORDER"])
            self._text.draw(
                canvas,
                title.upper(),
                rect.x + _CARD_INSET,
                rect.y + _TILE_TITLE_BASELINE,
                LABEL,
                colors["TEXT_SECONDARY"],
            )
        # The note that keeps a slow window from posing as a slow vehicle.
        self._text.draw(
            canvas,
            "Display Only, Not The Vehicle",
            layout.dashboard_fps.x + _CARD_INSET,
            layout.dashboard_fps.y + _DASHBOARD_TILE_NOTE_BASELINE,
            CAPTION,
            colors["TEXT_SECONDARY"],
        )
        return canvas

    def _static_card(self, canvas: np.ndarray, rect: Rect, title: str) -> None:
        fill_rect(canvas, rect, self._colors["CARD_BG"])
        outline_rect(canvas, rect, self._colors["CARD_BORDER"])
        self._text.draw(
            canvas,
            title.upper(),
            rect.x + _CARD_INSET,
            rect.y + _CARD_TITLE_BASELINE,
            LABEL,
            self._colors["TEXT_SECONDARY"],
        )

    def _static_label(self, canvas: np.ndarray, card: Rect, text: str, baseline: int) -> None:
        self._text.draw(
            canvas,
            text,
            card.x + _CARD_INSET,
            card.y + baseline,
            BODY,
            self._colors["TEXT_SECONDARY"],
        )

    def _static_caption(self, canvas: np.ndarray, card: Rect, text: str, baseline: int) -> None:
        self._text.draw(
            canvas,
            text,
            card.x + _CARD_INSET,
            card.y + baseline,
            CAPTION,
            self._colors["TEXT_SECONDARY"],
        )

    # ----------------------------------------------------------------- header

    def _draw_status_chip(self, canvas: np.ndarray, autopilot_active: bool) -> None:
        header = self._layout.header
        if autopilot_active:
            text, color, solid = "AUTONOMY ACTIVE", self._colors["GOOD"], False
        else:
            text, color, solid = "SAFE STOP | MANUAL CONTROL", self._colors["BAD"], True
        height = chip_height(self._text)
        draw_chip(
            canvas,
            self._text,
            header.right - 20,
            header.y + (header.h - height) // 2,
            text,
            color,
            self._colors["BG"],
            solid=solid,
            align="right",
        )

    # --------------------------------------------------------------- viewport

    def _draw_viewport_labels(self, canvas: np.ndarray, rect: Rect, debug_view: str) -> None:
        if debug_view == "default":
            text = f"Dashcam {self.sensor.model} {self.sensor.fov_x_deg:.0f} Deg"
            color = self._colors["TEXT_PRIMARY"]
        else:
            index = next(key for key, view in DEBUG_VIEW_KEYS.items() if view == debug_view)
            text = f"[{index}] {_VIEW_TITLES[debug_view]} (0 = Back)"
            color = self._colors["PATH_CORE"]
        self._draw_plate_text(canvas, rect.x + 16, rect.y + 16, text, BODY_STRONG, color)

    def _draw_plate_text(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        text: str,
        style: TextStyle,
        color: tuple[int, int, int],
    ) -> None:
        """Text on a dark translucent plate, so it reads over any scene."""
        cap = self._text.cap_height(style)
        plate = Rect(x, y, self._text.width(text, style) + 24, cap + 20)
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        self._text.draw(canvas, text, x + 12, y + 10 + cap, style, color)

    def _draw_mask_inset(
        self, canvas: np.ndarray, viewport: Rect, result: PipelineStepResult
    ) -> None:
        inset_w, inset_h = 172, 104
        inset = Rect(viewport.x + 16, viewport.bottom - 16 - inset_h, inset_w, inset_h)
        self._text.draw(
            canvas,
            "MASK",
            inset.x,
            inset.y - 8,
            LABEL,
            self._colors["TEXT_PRIMARY"],
        )
        image = self._render_mask_view(result)
        fitted = self._fit_image(image, inset.w, inset.h, fill=self._colors["CARD_BG"])
        canvas[inset.y : inset.bottom, inset.x : inset.right] = fitted
        outline_rect(canvas, inset, self._colors["CARD_BORDER"])

    def _draw_perception_legend(
        self,
        canvas: np.ndarray,
        viewport: Rect,
        plan: PathPlan | None,
        has_ego: bool,
    ) -> None:
        entries = [("Traversable Mask", self._colors["MASK_FILL"])]
        if plan is not None:
            if plan.fallback_active:
                entries.append(("Held Path", self._colors["WARN"]))
            else:
                entries.append(("Planned Path", self._colors["PATH_CORE"]))
        if has_ego:
            entries.append(("Ego Vehicle (Excluded)", self._colors["EGO_EXCLUDED"]))

        swatch = 14
        swatch_gap = 8
        entry_gap = 20
        pad_x = 16
        cap = self._text.cap_height(BODY)
        widths = [swatch + swatch_gap + self._text.width(label, BODY) for label, _ in entries]
        plate_w = sum(widths) + entry_gap * (len(entries) - 1) + 2 * pad_x
        plate_h = cap + 24
        plate = Rect(
            viewport.right - 16 - plate_w,
            viewport.bottom - 16 - plate_h,
            plate_w,
            plate_h,
        )
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        x = plate.x + pad_x
        baseline = plate.y + 12 + cap
        for (label, color), width in zip(entries, widths, strict=True):
            fill_rect(canvas, Rect(x, baseline - swatch + 2, swatch, swatch), color)
            self._text.draw(
                canvas,
                label,
                x + swatch + swatch_gap,
                baseline,
                BODY,
                self._colors["TEXT_PRIMARY"],
            )
            x += width + entry_gap

    def _draw_fallback_chip(self, canvas: np.ndarray, viewport: Rect) -> None:
        text = "PATH FALLBACK ACTIVE"
        width = chip_width(self._text, text)
        draw_chip(
            canvas,
            self._text,
            viewport.x + (viewport.w - width) // 2,
            viewport.y + 20,
            text,
            self._colors["WARN"],
            self._colors["BG"],
            solid=True,
        )

    def _draw_ego_label(
        self, canvas: np.ndarray, transform: CoverTransform, anchor: tuple[int, int]
    ) -> None:
        x, y = cover_point(transform, anchor)
        self._text.draw(
            canvas,
            EGO_LABEL,
            x,
            y,
            BODY_STRONG,
            self._colors["TEXT_PRIMARY"],
            align="center",
        )

    def _draw_timing_overlay(self, canvas: np.ndarray, viewport: Rect, lines: list[str]) -> None:
        row_h = 17
        plate_w = 360
        top = viewport.y + 60
        plate_h = min(viewport.h - 120, 16 + row_h * len(lines))
        plate = Rect(viewport.x + 16, top, plate_w, plate_h)
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        for index, line in enumerate(lines):
            baseline = top + 20 + row_h * index
            if baseline > plate.bottom - 4:
                break
            self._text.draw(
                canvas, line, plate.x + 10, baseline, CAPTION_MONO, self._colors["TEXT_PRIMARY"]
            )

    def _draw_safe_stop_overlay(self, canvas: np.ndarray, viewport: Rect) -> None:
        colors = self._colors
        blend_rect(canvas, viewport, colors["BAD"], 0.22)
        outline_rect(canvas, viewport, colors["BAD"], 6)

        banner_w, banner_h = 700, 190
        banner = Rect(
            viewport.x + (viewport.w - banner_w) // 2,
            viewport.y + (viewport.h - banner_h) // 2 - 40,
            banner_w,
            banner_h,
        )
        fill_rect(canvas, banner, colors["CARD_BG"])
        outline_rect(canvas, banner, colors["BAD"], 3)
        centre = banner.x + banner.w // 2
        self._text.draw(
            canvas, "SAFE STOP", centre, banner.y + 70, HERO_SANS, colors["BAD"], align="center"
        )
        self._text.draw(
            canvas,
            "Manual Control Required",
            centre,
            banner.y + 118,
            LARGE_SANS,
            colors["TEXT_PRIMARY"],
            align="center",
        )
        self._text.draw(
            canvas,
            "W/A/S/D Drive  |  Space Brake  |  P Resume Autopilot",
            centre,
            banner.y + 160,
            BODY,
            colors["TEXT_SECONDARY"],
            align="center",
        )

    # ---------------------------------------------------------------- vehicle

    def _draw_vehicle_card(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        card = self._layout.vehicle
        left = card.x + _CARD_INSET
        inner_w = card.w - 2 * _CARD_INSET

        speed = f"{telemetry.speed_mph:.1f}"
        speed_w = self._text.draw(
            canvas, speed, left, card.y + _SPEED_BASELINE, DISPLAY, colors["TEXT_PRIMARY"]
        )
        self._text.draw(
            canvas,
            "mph",
            left + speed_w + 12,
            card.y + _SPEED_BASELINE,
            TITLE,
            colors["TEXT_SECONDARY"],
        )

        direction = "Right"
        if telemetry.steering < 0.0:
            direction = "Left"
        right = card.right - _CARD_INSET
        word_w = self._text.draw(
            canvas,
            direction,
            right,
            card.y + _STEER_BASELINE,
            CAPTION,
            colors["TEXT_SECONDARY"],
            align="right",
        )
        self._text.draw(
            canvas,
            f"{telemetry.steering:+.2f}",
            right - word_w - 8,
            card.y + _STEER_BASELINE,
            VALUE,
            colors["TEXT_PRIMARY"],
            align="right",
        )
        draw_centered_bar(
            canvas,
            Rect(left, card.y + _STEER_BAR_TOP, inner_w, 10),
            telemetry.steering,
            colors["TEXT_PRIMARY"],
            colors["MUTED_LINE"],
            colors["TEXT_SECONDARY"],
        )

        half_gap = 20
        column_w = (inner_w - half_gap) // 2
        for index, (value, color) in enumerate(
            ((telemetry.throttle, colors["GOOD"]), (telemetry.brake, colors["BAD"]))
        ):
            x = left + index * (column_w + half_gap)
            self._text.draw(
                canvas,
                f"{value:.2f}",
                x + column_w,
                card.y + _PEDAL_BASELINE,
                VALUE,
                colors["TEXT_PRIMARY"],
                align="right",
            )
            draw_bar(
                canvas,
                Rect(x, card.y + _PEDAL_BAR_TOP, column_w, 10),
                value,
                color,
                colors["MUTED_LINE"],
            )

    # ------------------------------------------------------------- perception

    def _draw_perception_card(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        thresholds = self._thresholds
        card = self._layout.perception
        left = card.x + _CARD_INSET
        right = card.right - _CARD_INSET
        inner_w = card.w - 2 * _CARD_INSET

        # Road fraction, not confidence, is what the safe stop watches.
        road_color = colors[bands.road_level(telemetry.road_fraction, thresholds.road_floor)]
        self._text.draw(
            canvas,
            f"{telemetry.road_fraction:.3f}",
            right,
            card.y + _ROAD_BASELINE,
            VALUE,
            road_color,
            align="right",
        )
        draw_bar(
            canvas,
            Rect(left, card.y + _ROAD_BAR_TOP, inner_w, 12),
            telemetry.road_fraction / thresholds.road_bar_full_scale,
            road_color,
            colors["MUTED_LINE"],
            tick=thresholds.road_floor / thresholds.road_bar_full_scale,
            tick_color=colors["TEXT_PRIMARY"],
        )

        confidence = telemetry.perception_confidence
        confidence_color = colors[
            bands.confidence_level(
                confidence, thresholds.confidence_floor, thresholds.confidence_good
            )
        ]
        self._text.draw(
            canvas,
            f"{confidence:.2f}",
            right,
            card.y + _CONF_BASELINE,
            VALUE,
            confidence_color,
            align="right",
        )
        draw_bar(
            canvas,
            Rect(left, card.y + _CONF_BAR_TOP, inner_w, 10),
            confidence,
            confidence_color,
            colors["MUTED_LINE"],
            tick=thresholds.confidence_floor,
            tick_color=colors["TEXT_PRIMARY"],
        )

        valid = (1.0 - telemetry.ego_coverage) * 100.0
        self._text.draw(
            canvas,
            f"{valid:.0f}% / {telemetry.ego_coverage * 100:.0f}%",
            right,
            card.y + _ROI_BASELINE,
            VALUE,
            colors["TEXT_PRIMARY"],
            align="right",
        )

        chips_bottom = self._draw_fallback_chips(canvas, card, telemetry.fallback_state)
        self._draw_reason(canvas, card, chips_bottom, telemetry)

    def _draw_fallback_chips(self, canvas: np.ndarray, card: Rect, state: str) -> int:
        """One chip per active fallback, wrapping, since the state is a
        ``" + "`` joined list that can outgrow the card."""
        parts = [part.strip() for part in state.split("+") if part.strip()]
        if not parts:
            parts = ["NONE"]
        color = self._colors["BAD"]
        if state == "NONE":
            color = self._colors["GOOD"]

        chip_h = chip_height(self._text)
        row_start = card.x + _CARD_INSET
        row_end = card.right - _CARD_INSET
        x = row_start + self._fallback_label_w + 12
        # The label's baseline is the chip text's baseline, so the chip sits
        # level with it.
        y = card.y + _FALLBACK_BASELINE - 10 - self._text.cap_height(BODY_STRONG)
        for part in parts:
            width = chip_width(self._text, part)
            if x + width > row_end and x > row_start:
                x = row_start
                y += chip_h + 8
            draw_chip(canvas, self._text, x, y, part, color, self._colors["BG"])
            x += width + 8
        return y + chip_h

    def _draw_reason(
        self, canvas: np.ndarray, card: Rect, chips_bottom: int, telemetry: DashboardTelemetry
    ) -> None:
        reason = telemetry.fallback_reason.strip()
        left = card.x + _CARD_INSET
        if not reason:
            return
        color = self._colors["TEXT_PRIMARY"]
        lines = wrap_text(self._text, reason, BODY, card.w - 2 * _CARD_INSET)
        baseline = chips_bottom + 26
        room = (card.bottom - _CARD_BOTTOM_MARGIN - baseline) // _REASON_LINE_HEIGHT + 1
        room = max(room, 1)
        if len(lines) > room:
            lines = lines[:room]
            lines[-1] = self._ellipsize(lines[-1], card.w - 2 * _CARD_INSET, BODY)
        for index, line in enumerate(lines):
            self._text.draw(canvas, line, left, baseline + _REASON_LINE_HEIGHT * index, BODY, color)

    def _ellipsize(self, text: str, max_width: int, style: TextStyle) -> str:
        while text and self._text.width(text + "...", style) > max_width:
            text = text[:-1]
        return text.rstrip() + "..."

    # ---------------------------------------------------------------- runtime

    def _draw_runtime_strip(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        thresholds = self._thresholds
        layout = self._layout

        fps_level = bands.fps_level(
            telemetry.fps, thresholds.target_fps, thresholds.fps_warn_fraction
        )
        fps_note = f"Below {thresholds.target_fps:g} Fps Target"
        if fps_level == bands.GOOD:
            fps_note = "On Target"
        elif fps_level == bands.WARN:
            fps_note = f"Slightly Below {thresholds.target_fps:g} Fps Target"
        self._draw_health_tile(
            canvas,
            layout.autonomy_fps,
            f"{telemetry.fps:.1f}",
            "fps",
            fps_level,
            fps_note,
            telemetry.fps / (thresholds.target_fps * thresholds.fps_bar_scale),
            1.0 / thresholds.fps_bar_scale,
        )

        latency_level = bands.latency_level(telemetry.latency_p95_ms, thresholds.latency_budget_ms)
        latency_note = f"Over {thresholds.latency_budget_ms:.0f} Ms Budget"
        if latency_level == bands.GOOD:
            latency_note = f"Within {thresholds.latency_budget_ms:.0f} Ms Budget"
        self._draw_health_tile(
            canvas,
            layout.latency,
            f"{telemetry.latency_ms:.0f} / {telemetry.latency_p95_ms:.0f}",
            "ms",
            latency_level,
            latency_note,
            telemetry.latency_p95_ms
            / (thresholds.latency_budget_ms * thresholds.latency_bar_scale),
            1.0 / thresholds.latency_bar_scale,
        )

        tile = layout.dashboard_fps
        self._text.draw(
            canvas,
            f"{telemetry.dashboard_fps:.1f}",
            tile.x + _CARD_INSET,
            tile.y + _DASHBOARD_TILE_VALUE_BASELINE,
            LARGE,
            colors["TEXT_SECONDARY"],
        )

    def _draw_health_tile(
        self,
        canvas: np.ndarray,
        tile: Rect,
        value: str,
        unit: str,
        level: str,
        note: str,
        fraction: float,
        tick: float,
    ) -> None:
        color = self._colors[level]
        if level != bands.GOOD:
            outline_rect(canvas, tile, color, 2)
        left = tile.x + _CARD_INSET
        baseline = tile.y + _TILE_VALUE_BASELINE
        value_w = self._text.draw(canvas, value, left, baseline, HERO, color)
        self._text.draw(
            canvas, unit, left + value_w + 10, baseline, BODY, self._colors["TEXT_SECONDARY"]
        )
        self._text.draw(
            canvas,
            note,
            tile.right - _CARD_INSET,
            tile.y + _TILE_TITLE_BASELINE,
            LABEL,
            color,
            align="right",
        )
        draw_bar(
            canvas,
            Rect(left, tile.y + _TILE_BAR_TOP, tile.w - 2 * _CARD_INSET, _TILE_BAR_H),
            fraction,
            color,
            self._colors["MUTED_LINE"],
            tick=tick,
            tick_color=self._colors["TEXT_PRIMARY"],
        )

    # ------------------------------------------------------------ debug views

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

    def _render_pipeline_view(self, result: PipelineStepResult) -> tuple[np.ndarray, int, int]:
        """The 2 x 2 grid without any text, so it can be scaled freely. Returns
        the grid and the size of one quadrant.

        Uses ``result.plan``, which is kept even under safe stop, so this
        view always shows what the planner decided."""
        frame = result.frame.preprocessed
        h, w = frame.shape[:2]
        raw = cv2.resize(self._ensure_bgr(result.frame.raw), (w, h), interpolation=cv2.INTER_AREA)
        plan = result.plan

        def tint(image, mask, color, alpha=0.45):
            out = image.copy()
            mask = self._ensure_mask(mask, (h, w))
            out[mask] = (out[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
            return out

        seg = tint(frame, _model_mask(result), (0, 200, 255))

        processed = (frame * 0.4).astype(np.uint8)
        processed = tint(processed, result.stabilized.mask, (90, 90, 90), 0.6)
        if plan.planner_mask is not None:
            processed = tint(processed, plan.planner_mask, (0, 255, 0), 0.7)
        cv2.line(processed, (0, plan.roi_top), (w - 1, plan.roi_top), (255, 255, 0), 1)

        planner_mask = plan.planner_mask
        if planner_mask is None:
            planner_mask = np.zeros((h, w), bool)
        traj = tint(frame, planner_mask, (0, 255, 0), 0.25)
        path_color = (255, 200, 0)
        if plan.fallback_active:
            path_color = (0, 165, 255)
        if len(plan.centerline) >= 2:
            pts = np.round(plan.centerline).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(traj, [pts], False, path_color, 4, cv2.LINE_AA)
        cv2.line(traj, (w // 2, h - 1), (w // 2, h - 30), (255, 255, 255), 2)
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

        grid = np.vstack([np.hstack([raw, seg]), np.hstack([processed, traj])])
        cv2.line(grid, (w, 0), (w, 2 * h), (255, 255, 255), 2)
        cv2.line(grid, (0, h), (2 * w, h), (255, 255, 255), 2)
        return grid, w, h

    def _pipeline_readout(self, result: PipelineStepResult) -> list[tuple[str, str | None]]:
        """Label and value rows for the debug panel. A ``None`` value is a heading.

        Drawn as dashboard text rather than burned into the quadrants, where
        scaling the grid to fit made it unreadable."""
        perception = result.perception
        plan = result.plan
        command = result.command
        confs = perception.confidences
        status = "Perceived Path"
        if plan.fallback_active:
            status = plan.fallback_reason
        rows: list[tuple[str, str | None]] = [
            ("Segmentation", None),
            ("Detections", str(len(confs))),
            ("Max Confidence", f"{max(confs, default=0.0):.2f}"),
            ("Road Of Valid Px", f"{perception.road_fraction:.1%}"),
            ("Planner", None),
            ("Status", status),
            ("Throttle / Brake", f"{command.throttle:.2f} / {command.brake:.2f}"),
        ]
        dbg = command.debug
        if dbg is None:
            return rows

        def meters(value):
            if not math.isfinite(value):
                return "n/a"
            return f"{value:.2f} m"

        curvature = "Straight"
        if abs(dbg.curvature) > 1e-3:
            curvature = f"{dbg.curvature:+.3f} /m (R {abs(1.0 / dbg.curvature):.0f} m)"
        lookahead = f"{dbg.lookahead_m:.1f} m"
        if dbg.lookahead_px is None:
            lookahead += " (No Point)"

        rows += [
            ("Controller", None),
            ("Cross-Track", f"{dbg.cross_track_m:+.2f} m ({dbg.cross_track_rate_mps:+.2f} m/s)"),
            ("Trail Edge Left", meters(dbg.left_boundary_m)),
            ("Trail Edge Right", meters(dbg.right_boundary_m)),
            ("Edge Risk", f"{dbg.edge_risk * 100:.0f}%"),
            ("Edge Clearance", meters(dbg.edge_clearance_m)),
            ("Lateral Accel", f"{dbg.lateral_accel_mps2:.2f} m/s2"),
            ("Heading Error", f"{math.degrees(dbg.heading_error_rad):+.1f} deg"),
            ("Curvature", curvature),
            ("Steer Desired", f"{dbg.desired_steering:+.3f} (ff {dbg.feedforward_steering:+.2f})"),
            ("Steer Final", f"{dbg.final_steering:+.3f}"),
            ("Saturation", f"{dbg.saturation * 100:.0f}%"),
            ("Sharpest Ahead", f"{dbg.max_curvature_ahead:.2f} /m"),
            ("Target Speed", f"{dbg.target_speed_mps:.1f} m/s ({dbg.speed_reason})"),
            ("Lookahead", lookahead),
            ("Rejoin", f"{dbg.rejoin_m:.1f} m"),
        ]
        return rows

    def _draw_pipeline_view(
        self, canvas: np.ndarray, viewport: Rect, result: PipelineStepResult
    ) -> None:
        colors = self._colors
        fill_rect(canvas, viewport, colors["PANEL_BG"])

        readout = Rect(
            viewport.right - _PIPELINE_READOUT_W, viewport.y, _PIPELINE_READOUT_W, viewport.h
        )
        # Below the view chip, which sits over the top-left corner.
        chip_clearance = 64
        region = Rect(
            viewport.x,
            viewport.y + chip_clearance,
            viewport.w - _PIPELINE_READOUT_W - 16,
            viewport.h - chip_clearance,
        )
        grid, quad_w, quad_h = self._render_pipeline_view(result)
        scale = min(region.w / grid.shape[1], region.h / grid.shape[0])
        new_w = max(1, int(round(grid.shape[1] * scale)))
        new_h = max(1, int(round(grid.shape[0] * scale)))
        resized = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)
        off_x = region.x + (region.w - new_w) // 2
        off_y = region.y + (region.h - new_h) // 2
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized

        titles = (
            ("1 Raw RGB", 0, 0),
            ("2 Segmentation (Model)", 1, 0),
            ("3 Planner Input", 0, 1),
            ("4 Trajectory + Control", 1, 1),
        )
        for title, column, row in titles:
            self._draw_plate_text(
                canvas,
                off_x + int(round(column * quad_w * scale)) + 8,
                off_y + int(round(row * quad_h * scale)) + 8,
                title.upper(),
                LABEL,
                colors["TEXT_PRIMARY"],
            )

        fill_rect(canvas, readout, colors["CARD_BG"])
        outline_rect(canvas, readout, colors["CARD_BORDER"])
        left = readout.x + 16
        right = readout.right - 16
        baseline = readout.y + 28
        for label, value in self._pipeline_readout(result):
            if baseline > readout.bottom - 8:
                break
            if value is None:
                baseline += 8
                self._text.draw(
                    canvas, label.upper(), left, baseline, LABEL, colors["TEXT_SECONDARY"]
                )
            else:
                label_w = self._text.draw(
                    canvas, label, left, baseline, CAPTION, colors["TEXT_SECONDARY"]
                )
                value = self._ellipsize_left_room(value, right - left - label_w - 12)
                self._text.draw(
                    canvas,
                    value,
                    right,
                    baseline,
                    CAPTION_MONO,
                    colors["TEXT_PRIMARY"],
                    align="right",
                )
            baseline += _READOUT_ROW_H

    def _ellipsize_left_room(self, text: str, max_width: int) -> str:
        if self._text.width(text, CAPTION_MONO) <= max_width:
            return text
        return self._ellipsize(text, max_width, CAPTION_MONO)

    # ---------------------------------------------------------------- overlay

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

            core = self._colors["PATH_CORE"]
            if plan.fallback_active:
                # Dashed and amber: the path is being held, not freshly perceived.
                core = self._colors["WARN"]
                dash, gap = 8, 6
                for start in range(0, len(pts_i), dash + gap):
                    cv2.polylines(
                        display,
                        [pts_i[start : start + dash]],
                        False,
                        core,
                        4,
                        lineType=cv2.LINE_AA,
                    )
            else:
                glow = display.copy()
                cv2.polylines(
                    glow, [pts_i], False, self._colors["PATH_GLOW"], 12, lineType=cv2.LINE_AA
                )
                cv2.addWeighted(glow, 0.22, display, 0.78, 0, display)
                cv2.polylines(display, [pts_i], False, core, 4, lineType=cv2.LINE_AA)
                cv2.polylines(
                    display,
                    [pts_i],
                    False,
                    self._colors["PATH_HIGHLIGHT"],
                    1,
                    lineType=cv2.LINE_AA,
                )

            target = tuple(np.round(pts[max(0, len(pts) // 3)]).astype(int))
            cv2.circle(display, target, 6, core, -1, lineType=cv2.LINE_AA)
            cv2.circle(display, target, 6, self._colors["PATH_HIGHLIGHT"], 1, lineType=cv2.LINE_AA)

        return display

    def _ego_layer(self, ego: np.ndarray) -> tuple[np.ndarray, list, tuple[int, int] | None]:
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
            centre = None
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                moments = cv2.moments(biggest)
                if moments["m00"] > 0:
                    # Near the top edge, not the centroid: the bottom of the
                    # frame is where the legend sits and would cover the label.
                    top = int(biggest[:, 0, 1].min())
                    centre = (int(moments["m10"] / moments["m00"]), top + _EGO_LABEL_DROP_PX)
            cached = (stripes, list(contours), centre)
            self._ego_cache = {key: cached}
        return cached

    def _draw_ego_exclusion(self, display: np.ndarray, ego: np.ndarray) -> None:
        """Hatched so it reads as excluded, not as terrain."""
        colour = self._colors["EGO_EXCLUDED"]
        stripes, contours, centre = self._ego_layer(ego)

        tinted = display.copy()
        tinted[ego] = colour
        tinted[stripes] = colour
        cv2.addWeighted(tinted, 0.3, display, 0.7, 0, display)
        cv2.drawContours(display, contours, -1, colour, 1, lineType=cv2.LINE_AA)
        # The label is drawn on the canvas, in the dashboard font, once the
        # frame has been scaled: text drawn here would be scaled with it.
        self._ego_label_px = centre

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

    @staticmethod
    def _fit_image(
        image: np.ndarray, target_w: int, target_h: int, fill: tuple[int, int, int]
    ) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        out = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        out[off_y : off_y + new_h, off_x : off_x + new_w] = resized
        return out


def _model_mask(result: PipelineStepResult) -> np.ndarray:
    """The segmenter's mask before temporal stabilisation."""
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
