"""The 1920 x 1080 presentation video frame: orbit view, dashcam overlay, stats.

Shown live with ``--presentation-view`` or recorded with ``--presentation``.
The dashcam overlay comes from the operator dashboard, so both draw the same thing.
"""

from __future__ import annotations

import numpy as np

from offroad_autonomy.types import (
    DEFAULT_DASHBOARD_COLORS,
    CameraSensor,
    DashboardThresholds,
    PathPlan,
    PipelineStepResult,
)
from offroad_autonomy.visualization import bands
from offroad_autonomy.visualization.dashboard import (
    EGO_LABEL,
    AutonomyDashboard,
    DashboardTelemetry,
)
from offroad_autonomy.visualization.layout import Rect
from offroad_autonomy.visualization.text import (
    BODY,
    BODY_STRONG,
    CAPTION,
    HEADLINE,
    LABEL,
    LABEL_SPACED,
    METRIC,
    PLATE,
    SPEED,
    SUBTITLE,
    UNIT,
    VALUE,
    TextRenderer,
    TextStyle,
)
from offroad_autonomy.visualization.widgets import (
    blend_rect,
    blit_cover,
    chip_height,
    cover_point,
    draw_bar,
    draw_centered_bar,
    draw_chip,
    fill_rect,
    outline_rect,
)

PRESENTATION_WIDTH = 1920
PRESENTATION_HEIGHT = 1080

HEADER = Rect(32, 32, 1856, 64)
ORBIT_VIEW = Rect(32, 116, 1228, 932)
DASHCAM_CARD = Rect(1280, 116, 608, 412)
STATS_CARD = Rect(1280, 548, 608, 500)

_ALL_STYLES: tuple[TextStyle, ...] = (
    BODY,
    BODY_STRONG,
    CAPTION,
    HEADLINE,
    LABEL,
    LABEL_SPACED,
    METRIC,
    PLATE,
    SPEED,
    SUBTITLE,
    UNIT,
    VALUE,
)

_HEADER_INSET = 24
_STATS_PAD_X = 24
_STATS_PAD_TOP = 20

# Offsets below the stats card's inner top edge. They reproduce the approved
# mockup's CSS line boxes with these fonts' metrics, so a value lands where
# the mockup puts it rather than where a cap height would.
_SPEED_BASELINE = 80
_TOP_ROW_H = 92
_STEER_W = 230
_STEER_BASELINE = 68
_STEER_BAR_TOP = 82
_STEER_BAR_H = 10
_GRID_TOP = _TOP_ROW_H + 22
_COLUMN_W = 266
_COLUMN_GAP = 28
_ROW_PITCH = 124

# Offsets below a metric block's top edge, on the same CSS basis.
_METRIC_LABEL_BASELINE = 14
_METRIC_VALUE_BASELINE = 55
_METRIC_BAR_TOP = 68
_METRIC_BAR_H = 8
_METRIC_NOTE_BASELINE = 98

_PLATE_ALPHA = 0.8
# Extra height a CSS line box adds around a small label's cap height. The
# mockup pads the line box, not the letters, so the plates are this much taller.
_LINE_BOX_EXTRA = 8
# Same wash and border as the dashboard's safe stop, so both read alike.
_SAFE_STOP_ALPHA = 0.22
_SAFE_STOP_BORDER = 6


class PresentationRenderer:
    def __init__(
        self,
        colors: dict[str, tuple[int, int, int]] | None = None,
        sensor: CameraSensor | None = None,
        thresholds: DashboardThresholds | None = None,
    ) -> None:
        self.width = PRESENTATION_WIDTH
        self.height = PRESENTATION_HEIGHT
        if thresholds is None:
            thresholds = DashboardThresholds()
        self._thresholds = thresholds
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        if colors is not None:
            self._colors.update(colors)
        # Its own instance: the presentation is drawn on a different thread
        # from the operator dashboard, and the overlay keeps per-frame state.
        self._dashboard = AutonomyDashboard(colors=colors, sensor=sensor, thresholds=thresholds)
        # Loaded here so a missing font stops startup, not the first frame.
        self._text = TextRenderer()
        self._text.preload(_ALL_STYLES)
        # A 1920 x 1080 fill costs more than copying a cached one, so the
        # chrome that never changes is drawn once.
        self._static = self._build_static()

    def render(
        self,
        result: PipelineStepResult,
        telemetry: DashboardTelemetry,
        orbit_image: np.ndarray | None,
        plan: PathPlan | None,
        valid_roi: np.ndarray | None,
    ) -> np.ndarray:
        canvas = self._static.copy()
        active = telemetry.autopilot_active
        # The caller already withholds the plan under safe stop; this keeps
        # the video from ever showing a path the stack is not steering on.
        if not active:
            plan = None
        self._draw_status_chip(canvas, active)
        self._draw_orbit_view(canvas, orbit_image, active)
        self._draw_dashcam_card(canvas, result, plan, valid_roi)
        self._draw_stats(canvas, telemetry)
        return canvas

    # ----------------------------------------------------------- static layer

    def _build_static(self) -> np.ndarray:
        colors = self._colors
        canvas = np.full((self.height, self.width, 3), colors["BG"], dtype=np.uint8)

        fill_rect(canvas, HEADER, colors["PANEL_BG"])
        outline_rect(canvas, HEADER, colors["CARD_BORDER"])
        baseline = HEADER.y + (HEADER.h + self._text.cap_height(HEADLINE)) // 2
        title_x = HEADER.x + _HEADER_INSET
        title_w = self._text.draw(
            canvas, "OFF-ROAD AUTONOMY", title_x, baseline, HEADLINE, colors["TEXT_PRIMARY"]
        )
        self._text.draw(
            canvas,
            "Demonstration Run",
            title_x + title_w + 16,
            baseline,
            SUBTITLE,
            colors["TEXT_SECONDARY"],
        )

        # The placeholder for a run whose orbit camera has not delivered yet;
        # a live frame covers it.
        fill_rect(canvas, ORBIT_VIEW, colors["CARD_BG"])
        self._text.draw(
            canvas,
            "No Orbit Signal",
            ORBIT_VIEW.x + ORBIT_VIEW.w // 2,
            ORBIT_VIEW.y + (ORBIT_VIEW.h + self._text.cap_height(SUBTITLE)) // 2,
            SUBTITLE,
            colors["TEXT_SECONDARY"],
            align="center",
        )

        fill_rect(canvas, STATS_CARD, colors["CARD_BG"])
        outline_rect(canvas, STATS_CARD, colors["CARD_BORDER"])
        left = STATS_CARD.x + _STATS_PAD_X
        top = STATS_CARD.y + _STATS_PAD_TOP
        self._text.draw(
            canvas,
            "Steering",
            STATS_CARD.right - _STATS_PAD_X - _STEER_W,
            top + _STEER_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )
        labels = ("AUTONOMY FPS", "LATENCY MEAN / P95", "ROAD / VALID PX", "SEG. CONFIDENCE")
        for index, label in enumerate(labels):
            block_x, block_y = _metric_origin(left, top, index)
            self._text.draw(
                canvas,
                label,
                block_x,
                block_y + _METRIC_LABEL_BASELINE,
                LABEL_SPACED,
                colors["TEXT_SECONDARY"],
            )
        return canvas

    # ----------------------------------------------------------------- header

    def _draw_status_chip(self, canvas: np.ndarray, active: bool) -> None:
        text = "AUTONOMY ACTIVE"
        color = self._colors["GOOD"]
        solid = False
        if not active:
            text = "SAFE STOP | MANUAL CONTROL"
            color = self._colors["BAD"]
            solid = True
        draw_chip(
            canvas,
            self._text,
            HEADER.right - _HEADER_INSET,
            HEADER.y + (HEADER.h - chip_height(self._text)) // 2,
            text,
            color,
            self._colors["BG"],
            solid=solid,
            align="right",
        )

    # ------------------------------------------------------------------ views

    def _draw_orbit_view(
        self, canvas: np.ndarray, orbit_image: np.ndarray | None, active: bool
    ) -> None:
        colors = self._colors
        if orbit_image is not None:
            blit_cover(canvas, orbit_image, ORBIT_VIEW)
        self._draw_plate(canvas, ORBIT_VIEW.x + 16, ORBIT_VIEW.y + 16, "ORBIT CAMERA")
        outline_rect(canvas, ORBIT_VIEW, colors["CARD_BORDER"])
        if not active:
            blend_rect(canvas, ORBIT_VIEW, colors["BAD"], _SAFE_STOP_ALPHA)
            outline_rect(canvas, ORBIT_VIEW, colors["BAD"], _SAFE_STOP_BORDER)

    def _draw_dashcam_card(
        self,
        canvas: np.ndarray,
        result: PipelineStepResult,
        plan: PathPlan | None,
        valid_roi: np.ndarray | None,
    ) -> None:
        colors = self._colors
        card = DASHCAM_CARD
        overlay = self._dashboard.dashcam_overlay(result, plan, valid_roi)
        transform = blit_cover(canvas, overlay.image, card)
        if overlay.ego_label_px is not None:
            x, y = cover_point(transform, overlay.ego_label_px)
            self._text.draw(
                canvas, EGO_LABEL, x, y, LABEL_SPACED, colors["TEXT_PRIMARY"], align="center"
            )
        self._draw_plate(canvas, card.x + 12, card.y + 12, "WHAT THE VEHICLE SEES")
        self._draw_legend(canvas, plan)
        outline_rect(canvas, card, colors["CARD_BORDER"])

    def _draw_plate(self, canvas: np.ndarray, x: int, y: int, text: str) -> None:
        """Text on a dark translucent plate, so it reads over any scene."""
        cap = self._text.cap_height(PLATE)
        pad_y = 6 + _LINE_BOX_EXTRA // 2
        plate = Rect(x, y, self._text.width(text, PLATE) + 24, cap + 2 * pad_y)
        blend_rect(canvas, plate, self._colors["BG"], _PLATE_ALPHA)
        self._text.draw(canvas, text, x + 12, y + pad_y + cap, PLATE, self._colors["TEXT_PRIMARY"])

    def _draw_legend(self, canvas: np.ndarray, plan: PathPlan | None) -> None:
        # Entries follow the dashboard's rules: no path swatch when no path is
        # drawn, and a held path is named and coloured as one.
        entries = [("Traversable Mask", self._colors["MASK_FILL"])]
        if plan is not None:
            if plan.fallback_active:
                entries.append(("Held Path", self._colors["WARN"]))
            else:
                entries.append(("Planned Path", self._colors["PATH_CORE"]))

        swatch = 12
        swatch_gap = 6
        entry_gap = 14
        pad_x = 12
        pad_y = 8
        cap = self._text.cap_height(CAPTION)
        widths = [swatch + swatch_gap + self._text.width(label, CAPTION) for label, _ in entries]
        plate_w = sum(widths) + entry_gap * (len(entries) - 1) + 2 * pad_x
        plate_h = max(swatch, cap + _LINE_BOX_EXTRA) + 2 * pad_y
        plate = Rect(
            DASHCAM_CARD.right - 12 - plate_w,
            DASHCAM_CARD.bottom - 12 - plate_h,
            plate_w,
            plate_h,
        )
        blend_rect(canvas, plate, self._colors["BG"], _PLATE_ALPHA)
        x = plate.x + pad_x
        middle = plate.y + plate_h // 2
        for (label, color), width in zip(entries, widths, strict=True):
            fill_rect(canvas, Rect(x, middle - swatch // 2, swatch, swatch), color)
            self._text.draw(
                canvas,
                label,
                x + swatch + swatch_gap,
                middle + cap // 2,
                CAPTION,
                self._colors["TEXT_PRIMARY"],
            )
            x += width + entry_gap

    # ------------------------------------------------------------------ stats

    def _draw_stats(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        thresholds = self._thresholds
        left = STATS_CARD.x + _STATS_PAD_X
        right = STATS_CARD.right - _STATS_PAD_X
        top = STATS_CARD.y + _STATS_PAD_TOP

        speed_w = self._text.draw(
            canvas,
            f"{telemetry.speed_mph:.1f}",
            left,
            top + _SPEED_BASELINE,
            SPEED,
            colors["TEXT_PRIMARY"],
        )
        self._text.draw(
            canvas,
            "mph",
            left + speed_w + 10,
            top + _SPEED_BASELINE,
            UNIT,
            colors["TEXT_SECONDARY"],
        )

        direction = "Right"
        if telemetry.steering < 0.0:
            direction = "Left"
        word_w = self._text.draw(
            canvas,
            direction,
            right,
            top + _STEER_BASELINE,
            CAPTION,
            colors["TEXT_SECONDARY"],
            align="right",
        )
        self._text.draw(
            canvas,
            f"{telemetry.steering:+.2f}",
            right - word_w - 6,
            top + _STEER_BASELINE,
            VALUE,
            colors["TEXT_PRIMARY"],
            align="right",
        )
        draw_centered_bar(
            canvas,
            Rect(right - _STEER_W, top + _STEER_BAR_TOP, _STEER_W, _STEER_BAR_H),
            telemetry.steering,
            colors["TEXT_PRIMARY"],
            colors["MUTED_LINE"],
            colors["TEXT_SECONDARY"],
        )

        levels = _metric_levels(telemetry, thresholds)
        fps_note = f"Below {thresholds.target_fps:g} Fps Target"
        if levels["fps"] == bands.GOOD:
            fps_note = "On Target"
        elif levels["fps"] == bands.WARN:
            fps_note = f"Slightly Below {thresholds.target_fps:g} Fps Target"

        latency_note = f"Over {thresholds.latency_budget_ms:.0f} Ms Budget"
        if levels["latency"] == bands.GOOD:
            latency_note = f"Within {thresholds.latency_budget_ms:.0f} Ms Budget"

        blocks = (
            (
                f"{telemetry.fps:.1f}",
                levels["fps"],
                telemetry.fps / (thresholds.target_fps * thresholds.fps_bar_scale),
                1.0 / thresholds.fps_bar_scale,
                fps_note,
            ),
            (
                f"{telemetry.latency_ms:.0f} / {telemetry.latency_p95_ms:.0f}",
                levels["latency"],
                telemetry.latency_p95_ms
                / (thresholds.latency_budget_ms * thresholds.latency_bar_scale),
                1.0 / thresholds.latency_bar_scale,
                latency_note,
            ),
            (
                f"{telemetry.road_fraction:.3f}",
                levels["road"],
                telemetry.road_fraction / thresholds.road_bar_full_scale,
                thresholds.road_floor / thresholds.road_bar_full_scale,
                "",
            ),
            (
                f"{telemetry.perception_confidence:.2f}",
                levels["confidence"],
                telemetry.perception_confidence,
                thresholds.confidence_floor,
                "",
            ),
        )
        for index, (value, level, fraction, tick, note) in enumerate(blocks):
            block_x, block_y = _metric_origin(left, top, index)
            self._draw_metric(canvas, block_x, block_y, value, level, fraction, tick, note)

    def _draw_metric(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        value: str,
        level: str,
        fraction: float,
        tick: float,
        note: str,
    ) -> None:
        color = self._colors[level]
        self._text.draw(canvas, value, x, y + _METRIC_VALUE_BASELINE, METRIC, color)
        draw_bar(
            canvas,
            Rect(x, y + _METRIC_BAR_TOP, _COLUMN_W, _METRIC_BAR_H),
            fraction,
            color,
            self._colors["MUTED_LINE"],
            tick=tick,
            tick_color=self._colors["TEXT_PRIMARY"],
        )
        self._text.draw(canvas, note, x, y + _METRIC_NOTE_BASELINE, LABEL, color)


def _metric_levels(
    telemetry: DashboardTelemetry, thresholds: DashboardThresholds
) -> dict[str, str]:
    """The health level of each stat, by the same bands as the dashboard."""
    return {
        "fps": bands.fps_level(telemetry.fps, thresholds.target_fps, thresholds.fps_warn_fraction),
        "latency": bands.latency_level(telemetry.latency_p95_ms, thresholds.latency_budget_ms),
        "road": bands.road_level(telemetry.road_fraction, thresholds.road_floor),
        "confidence": bands.confidence_level(
            telemetry.perception_confidence,
            thresholds.confidence_floor,
            thresholds.confidence_good,
        ),
    }


def _metric_origin(left: int, top: int, index: int) -> tuple[int, int]:
    """Top-left of metric block ``index``, in reading order on a 2 x 2 grid."""
    column = index % 2
    row = index // 2
    return left + column * (_COLUMN_W + _COLUMN_GAP), top + _GRID_TOP + row * _ROW_PITCH
