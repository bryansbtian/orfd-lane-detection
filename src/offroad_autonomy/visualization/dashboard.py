"""Professional development dashboard for autonomy visualization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from offroad_autonomy.types import DEFAULT_DASHBOARD_COLORS, PathPlan


@dataclass
class DashboardTelemetry:
    """Compact status values shown in the dashboard sidebar."""

    speed_mps: float
    steering: float
    throttle: float
    brake: float
    perception_confidence: float
    stability_score: float
    kalman_active: bool
    fps: float
    latency_ms: float


class AutonomyDashboard:
    """Render a clean dashboard view for the autonomy stack."""

    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        colors: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self._pad = 24
        self._gap = 20
        self._header_h = 72
        self._sidebar_w = 360
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        if colors is not None:
            self._colors.update(colors)

    def render(
        self,
        camera_bgr: np.ndarray,
        postprocessed_mask: np.ndarray,
        plan: PathPlan | None,
        telemetry: DashboardTelemetry,
    ) -> np.ndarray:
        """Build the dashboard frame."""
        camera = self._ensure_bgr(camera_bgr)
        mask_source_shape = postprocessed_mask.shape[:2]
        mask = self._ensure_mask(postprocessed_mask, camera.shape[:2])
        overlay = self._build_overlay(camera, mask, plan, mask_source_shape)

        canvas = np.full((self.height, self.width, 3), self._colors["BG"], dtype=np.uint8)
        self._draw_header(canvas)

        content_top = self._pad + self._header_h
        content_h = self.height - content_top - self._pad
        main_w = self.width - (self._pad * 2) - self._gap - self._sidebar_w

        main_rect = (self._pad, content_top, main_w, content_h)
        side_rect = (self._pad + main_w + self._gap, content_top, self._sidebar_w, content_h)

        self._draw_panel(canvas, main_rect)
        self._draw_panel(canvas, side_rect)

        inner_pad = 16
        viewport_rect = (
            main_rect[0] + inner_pad,
            main_rect[1] + inner_pad,
            main_rect[2] - inner_pad * 2,
            main_rect[3] - inner_pad * 2,
        )
        viewport = self._fit_image(
            overlay,
            viewport_rect[2],
            viewport_rect[3],
            fill=self._colors["PANEL_BG"],
        )
        self._blit(canvas, viewport, viewport_rect[0], viewport_rect[1])
        self._draw_viewport_labels(canvas, viewport_rect)

        self._draw_sidebar(canvas, side_rect, mask, telemetry)
        self._draw_pipeline_footer(canvas, main_rect)
        return canvas

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    @staticmethod
    def _ensure_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
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
    ) -> np.ndarray:
        display = camera.copy()

        if mask.any():
            mask_layer = display.copy()
            mask_layer[mask] = self._colors["MASK_FILL"]
            display = cv2.addWeighted(mask_layer, 0.28, display, 0.72, 0)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                display,
                contours,
                -1,
                self._colors["MASK_EDGE"],
                2,
                lineType=cv2.LINE_AA,
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
            cv2.polylines(
                glow,
                [pts_i],
                False,
                self._colors["PATH_GLOW"],
                18,
                lineType=cv2.LINE_AA,
            )
            display = cv2.addWeighted(glow, 0.22, display, 0.78, 0)

            cv2.polylines(
                display,
                [pts_i],
                False,
                self._colors["PATH_CORE"],
                6,
                lineType=cv2.LINE_AA,
            )
            cv2.polylines(
                display,
                [pts_i],
                False,
                self._colors["PATH_HIGHLIGHT"],
                2,
                lineType=cv2.LINE_AA,
            )

            target_idx = max(0, len(pts) // 3)
            target = tuple(np.round(pts[target_idx]).astype(int))
            cv2.circle(
                display,
                target,
                9,
                self._colors["PATH_CORE"],
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                display,
                target,
                9,
                self._colors["PATH_HIGHLIGHT"],
                2,
                lineType=cv2.LINE_AA,
            )

        return display

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
    def _fit_image(image: np.ndarray, target_w: int, target_h: int, fill: tuple[int, int, int]) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        out = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
        off_x = (target_w - new_w) // 2
        off_y = (target_h - new_h) // 2
        out[off_y:off_y + new_h, off_x:off_x + new_w] = resized
        return out

    @staticmethod
    def _blit(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
        h, w = image.shape[:2]
        canvas[y:y + h, x:x + w] = image

    def _draw_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["PANEL_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)

    def _draw_header(self, canvas: np.ndarray) -> None:
        x = self._pad
        y = self._pad
        w = self.width - self._pad * 2
        h = self._header_h

        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["PANEL_BG"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), self._colors["CARD_BORDER"], 1)

        cv2.putText(
            canvas,
            "OFF-ROAD AUTONOMY DEVELOPMENT DASHBOARD",
            (x + 20, y + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "BeamNG.tech camera input  |  postprocessed traversability  |  smoothed controller reference path",
            (x + 20, y + 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        chip_text = "AUTONOMY ACTIVE"
        chip_size, _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        self._draw_chip(
            canvas,
            x + w - chip_size[0] - 44,
            y + 18,
            chip_text,
            self._colors["GOOD"],
        )

    def _draw_viewport_labels(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        self._draw_chip(canvas, x + 18, y + 18, "FRONT CAMERA POV", self._colors["WARN"])
        self._draw_chip(
            canvas,
            x + 18,
            y + h - 64,
            "POSTPROCESSED TRAVERSABLE REGION",
            self._colors["MASK_FILL"],
        )
        self._draw_chip(
            canvas,
            x + 18,
            y + h - 30,
            "SMOOTHED CONTROLLER REFERENCE PATH",
            self._colors["PATH_CORE"],
        )

    def _draw_sidebar(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        mask: np.ndarray,
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        inner_x = x + 16
        inner_y = y + 16
        inner_w = w - 32
        gap = 14

        available_h = h - 32 - gap * 4
        vehicle_h = int(round(available_h * 0.24))
        perception_h = int(round(available_h * 0.22))
        performance_h = int(round(available_h * 0.16))
        provenance_h = int(round(available_h * 0.17))
        preview_h = available_h - vehicle_h - perception_h - performance_h - provenance_h

        self._draw_vehicle_card(canvas, (inner_x, inner_y, inner_w, vehicle_h), telemetry)
        inner_y += vehicle_h + gap
        self._draw_perception_card(canvas, (inner_x, inner_y, inner_w, perception_h), telemetry)
        inner_y += perception_h + gap
        self._draw_performance_card(canvas, (inner_x, inner_y, inner_w, performance_h), telemetry)
        inner_y += performance_h + gap
        self._draw_provenance_card(canvas, (inner_x, inner_y, inner_w, provenance_h))
        inner_y += provenance_h + gap
        self._draw_mask_preview(canvas, (inner_x, inner_y, inner_w, preview_h), mask)

    def _draw_vehicle_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "VEHICLE")

        cv2.putText(
            canvas,
            f"{telemetry.speed_mps:4.1f} m/s",
            (x + 18, y + 66),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Speed",
            (x + 18, y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        self._draw_value_row(
            canvas,
            x + 18,
            y + min(h - 62, 98),
            w - 36,
            "Steering",
            f"{telemetry.steering:+.2f}",
        )
        self._draw_progress_row(
            canvas,
            x + 18,
            y + h - 40,
            w - 36,
            "Throttle",
            telemetry.throttle,
            self._colors["GOOD"],
        )
        self._draw_progress_row(
            canvas,
            x + 18,
            y + h - 18,
            w - 36,
            "Brake",
            telemetry.brake,
            self._colors["BAD"],
        )

    def _draw_perception_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "PERCEPTION AND PLANNING")

        conf_y = y + min(54, h - 58)
        stability_y = y + min(92, h - 34)
        fallback_y = y + h - 18

        self._draw_progress_row(
            canvas,
            x + 18,
            conf_y,
            w - 36,
            "Perception confidence",
            telemetry.perception_confidence,
            self._colors["WARN"],
        )
        self._draw_progress_row(
            canvas,
            x + 18,
            stability_y,
            w - 36,
            "Stability score",
            telemetry.stability_score,
            self._colors["GOOD"],
        )
        self._draw_value_row(
            canvas,
            x + 18,
            fallback_y,
            w - 36,
            "Kalman fallback",
            "ACTIVE" if telemetry.kalman_active else "INACTIVE",
            value_color=(
                self._colors["BAD"]
                if telemetry.kalman_active
                else self._colors["GOOD"]
            ),
        )

    def _draw_performance_card(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        telemetry: DashboardTelemetry,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "RUNTIME")
        mid = x + w // 2
        value_y = y + min(62, h - 28)
        label_y = y + h - 16

        cv2.putText(
            canvas,
            f"{telemetry.fps:4.1f}",
            (x + 18, value_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.95,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "FPS",
            (x + 18, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

        cv2.line(
            canvas,
            (mid, y + 18),
            (mid, y + h - 16),
            self._colors["MUTED_LINE"],
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"{telemetry.latency_ms:4.0f} ms",
            (mid + 18, value_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.82,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Loop latency",
            (mid + 18, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )

    def _draw_provenance_card(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "PIPELINE OUTPUT")
        row_1_y = y + min(44, h // 2 - 6)
        row_2_y = y + h - 28

        self._draw_legend_row(
            canvas,
            x + 18,
            row_1_y,
            self._colors["MASK_FILL"],
            "Mask",
            "Temporal smoothing + morphology cleanup",
            "Postprocessed traversable region",
        )
        self._draw_legend_row(
            canvas,
            x + 18,
            row_2_y,
            self._colors["PATH_CORE"],
            "Path",
            "Smoothed planning output",
            "Controller reference centerline",
        )

    def _draw_mask_preview(
        self,
        canvas: np.ndarray,
        rect: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> None:
        x, y, w, h = rect
        self._draw_card(canvas, rect, "POSTPROCESSED MASK")

        preview = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        preview[mask] = self._colors["MASK_FILL"]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            preview,
            contours,
            -1,
            self._colors["MASK_EDGE"],
            2,
            lineType=cv2.LINE_AA,
        )

        fitted = self._fit_image(
            preview,
            w - 24,
            h - 44,
            fill=self._colors["CARD_BG"],
        )
        self._blit(canvas, fitted, x + 12, y + 28)

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
        value_clamped = float(np.clip(value, 0.0, 1.0))
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
            f"{value_clamped:.2f}",
            (x + width - 38, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )

        bar_y = y + 8
        cv2.rectangle(
            canvas,
            (x, bar_y),
            (x + width, bar_y + 8),
            self._colors["MUTED_LINE"],
            -1,
        )
        fill_w = int(round(width * value_clamped))
        if fill_w > 0:
            cv2.rectangle(canvas, (x, bar_y), (x + fill_w, bar_y + 8), color, -1)

    def _draw_legend_row(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        color: tuple[int, int, int],
        title: str,
        detail_1: str,
        detail_2: str,
    ) -> None:
        cv2.rectangle(canvas, (x, y - 10), (x + 12, y + 2), color, -1)
        cv2.putText(
            canvas,
            title,
            (x + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            self._colors["TEXT_PRIMARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            detail_1,
            (x + 84, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self._colors["TEXT_SECONDARY"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            detail_2,
            (x + 20, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self._colors["TEXT_SECONDARY"],
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

    def _draw_pipeline_footer(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x, y, w, h = rect
        footer_y = y + h - 34
        cv2.line(
            canvas,
            (x + 16, footer_y - 18),
            (x + w - 16, footer_y - 18),
            self._colors["MUTED_LINE"],
            1,
            cv2.LINE_AA,
        )

        stages = [
            "CAMERA",
            "PREPROCESS",
            "PERCEPTION",
            "POSTPROCESS",
            "PLANNING",
            "CONTROL",
        ]

        step_w = (w - 32) // len(stages)
        for idx, stage in enumerate(stages):
            sx = x + 16 + idx * step_w
            color = (
                self._colors["TEXT_PRIMARY"]
                if idx in {0, 2, 3, 4, 5}
                else self._colors["TEXT_SECONDARY"]
            )
            cv2.putText(
                canvas,
                stage,
                (sx + 4, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
                cv2.LINE_AA,
            )
