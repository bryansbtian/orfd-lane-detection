"""Presentation Frames From Actual R3 Cameras, Geometry, And Recorded Telemetry."""

from __future__ import annotations

import cv2
import numpy as np

from offroad_autonomy.visualization.layout import Rect
from offroad_autonomy.visualization.text import SANS, SANS_SEMIBOLD, TextRenderer, TextStyle
from offroad_autonomy.visualization.widgets import blit_cover

BG = (22, 18, 16)
PANEL = (40, 34, 29)
INK = (244, 242, 235)
MUTED = (191, 182, 166)
TEAL = (172, 205, 69)
ORANGE = (77, 161, 244)
TITLE = TextStyle(SANS_SEMIBOLD, 40)
HEADING = TextStyle(SANS_SEMIBOLD, 28)
BODY = TextStyle(SANS, 24)
SMALL = TextStyle(SANS, 19)
VALUE = TextStyle(SANS_SEMIBOLD, 38)


class RoutePresentation:
    def __init__(self, geometry, route, baseline, improved):
        self.geometry = geometry
        self.route = route
        self.baseline = baseline
        self.improved = improved
        self.text = TextRenderer()
        self.text.preload((TITLE, HEADING, BODY, SMALL, VALUE))
        selected = slice(route["start_node_index"], route["finish_node_index"] + 1)
        self.center = geometry.center[selected, :2]
        self.left = geometry.left[selected, :2]
        self.right = geometry.right[selected, :2]

    def words(self, canvas, text, x, y, style=BODY, color=INK):
        self.text.draw(canvas, text, x, y, style, color)

    def canvas(self, title, subtitle):
        image = np.full((1080, 1920, 3), BG, dtype=np.uint8)
        self.words(image, title, 32, 56, TITLE)
        self.words(image, subtitle, 32, 92, SMALL, MUTED)
        return image

    def map(self, canvas, rect, points=()):
        cv2.rectangle(canvas, (rect.x, rect.y), (rect.right, rect.bottom), PANEL, -1)
        low = np.minimum(self.left.min(axis=0), self.right.min(axis=0)) - 5
        high = np.maximum(self.left.max(axis=0), self.right.max(axis=0)) + 5
        scale = min((rect.w - 48) / (high[0] - low[0]), (rect.h - 60) / (high[1] - low[1]))
        origin = np.array(
            [
                rect.x + (rect.w - (high[0] - low[0]) * scale) / 2,
                rect.y + 40 + (rect.h - 48 - (high[1] - low[1]) * scale) / 2,
            ]
        )

        def project(xy):
            xy = np.atleast_2d(xy)
            return np.rint(
                origin + np.column_stack((xy[:, 0] - low[0], high[1] - xy[:, 1])) * scale
            ).astype(np.int32)

        polygon = project(np.r_[self.left, self.right[::-1]])
        cv2.fillPoly(canvas, [polygon], (107, 108, 105))
        cv2.polylines(canvas, [project(self.center)], False, TEAL, 3, cv2.LINE_AA)
        for point, label in ((self.center[0], "A"), (self.center[-1], "B")):
            xy = tuple(project(point)[0])
            cv2.circle(canvas, xy, 7, INK, -1, cv2.LINE_AA)
            self.words(canvas, label, xy[0] + 10, xy[1] - 8, SMALL)
        for point, color in points:
            cv2.circle(canvas, tuple(project(point)[0]), 7, color, -1, cv2.LINE_AA)
        if rect.w > 500:
            self.words(
                canvas, "Approved Route | BeamNG Authored Geometry", rect.x + 16, rect.y + 28, SMALL
            )

    def intro(self):
        frame = self.canvas(
            "R3 | Centerline Tracking Through Dirt-Road Bends",
            "Automation Test Track  |  Hopper  |  Sunny Daytime  |  Approved Open A-To-B Course",
        )
        self.map(frame, Rect(32, 124, 1150, 860))
        x = 1230
        self.words(frame, f"{self.route['route_length_m']:.3f} m", x, 210, VALUE, TEAL)
        self.words(frame, "Same Route. Same Vehicle.", x, 270, HEADING)
        self.words(frame, "Existing Controller And Safety Guards", x, 312, BODY)
        self.words(frame, "Baseline", x, 420, HEADING, ORANGE)
        self.words(frame, "Original Preview Starting Point", x, 462)
        self.words(frame, "Improved", x, 570, HEADING, TEAL)
        self.words(frame, "Preview Starts At Visible Road", x, 612)
        self.words(frame, "Fresh Demonstration Footage", x, 770, HEADING)
        self.words(frame, "Results Shown Later Come From", x, 820)
        self.words(frame, "Separate Verification Batches", x, 858)
        self.words(
            frame,
            "Bend Highlights Are Aligned By Route Progress; Playback Is Not Real-Time",
            32,
            1042,
            BODY,
            MUTED,
        )
        return frame

    def comparison(self, before, after, before_state, after_state, progress, chapter):
        frame = self.canvas(
            "R3 | Baseline Versus Visibility-Aware Preview",
            f"{chapter}  |  Distance-Synchronized Playback  |  {progress:.1f} / {self.route['route_length_m']:.1f} m",
        )
        for x, label, camera, state, color in (
            (32, "Baseline", before, before_state, ORANGE),
            (980, "Improved", after, after_state, TEAL),
        ):
            self.words(frame, label, x, 151, HEADING, color)
            blit_cover(frame, camera[:, :960], Rect(x, 174, 908, 688))
            self.words(frame, f"Snapshot Offset: {state['offset_m']:+.2f} m", x, 908)
            self.words(frame, f"Speed: {state['native_pose']['speed_mph']:.2f} mph", x + 455, 908)
        self.map(
            frame,
            Rect(32, 940, 430, 112),
            (
                (before_state["native_pose"]["position"][:2], ORANGE),
                (after_state["native_pose"]["position"][:2], TEAL),
            ),
        )
        self.words(frame, "Same Bend, Matched By Road Position", 500, 978, HEADING)
        self.words(
            frame,
            "Actual Recorded Cameras; Snapshot Telemetry Is Not Exposure-Synchronized",
            500,
            1018,
            SMALL,
            MUTED,
        )
        self.words(
            frame,
            "Illustrative Recordings | Original And Improved Batch Evidence Remains Separate",
            500,
            1050,
            SMALL,
            MUTED,
        )
        return frame

    def full(self, image, state):
        native = state["native_pose"]
        frame = self.canvas(
            "R3 | Approved Route, Full Autonomous Demonstration",
            "Visibility-Aware Preview  |  Actual Chase Camera, Dashcam, And Route Geometry",
        )
        blit_cover(frame, image[:, :960], Rect(24, 114, 1238, 938))
        self.map(frame, Rect(1286, 114, 610, 330), ((native["position"][:2], TEAL),))
        self.words(frame, "Front Camera | Same Inference Snapshot", 1286, 480, SMALL)
        blit_cover(frame, image[54:674, 960:], Rect(1286, 498, 610, 394))
        self.words(
            frame,
            f"Progress: {state['progress_m']:.1f} / {self.route['route_length_m']:.1f} m",
            1286,
            937,
            HEADING,
        )
        self.words(frame, f"Speed: {native['speed_mph']:.2f} mph", 1286, 979)
        self.words(frame, f"Signed Offset: {state['offset_m']:+.2f} m", 1286, 1017)
        self.words(
            frame, f"Simulation Time: {native['timestamp_s']:.1f} s", 1286, 1051, SMALL, MUTED
        )
        return frame

    def results(self):
        frame = self.canvas(
            "R3 | Measured Batch Comparison",
            "Statistics Below Are From The Separate Original And Improved Batches",
        )
        x_positions = (80, 690, 1300)
        specs = (
            ("Mean Absolute Offset", "pooled_mean_absolute_lateral_offset_m"),
            ("Signed-Offset SD", "pooled_std_signed_lateral_offset_m"),
            ("Maximum Absolute Offset", "maximum_absolute_lateral_offset_m"),
        )
        for x, (label, key) in zip(x_positions, specs):
            self.words(frame, label, x, 230, HEADING)
            self.words(frame, f"{self.baseline[key]:.3f} m", x, 320, VALUE, ORANGE)
            self.words(frame, "To", x, 380, BODY, MUTED)
            self.words(frame, f"{self.improved[key]:.3f} m", x, 460, VALUE, TEAL)
        self.words(
            frame,
            f"{self.improved['completed_runs']}/{self.improved['measured_attempts']} Routes Completed"
            f" | {self.improved['manual_interventions']} Recorded Interventions"
            f" | {self.improved['road_departures']} Authored-Corridor Departures",
            80,
            620,
            HEADING,
        )
        self.words(
            frame, f"Maximum Recorded Speed: {self.improved['maximum_speed_mph']:.3f} mph", 80, 690
        )
        targets_met = sum(
            (
                self.improved["pooled_mean_absolute_lateral_offset_m"]
                <= self.route["mean_absolute_offset_target_m"],
                self.improved["pooled_std_signed_lateral_offset_m"]
                <= self.route["std_signed_offset_target_m"],
            )
        )
        self.words(
            frame,
            f"Improved Batch | Proposed Centering Targets Met: {targets_met}/2",
            80,
            800,
            HEADING,
        )
        self.words(
            frame,
            f"Proposed Limits: Mean {self.route['mean_absolute_offset_target_m']:.2f} m"
            f" | Signed SD {self.route['std_signed_offset_target_m']:.2f} m",
            80,
            860,
        )
        self.words(
            frame,
            "Official Compliance Requires Approved Targets, Boundaries, And Vehicle Datum",
            80,
            910,
            SMALL,
            MUTED,
        )
        self.words(
            frame,
            "Recorded Physics Measurements And Full-Vehicle Envelope Checks | Review Batch Evidence",
            80,
            1010,
            SMALL,
            MUTED,
        )
        return frame
