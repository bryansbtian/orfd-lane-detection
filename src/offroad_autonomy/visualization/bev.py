"""Metric top-down view of the evidence actually used by the grid planner.

Up is the direction the vehicle faced when the run started, and it stays
that way: the vehicle sits in the middle and its icon turns, rather than the
whole map spinning every time it steers. The grid itself is in the vehicle's
frame, so it is rotated into this fixed frame for drawing only.
"""

import math
from itertools import pairwise

import cv2
import numpy as np

from offroad_autonomy.types import BEVSnapshot

ROAD = (82, 153, 63)
BLOCKED = (68, 66, 154)
UNKNOWN = (42, 39, 35)
OUTSIDE = (30, 26, 21)
PATH = (30, 225, 255)
GRID_LINE = (70, 70, 70)
BACKGROUND = (26, 22, 17)
#: Vehicle outline in camera-ground metres (forward, right): bumper ~1.8 m
#: ahead of the camera, tail ~2.3 m behind, 1.9 m wide, pointed at the front.
VEHICLE_OUTLINE = np.array([[1.8, 0.0], [1.2, 0.95], [-2.3, 0.95], [-2.3, -0.95], [1.2, -0.95]])


class _FixedFrame:
    """Vehicle-frame metres (forward, right) to screen pixels, with screen up
    along the starting heading and the vehicle at the centre."""

    def __init__(self, heading_rad: float, centre: tuple[float, float], px_per_m: float) -> None:
        self.cos = math.cos(heading_rad)
        self.sin = math.sin(heading_rad)
        self.cx, self.cy = centre
        self.ppm = px_per_m

    def points(self, forward_right: np.ndarray) -> np.ndarray:
        f = forward_right[:, 0]
        r = forward_right[:, 1]
        # Heading is positive turned left, so the vehicle's forward appears
        # rotated anticlockwise on screen.
        x = self.cx + (-f * self.sin + r * self.cos) * self.ppm
        y = self.cy - (f * self.cos + r * self.sin) * self.ppm
        return np.column_stack((x, y))

    def cell_affine(self, snapshot: BEVSnapshot) -> np.ndarray:
        """Affine map from grid cell (col, row) centres to screen pixels."""
        c = snapshot.cell_m
        f0 = snapshot.ahead_m - 0.5 * c  # forward of row 0's centre
        r0 = -snapshot.half_width_m + 0.5 * c  # right of col 0's centre
        k = self.ppm * c
        return np.array(
            [
                [k * self.cos, k * self.sin, self.cx + self.ppm * (-f0 * self.sin + r0 * self.cos)],
                [-k * self.sin, k * self.cos, self.cy - self.ppm * (f0 * self.cos + r0 * self.sin)],
            ]
        )


def render_bev(snapshot: BEVSnapshot | None, width: int, height: int, show_path=True):
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    def label(text, x, y, color=(224, 224, 224), scale=0.45):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    label("BEV / LOCAL METRES", 12, 24, scale=0.6)
    if snapshot is None:
        label("Select the grid planner to see its BEV", 12, 60)
        return canvas
    s = snapshot
    heading = 0.0
    if s.heading_rad is not None:
        heading = float(s.heading_rad)

    # A square around the vehicle, large enough for the full look-ahead in any
    # direction it might face.
    radius_m = s.ahead_m
    side = max(1, min(width - 24, height - 135))
    left, top = (width - side) // 2, 52
    frame = _FixedFrame(heading, (side / 2.0, side / 2.0), side / (2.0 * radius_m))

    rows, cols = s.road.shape
    cells = np.full((rows, cols, 3), UNKNOWN, dtype=np.uint8)
    cells[s.road] = ROAD
    cells[s.blocked] = BLOCKED
    if s.confidence is not None:
        # A cell seen once from far away is a guess; drawn as solidly as one
        # confirmed many times, every revision looks like the map changing.
        strength = 0.3 + 0.7 * s.confidence[..., None]
        unknown = np.array(UNKNOWN, dtype=np.float32)
        cells = (unknown + (cells.astype(np.float32) - unknown) * strength).astype(np.uint8)
    # Evidence outside the current camera view is retained memory.
    cells[~s.visible] = (cells[~s.visible].astype(float) * 0.65).astype(np.uint8)
    view = cv2.warpAffine(
        cells,
        frame.cell_affine(s),
        (side, side),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=OUTSIDE,
    )

    def line(path, color, thickness=1):
        pts = frame.points(np.asarray(path, dtype=float)).round().astype(np.int32)
        # Clip to the map so curves cannot overwrite the legend.
        for a, b in pairwise(pts):
            ok, a, b = cv2.clipLine((0, 0, side, side), tuple(a), tuple(b))
            if ok:
                cv2.line(view, a, b, color, thickness, cv2.LINE_AA)

    # Distance rings from the vehicle, fixed to the screen.
    centre_px = (side // 2, side // 2)
    for metres in range(5, int(radius_m) + 1, 5):
        cv2.circle(view, centre_px, int(round(metres * frame.ppm)), GRID_LINE, 1, cv2.LINE_AA)
    cv2.line(view, (side // 2, 0), (side // 2, side), GRID_LINE, 1)
    cv2.line(view, (0, side // 2), (side, side // 2), GRID_LINE, 1)

    contours, _ = cv2.findContours(
        s.visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    affine = frame.cell_affine(s)
    for contour in contours:
        xy = contour[:, 0, :].astype(float)
        xy = xy @ affine[:, :2].T + affine[:, 2]
        cv2.polylines(view, [xy.round().astype(np.int32)], True, (175, 175, 175), 1)
    if show_path:
        for candidate in s.candidates:
            line(candidate, (95, 102, 99))
        if s.path is not None:
            line(s.path, PATH, 3)
    # The vehicle, roughly to scale around the camera: it is now the only
    # thing on screen that shows which way the vehicle points.
    ego = frame.points(VEHICLE_OUTLINE)
    cv2.fillPoly(view, [ego.round().astype(np.int32)], (240, 240, 240), cv2.LINE_AA)

    canvas[top : top + side, left : left + side] = view
    for metres in range(5, int(radius_m) + 1, 5):
        y = top + side // 2 - int(round(metres * frame.ppm))
        label(f"{metres}m", left + side // 2 + 4, y + 14, color=(150, 150, 150), scale=0.38)
    lock = "unlocked"
    if s.branch_locked:
        lock = "LOCKED"
    label(f"Branches {s.branch_count} / {lock}", 12, 42, scale=0.4)
    turned = math.degrees(heading)
    label(f"Heading {turned:+.0f} deg", width - 150, 42, scale=0.4)
    label("Up = direction at the start; the car icon turns", 12, height - 65, scale=0.4)
    label("Green road | Red non-road | Dark unknown", 12, height - 48, scale=0.4)
    label("Yellow selected path | Gray candidate arcs", 12, height - 31, scale=0.4)
    label("White outline: visible now; dim: memory; faint: unsure", 12, height - 14, scale=0.4)
    return canvas
