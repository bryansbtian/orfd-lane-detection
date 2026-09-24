#!/usr/bin/env python3
"""Derive each camera's ego-vehicle silhouette from the vehicle's own geometry.

The polygons in ``configs/default.yaml`` are not hand-drawn: they come from
ray-casting the Hopper's front bumper and hood deck through each configured
camera and simplifying the silhouette. That makes them reproducible, and it
makes them *wrong* the moment a camera moves - so re-run this after changing
``beamng.cameras.stereo_rig`` or ``beamng.cameras.visualization`` and paste
the result back.

    python scripts/derive_ego_mask.py [--config configs/default.yaml]         [--camera left|right|display|pair|all]

The two kinds of camera use the answer differently:

* **left / right** are the perception pair. A silhouette here is a genuine
  exclusion - those pixels are not terrain and nothing may be concluded from
  them. From the front-bumper mount there is no silhouette at all, which is
  the point of that mount, and the script says so instead of failing.
* **display** is the visualization camera, which computes nothing. Its
  silhouette is only a drawing clip, so the reprojected road mask is not
  painted over the bonnet. It goes under ``visualization.overlay_clip``.

The fit is deliberately tight: a 1 px margin, a small closing kernel and a
fine simplification tolerance, then a check that the polygon still covers
the whole silhouette. The earlier 2 px margin, 31 px closing and 4 px
tolerance rounded the hood outline outward and threw away road beside it.

The body model is the h1..h4 node rows of ``hopper_hood.jbeam`` (the ant*
nodes there are a whip antenna, not bodywork) plus the cowl running back to
the windshield base, and the front bumper's fb1/fb3 rows from
``hopper_bumper_F.jbeam`` - which is what a low forward mount has to clear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.utils.config import load_config

# (y, z_centre, z_outer, half_width) per node row, front -> rear: the front
# bumper's top face, then the hood deck, then the cowl up to the windshield
# base. Metres, BeamNG vehicle space. The bumper rows come from
# hopper_bumper_F.jbeam (fb1* at y = -1.78/-1.74, top face z = 0.615..0.635,
# half-width 0.80) and matter for any mount low enough to look along it.
HOPPER_DECK = [
    (-1.780, 0.635, 0.615, 0.800),
    (-1.620, 0.635, 0.615, 0.800),
    (-1.550, 0.900, 0.880, 0.700),
    (-1.510, 1.095, 1.050, 0.590),
    (-1.180, 1.115, 1.070, 0.650),
    (-0.820, 1.125, 1.080, 0.690),
    (-0.440, 1.130, 1.080, 0.710),
    (-0.165, 1.410, 1.410, 0.710),
]


def deck_height(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Body surface height at (x, y), or -inf beyond the body half-width."""
    ys = np.array([r[0] for r in HOPPER_DECK])
    z_centre = np.interp(y, ys, [r[1] for r in HOPPER_DECK])
    z_outer = np.interp(y, ys, [r[2] for r in HOPPER_DECK])
    half = np.interp(y, ys, [r[3] for r in HOPPER_DECK])
    blend = np.clip(np.abs(x) / np.maximum(half, 1e-6), 0.0, 1.0)
    return np.where(np.abs(x) <= half, z_centre + (z_outer - z_centre) * blend, -np.inf)


def silhouette(camera: CameraModel, steps: int = 400) -> np.ndarray:
    """Mask of rays blocked by the ego body before they reach the ground."""
    uu, vv = camera.pixel_grid()
    dirs = np.stack(
        [
            (uu - camera.cx) / camera.focal_px,
            (vv - camera.cy) / camera.focal_px,
            np.ones_like(uu),
        ],
        axis=-1,
    ) @ camera.rotation.astype(np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    origin = camera.position.astype(np.float32)

    front = min(r[0] for r in HOPPER_DECK)
    blocked = np.zeros(uu.shape, dtype=bool)
    for slice_y in np.linspace(origin[1] - 0.02, front, steps):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (slice_y - origin[1]) / dirs[..., 1]
        ahead = np.isfinite(t) & (t > 0)
        point = origin + dirs * np.where(ahead, t, 0.0)[..., None]
        deck = deck_height(np.full_like(point[..., 0], slice_y), point[..., 0])
        blocked |= ahead & (point[..., 2] <= deck)
    return blocked


def to_polygon(mask: np.ndarray, margin_px: int, epsilon: float, close_px: int = 5) -> np.ndarray:
    """Simplify a silhouette into a normalised polygon, with a safety margin."""
    height, width = mask.shape
    grown = mask.astype(np.uint8)
    if margin_px > 0:
        grown = cv2.dilate(
            grown,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2 + 1,) * 2),
        )
    if close_px > 1:
        # Only to seal ray-marching speckle, not to round the outline.
        grown = cv2.morphologyEx(
            grown,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)),
        )

    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    points = cv2.approxPolyDP(max(contours, key=cv2.contourArea), epsilon, True).reshape(-1, 2)
    # Snap to the frame border so the filled polygon seals against the edge.
    points[:, 0] = np.where(points[:, 0] <= 3, 0, points[:, 0])
    points[:, 0] = np.where(points[:, 0] >= width - 4, width - 1, points[:, 0])
    points[:, 1] = np.where(points[:, 1] >= height - 4, height - 1, points[:, 1])

    normalised = np.stack([points[:, 0] / (width - 1), points[:, 1] / (height - 1)], axis=1)
    start = int(np.lexsort((normalised[:, 0], -normalised[:, 1]))[0])
    return np.roll(normalised, -start, axis=0)


def fill(polygon: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    filled = np.zeros(shape, np.uint8)
    pixels = np.stack([polygon[:, 0] * (width - 1), polygon[:, 1] * (height - 1)], axis=1)
    cv2.fillPoly(filled, [np.round(pixels).astype(np.int32)], 1)
    return filled.astype(bool)


def derive(spec, width: int, height: int, args, key: str) -> None:
    camera = CameraModel(spec, width, height)
    mask = silhouette(camera)

    print(
        f"# {spec.name}: pos={tuple(round(v, 3) for v in spec.pos)} "
        f"dir={tuple(round(v, 3) for v in spec.dir)}"
    )
    print(f"#   working size   : {camera.width}x{camera.height}")
    print(f"#   body silhouette: {mask.mean():.1%} of frame")

    polygon = to_polygon(mask, args.margin_px, args.epsilon, args.close_px)
    if len(polygon) < 3:
        # The mount clears the vehicle entirely. For the perception pair this
        # is the desired outcome, not a failure: there is nothing to exclude,
        # so every pixel is valid and the near-field road is kept whole.
        print("#   -> no bodywork in frame; nothing to exclude")
        print(f"      {key}:")
        print("        enabled: false")
        print()
        return

    filled = fill(polygon, mask.shape)
    missed = mask & ~filled
    print(
        f"#   polygon covers : {filled.mean():.1%} ({len(polygon)} vertices, "
        f"{args.margin_px} px margin)"
    )
    print(f"#   body px missed : {int(missed.sum())} (before the runtime margin)")
    print(f"      {key}:")
    print("        enabled: true")
    print(f"        margin_px: {args.margin_px}")
    print("        polygon:")
    for x, y in polygon:
        print(f"          - [{x:.4f}, {y:.4f}]")
    print()


#: Which config key each camera's polygon belongs under. The display camera's
#: is an overlay clip, not an exclusion, and the name keeps that straight.
_KEYS = {"left": "ego_mask", "right": "ego_mask", "display": "overlay_clip"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--camera",
        choices=("left", "right", "display", "pair", "all", "both"),
        default="all",
    )
    parser.add_argument("--margin-px", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--close-px", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    groups = {
        "all": ("left", "right", "display"),
        "pair": ("left", "right"),
        "both": ("left", "right"),
    }
    sides = groups.get(args.camera, (args.camera,))
    specs = {
        "left": config.left_camera,
        "right": config.right_camera,
        "display": config.display_camera,
    }
    for side in sides:
        spec = specs[side]
        # Each camera is evaluated on its own grid: the pair on the perception
        # working size, the display camera on its own, since a polygon is
        # normalised to the frame it was derived in.
        if side == "display":
            width, height = spec.width, spec.height
        else:
            width, height = config.preprocess_width, config.preprocess_height
        print(f"# --- {side} ---")
        derive(spec, width, height, args, _KEYS[side])


if __name__ == "__main__":
    main()
