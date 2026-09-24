"""Fuse the segmentation mask with stereo geometry.

The traversable mask is the primary signal; depth is a safety constraint on
top of it. Geometry may *veto* mask pixels it positively measured as
obstacles and may *weight* confidence, but it never invents road where the
segmenter saw none - stereo is sparse and noisy at range, and a false road
there would steer the vehicle off the trail.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from offroad_autonomy.perception.ego_mask import road_fraction
from offroad_autonomy.types import (
    PerceptionResult,
    PipelineConfig,
    TerrainAnalysis,
    TraversabilityMap,
)

logger = logging.getLogger("offroad_autonomy.perception.fusion")


def fuse_rgb_depth(
    perception: PerceptionResult,
    terrain: TerrainAnalysis,
    config: PipelineConfig,
) -> PerceptionResult:
    """Where depth is missing the mask is kept at ``depth_unknown_support``
    rather than discarded, so the stack degrades to appearance-only behaviour
    instead of blanking out."""
    rgb_mask = perception.mask.astype(bool)
    valid_roi = perception.valid_roi
    if terrain.height_above_ground.shape != rgb_mask.shape:
        logger.debug(
            "Skipping fusion: terrain %s does not match mask %s",
            terrain.height_above_ground.shape,
            rgb_mask.shape,
        )
        return perception

    unknown_support = float(np.clip(config.depth_unknown_support, 0.0, 1.0))
    weight = float(np.clip(config.depth_fusion_weight, 0.0, 1.0))

    support = np.where(terrain.valid, terrain.traversability, unknown_support).astype(np.float32)

    # Only triangulated obstacles may carve the mask; inferred ground proves
    # nothing about what stands on it.
    measured = terrain.measured
    if measured is None:
        measured = terrain.valid
    fused_mask = rgb_mask & ~(terrain.obstacle_mask & measured)

    # The appearance mask is already ROI-gated, but re-apply it here so depth
    # can never reintroduce a pixel the ego mask removed.
    if valid_roi is not None and valid_roi.shape == fused_mask.shape:
        fused_mask &= valid_roi
        support = np.where(valid_roi, support, 0.0).astype(np.float32)

    traversability = rgb_mask.astype(np.float32) * ((1.0 - weight) + weight * support)
    traversability[~fused_mask] = 0.0

    return PerceptionResult(
        mask=fused_mask,
        confidences=perception.confidences,
        num_detections=perception.num_detections,
        inference_time_ms=perception.inference_time_ms,
        traversability=traversability,
        rgb_mask=rgb_mask,
        valid_roi=valid_roi,
        road_fraction=road_fraction(fused_mask, valid_roi),
    )


def build_traversability_map(
    mask: np.ndarray,
    terrain: TerrainAnalysis | None,
    focal_px: float,
    traversability: np.ndarray | None = None,
    forward_clearance_m: float = math.inf,
    depth_age_s: float = 0.0,
) -> TraversabilityMap:
    """``boundary_distance_m`` uses ``lateral = pixels * depth / f``, so it is
    only defined where depth is."""
    mask = mask.astype(bool)
    shape = mask.shape
    valid_depth = np.zeros(shape, dtype=bool)
    depth = np.zeros(shape, dtype=np.float32)
    obstacle = np.zeros(shape, dtype=bool)
    if terrain is not None and terrain.valid.shape == shape:
        valid_depth = terrain.valid.copy()
        if terrain.range_m is not None:
            depth = terrain.range_m
        obstacle = terrain.obstacle_mask

    confidence = mask.astype(np.float32)
    if traversability is not None and traversability.shape == shape:
        confidence = traversability

    boundary = np.zeros(shape, dtype=np.float32)
    if mask.any() and valid_depth.any():
        edge_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        known = mask & valid_depth
        boundary[known] = edge_px[known] * depth[known] / max(float(focal_px), 1e-6)

    return TraversabilityMap(
        binary_mask=mask,
        depth_map=depth,
        confidence_map=confidence,
        valid_depth_mask=valid_depth,
        obstacle_mask=obstacle,
        boundary_distance_m=boundary,
        forward_clearance_m=forward_clearance_m,
        depth_age_s=depth_age_s,
    )
