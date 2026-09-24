"""Perception sub-package.

``RoadSegmenter`` is deliberately not re-exported here: importing it pulls in
Ultralytics and Torch, and the geometry stages below are useful (and tested)
without that cost. Import it from its own module when you need it.
"""

from offroad_autonomy.perception.camera_geometry import (
    CameraModel,
    StereoRig,
    build_camera_models,
)
from offroad_autonomy.perception.ego_mask import EgoMask
from offroad_autonomy.perception.fusion import build_traversability_map, fuse_rgb_depth
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.stereo_depth import StereoDepthEstimator
from offroad_autonomy.perception.stereo_rectification import (
    StereoRectifier,
    draw_epipolar_pair,
)
from offroad_autonomy.perception.stitching import WideViewStitcher
from offroad_autonomy.perception.terrain_analyzer import TerrainAnalyzer

__all__ = [
    "CameraModel",
    "EgoMask",
    "PerceptionView",
    "StereoDepthEstimator",
    "StereoRectifier",
    "StereoRig",
    "TerrainAnalyzer",
    "WideViewStitcher",
    "build_camera_models",
    "build_traversability_map",
    "draw_epipolar_pair",
    "fuse_rgb_depth",
]
