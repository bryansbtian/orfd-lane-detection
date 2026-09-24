"""The image segmentation runs on, and the geometry that goes with it.

``segmentation_mode`` picks one of:

* ``left`` / ``right`` - one camera of the stereo pair, as captured.
* ``stitched`` - the precomputed wide view built from both.

Whatever the choice, this module answers the same three questions for the
rest of the stack: which pixels does segmentation see, which of them are the
ego vehicle, and what camera model maps them to rays. Depth is reprojected
into this view and terrain rays are cast from it, so the mask, the depth map
and the terrain maps always share one grid.
"""

from __future__ import annotations

import logging

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.ego_mask import EgoMask
from offroad_autonomy.perception.stitching import WideViewStitcher
from offroad_autonomy.types import SEGMENTATION_MODES, PipelineConfig, StereoFramePair

logger = logging.getLogger("offroad_autonomy.perception.view")


class PerceptionView:
    def __init__(self, config: PipelineConfig) -> None:
        mode = str(config.segmentation_mode).lower()
        if mode not in SEGMENTATION_MODES:
            raise ValueError(f"segmentation.mode must be one of {SEGMENTATION_MODES}, got {mode!r}")
        self.mode = mode

        # Built whenever it is used for segmentation or asked for by the
        # dashboard; otherwise it would only cost start-up time.
        self.stitcher: WideViewStitcher | None = None
        if mode == "stitched" or config.stitch_enabled:
            self.stitcher = WideViewStitcher.from_config(config)

        if mode == "stitched":
            assert self.stitcher is not None
            self.camera = self.stitcher.camera
            capture = (config.left_camera.height, config.left_camera.width)
            excluded = self.stitcher.warp_masks(
                EgoMask(config.left_camera.ego_mask).excluded(capture),
                EgoMask(config.right_camera.ego_mask).excluded(capture),
            )
            self.valid_roi = ~excluded
        else:
            spec = config.segmentation_camera
            self.camera = CameraModel(spec, config.preprocess_width, config.preprocess_height)
            self.valid_roi = EgoMask(spec.ego_mask).valid_roi(
                (self.camera.height, self.camera.width)
            )

        logger.info(
            "Segmentation view: %s %dx%d, ego exclusion %.1f%% of frame",
            mode,
            self.camera.width,
            self.camera.height,
            100.0 * self.ego_coverage,
        )

    @property
    def size(self) -> tuple[int, int]:
        return self.camera.width, self.camera.height

    @property
    def ego_coverage(self) -> float:
        return float(1.0 - self.valid_roi.mean())

    def image(self, pair: StereoFramePair) -> np.ndarray | None:
        """Single-camera modes keep running when the *other* camera drops a
        frame; only the stitched view needs both."""
        if self.mode == "stitched":
            if pair.left is None or pair.right is None or self.stitcher is None:
                return None
            return self.stitcher.stitch(pair.left, pair.right)
        return pair.frame(self.mode)

    def stitched(self, pair: StereoFramePair) -> np.ndarray | None:
        if self.stitcher is None or pair.left is None or pair.right is None:
            return None
        return self.stitcher.stitch(pair.left, pair.right)
