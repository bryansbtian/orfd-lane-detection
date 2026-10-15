"""The steering controller ``control.mode`` selects."""

from __future__ import annotations

from collections.abc import Callable

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PipelineConfig


def build_controller(
    config: PipelineConfig,
    camera: CameraModel | None = None,
    grid_source: Callable[[], object] | None = None,
):
    """``stanley`` (the default), ``mppi`` or ``mpc``. Imported lazily so the
    default never loads the predictive modules."""
    if config.control_mode == "mppi":
        from offroad_autonomy.control.mppi_controller import MppiController

        return MppiController(config, camera=camera, grid_source=grid_source)
    if config.control_mode == "mpc":
        from offroad_autonomy.control.mpc_controller import MpcController

        return MpcController(config, camera=camera)
    return StanleyController(config, camera=camera)
