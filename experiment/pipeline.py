"""The default pipeline with an experimental planner or controller swapped in."""

from __future__ import annotations

import time
from collections.abc import Callable

from experiment.config import ExperimentConfig
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.pipeline import AutonomyPipeline
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.types import ControlCommand, PlanningStepResult, VehicleState


def build_planner(config: ExperimentConfig, camera: CameraModel | None = None):
    """Imported lazily so a run only loads the planner it uses."""
    if config.planner_mode == "grid":
        from experiment.grid.planner import GridPlanner

        return GridPlanner(config, camera=camera)
    if config.planner_mode == "advanced":
        from experiment.advanced.planner import AdvancedPlanner

        return AdvancedPlanner(config, camera=camera)
    return CenterlinePlanner(config, camera=camera)


def build_controller(
    config: ExperimentConfig,
    camera: CameraModel | None = None,
    grid_source: Callable[[], object] | None = None,
    clock: Callable[[], float] = time.perf_counter,
):
    """Imported lazily so a run only loads the controller it uses."""
    if config.control_mode == "imitation":
        from experiment.imitation.controller import ImitationController

        return ImitationController(config, camera=camera, clock=clock)
    if config.control_mode == "rl":
        from experiment.rl.controller import RLController

        return RLController(config, camera=camera, clock=clock)
    if config.control_mode == "mppi":
        from experiment.predictive.mppi_controller import MppiController

        return MppiController(config, camera=camera, grid_source=grid_source, clock=clock)
    if config.control_mode == "mpc":
        from experiment.predictive.mpc_controller import MpcController

        return MpcController(config, camera=camera, clock=clock)
    return StanleyController(config, camera=camera, clock=clock)


class ExperimentPipeline(AutonomyPipeline):
    def __init__(
        self, config: ExperimentConfig, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        # Read by _build_controller, which the parent constructor calls.
        self._clock = clock
        self._control_mode = config.control_mode
        super().__init__(config)

    def _build_planner(self, config: ExperimentConfig):
        return build_planner(config, camera=self.view.camera)

    def _build_controller(self, config: ExperimentConfig):
        return build_controller(
            config,
            camera=self.view.camera,
            grid_source=lambda: getattr(self.planner, "grid", None),
            clock=self._clock,
        )

    def _compute_command(
        self, result: PlanningStepResult, vehicle_state: VehicleState
    ) -> ControlCommand:
        if self._control_mode == "imitation":
            # The imitation policy steers from the image, so it reads the same
            # snapshot perception used rather than capturing its own.
            return self.controller.compute_observation(result.plan, vehicle_state, result.frame.raw)
        return self.controller.compute(result.plan, vehicle_state)
