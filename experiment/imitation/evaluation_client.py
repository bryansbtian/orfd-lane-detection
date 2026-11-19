"""BeamNG client for matched comparison trials: a procedural road and
deterministic, stepped physics, so every controller sees the same run."""

from __future__ import annotations

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import PipelineConfig


class EvaluationClient(BeamNGClient):
    def __init__(self, config: PipelineConfig, route: dict, physics_hz: int) -> None:
        super().__init__(config)
        self._route = route
        self._physics_hz = physics_hz

    def _make_scenario(self, scenario) -> None:
        # Imported here, as in BeamNGClient.connect, so the module loads without beamngpy.
        from beamngpy import Road

        road = Road(self._route["material"], default_width=self._route["width_m"])
        road.add_nodes(*self._route["nodes"])
        scenario.add_road(road)
        super()._make_scenario(scenario)
        self._bng.settings.set_deterministic(steps_per_second=self._physics_hz)
        self._bng.control.pause()

    def _settle(self, seconds: float) -> None:
        # Wall-clock sleeps would let physics run unevenly between trials.
        self._bng.control.pause()
        self._bng.control.step(int(seconds * self._physics_hz), wait=True)

    def begin_evaluation(self, physics_hz: int) -> None:
        """Deterministic stepping is confined to the simulation evaluation harness."""
        self._bng.settings.set_deterministic(steps_per_second=physics_hz)
        self._bng.control.pause()

    def step_evaluation(self, count: int) -> None:
        self._bng.control.step(count, wait=True)

    def evaluation_state(self) -> dict:
        """World pose is evaluation ground truth; production perception never consumes it."""
        self._vehicle.sensors.poll("state")
        return dict(self._vehicle.state)
