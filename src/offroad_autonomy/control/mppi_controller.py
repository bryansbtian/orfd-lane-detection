"""Smooth MPPI steering (``control.mode: mppi``).

Model Predictive Path Integral control samples many steering sequences,
predicts each through the vehicle model, scores them, and averages them
weighted by ``exp(-cost / temperature)`` (Williams et al., the AutoRally
work). Plain MPPI sampling raw commands chatters on a vehicle whose steering
is slow and delayed; as in smooth MPPI (Kim et al. 2022), the samples here
are changes in the command, penalised for size, so every sampled sequence is
already smooth.

Unlike the tracking-only MPC, the cost also reads the BEV grid the planner
keeps: a sequence that puts the camera point closer to non-road than the
vehicle's half width plus a margin pays for it, so the controller can leave
the planned path to stay on the road. Costs, in the spirit of Nav2's MPPI
critics: path lateral and heading error, command change, off-road.
"""

from __future__ import annotations

import numpy as np

from offroad_autonomy.control.predictive import (
    PredictiveSteering,
    rollout,
    tracking_residuals,
)
from offroad_autonomy.control.stanley_controller import _SampledPath


class MppiController(PredictiveSteering):
    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed)
        self._nominal = np.zeros(self._cfg.horizon_steps)

    def _reset_solution(self) -> None:
        self._nominal = np.zeros(self._cfg.horizon_steps)

    def _solve(
        self, path: _SampledPath, committed: np.ndarray, speed: float, input0: float
    ) -> float:
        cfg = self._cfg
        free = cfg.horizon_steps - len(committed)
        if free < 1:
            return self._last_command
        nominal = self._nominal[:free]
        offroad = self._offroad_lookup()
        for _ in range(cfg.iterations):
            noise = self._rng.normal(0.0, cfg.rate_noise, (cfg.samples, free))
            noise[0] = 0.0  # the warm-started nominal is always a candidate
            change = np.clip(nominal + noise, -cfg.max_command_step, cfg.max_command_step)
            commands = np.clip(self._last_command + np.cumsum(change, axis=1), -1.0, 1.0)
            # The command sequence actually used, after clipping at full lock.
            change = np.diff(commands, axis=1, prepend=self._last_command)
            targets = np.concatenate(
                (np.broadcast_to(committed, (cfg.samples, len(committed))), commands), axis=1
            )
            fc, rc, heading = rollout(
                targets, input0, speed, cfg.dt_s, cfg.steer_slew_per_s, self._geometry
            )
            lateral, head = tracking_residuals(path, fc, rc, heading, cfg)
            cost = (lateral**2 + head**2).mean(axis=1)
            cost += cfg.weight_steer_rate * (change**2).sum(axis=1)
            if offroad is not None:
                cost += offroad(fc, rc)
            weights = np.exp(-(cost - cost.min()) / cfg.temperature)
            weights /= weights.sum()
            nominal = (weights[:, None] * change).sum(axis=0)
        command = float(np.clip(self._last_command + nominal[0], -1.0, 1.0))
        # Warm start: the rest of this plan is next frame's first guess.
        self._nominal = np.concatenate((nominal[1:], [0.0], np.zeros(len(committed))))
        return command

    def _offroad_lookup(self):
        """``f(fc, rc) -> (samples,)`` off-road cost from the grid, or None."""
        grid = self._grid()
        if grid is None:
            return None
        cfg = self._cfg
        clearance = grid.clearance_m()
        required = self._geometry.half_width + cfg.offroad_margin_m

        def cost(fc: np.ndarray, rc: np.ndarray) -> np.ndarray:
            row, col = grid.to_cell(fc, rc)
            inside = (row >= 0) & (row < grid.rows) & (col >= 0) & (col < grid.cols)
            r = np.clip(row, 0, grid.rows - 1)
            c = np.clip(col, 0, grid.cols - 1)
            # Outside the grid nothing is known; it is not penalised.
            short = np.where(inside, np.maximum(required - clearance[r, c], 0.0), 0.0)
            return cfg.weight_offroad * short.mean(axis=1)

        return cost
