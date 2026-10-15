"""Delay-aware tracking MPC steering (``control.mode: mpc``).

Chooses the command sequence over the horizon that minimises lateral and
heading error to the planned path plus the size of each command change,
through the same prediction MPPI uses (commands in flight replayed first,
steering slew limited). The problem is a small nonlinear least squares,
solved by Gauss-Newton with a finite-difference Jacobian taken from one
batched prediction. Deterministic and smooth; it tracks the planner's path
and does not read the BEV grid.
"""

from __future__ import annotations

import numpy as np

from offroad_autonomy.control.predictive import (
    PredictiveSteering,
    rollout,
    tracking_residuals,
)
from offroad_autonomy.control.stanley_controller import _SampledPath

#: Command perturbation for the finite-difference Jacobian; well under the
#: slew per step (0.09) so the limit rarely hides the gradient.
_PROBE = 1e-3
#: Levenberg damping on the normal equations.
_DAMPING = 1e-4


class MpcController(PredictiveSteering):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._nominal = np.zeros(self._cfg.horizon_steps)

    def _reset_solution(self) -> None:
        self._nominal = np.zeros(self._cfg.horizon_steps)

    def _residuals(self, path, committed, commands, speed, input0):
        """Stacked residuals for each row of ``commands``: lateral and heading
        error per step, then each command change."""
        cfg = self._cfg
        targets = np.concatenate(
            (np.broadcast_to(committed, (len(commands), len(committed))), commands), axis=1
        )
        fc, rc, heading = rollout(
            targets, input0, speed, cfg.dt_s, cfg.steer_slew_per_s, self._geometry
        )
        lateral, head = tracking_residuals(path, fc, rc, heading, cfg)
        scale = 1.0 / np.sqrt(fc.shape[1])  # mean over steps, as in MPPI
        change = np.diff(commands, axis=1, prepend=self._last_command)
        return np.concatenate(
            (scale * lateral, scale * head, np.sqrt(cfg.weight_steer_rate) * change), axis=1
        )

    def _solve(
        self, path: _SampledPath, committed: np.ndarray, speed: float, input0: float
    ) -> float:
        cfg = self._cfg
        free = cfg.horizon_steps - len(committed)
        if free < 1:
            return self._last_command
        commands = np.clip(self._last_command + np.cumsum(self._nominal[:free]), -1.0, 1.0)
        for _ in range(cfg.gauss_newton_iterations):
            # Row 0 is the current sequence, row j+1 nudges command j.
            batch = np.repeat(commands[None, :], free + 1, axis=0)
            batch[np.arange(1, free + 1), np.arange(free)] += _PROBE
            residuals = self._residuals(path, committed, batch, speed, input0)
            r0 = residuals[0]
            jacobian = ((residuals[1:] - r0) / _PROBE).T
            normal = jacobian.T @ jacobian + _DAMPING * np.eye(free)
            delta = np.linalg.solve(normal, -jacobian.T @ r0)
            commands = np.clip(commands + delta, -1.0, 1.0)
        change = np.diff(commands, prepend=self._last_command)
        change = np.clip(change, -cfg.max_command_step, cfg.max_command_step)
        self._nominal = np.concatenate((change[1:], [0.0], np.zeros(len(committed))))
        return float(np.clip(self._last_command + change[0], -1.0, 1.0))
