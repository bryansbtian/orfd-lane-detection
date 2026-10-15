"""Settings for the predictive steering controllers (``control.mode: mppi``
or ``mpc``), in SI units; vehicle geometry stays in PipelineConfig."""

import math
from dataclasses import dataclass, fields

#: Steering controllers ``control.mode`` can select.
CONTROL_MODES = ("stanley", "mppi", "mpc")


@dataclass
class PredictiveConfig:
    # Prediction: model step and horizon.
    dt_s: float = 0.1
    horizon_steps: int = 30
    # BeamNG's steering input slews at about this rate (command units per
    # second; measured on the Hopper, the same value scripts/sbend_sim.py uses).
    steer_slew_per_s: float = 0.9
    # Slowest speed assumed for prediction, so a standing start still steers.
    min_predict_speed_mps: float = 0.5
    # Largest change in the sent command between model steps.
    max_command_step: float = 0.3

    # Cost weights, shared by both controllers. Lateral and heading error are
    # read at the camera point against the planned metric path.
    weight_lateral: float = 1.0
    weight_heading: float = 2.0
    weight_steer_rate: float = 4.0
    # Steps whose camera point is past the planned path's end count this much.
    beyond_path_weight: float = 0.2

    # MPPI only.
    samples: int = 512
    iterations: int = 1
    temperature: float = 0.5
    # Noise on the per-step change in command (smooth MPPI samples changes,
    # not commands, which is what keeps it from chattering).
    rate_noise: float = 0.08
    # Off-road cost from the BEV grid: penalty per metre the clearance at the
    # camera point falls short of vehicle half width + margin.
    weight_offroad: float = 20.0
    offroad_margin_m: float = 0.2

    # MPC only: Gauss-Newton iterations on the least-squares problem.
    gauss_newton_iterations: int = 2

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"control.predictive.{f.name} must be finite and numeric")
        for name in ("horizon_steps", "samples", "iterations", "gauss_newton_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"control.predictive.{name} must be an integer >= 1")
        for name in ("dt_s", "steer_slew_per_s", "max_command_step", "temperature", "rate_noise"):
            if getattr(self, name) <= 0:
                raise ValueError(f"control.predictive.{name} must be > 0")
        for name in (
            "min_predict_speed_mps",
            "weight_lateral",
            "weight_heading",
            "weight_steer_rate",
            "beyond_path_weight",
            "weight_offroad",
            "offroad_margin_m",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"control.predictive.{name} must be >= 0")
