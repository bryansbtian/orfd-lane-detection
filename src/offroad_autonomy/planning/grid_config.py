"""Grid planner tuning in SI units; vehicle geometry stays in PipelineConfig."""

import math
from dataclasses import dataclass, fields


@dataclass
class GridPlannerConfig:
    cell_m: float = 0.10
    ahead_m: float = 26.0
    behind_m: float = 4.0
    half_width_m: float = 10.0
    memory_s: float = 4.0
    road_logodds: float = 0.5
    offroad_logodds: float = -0.5
    logodds_limit: float = 4.0
    road_threshold: float = 0.5
    blocked_threshold: float = -0.5
    arc_count: int = 41
    lookahead_m: float = 14.0
    step_m: float = 0.25
    start_m: float = 1.0
    min_path_m: float = 4.0
    clearance_margin_m: float = 0.1
    clearance_cap_m: float = 4.0
    weight_length: float = 1.0
    weight_clearance: float = 1.0
    weight_consistency: float = 0.2
    weight_curvature: float = 0.1
    weight_centering: float = 2.0
    weight_trajectory_distance: float = 0.10  # Score penalty per metre of path separation.
    # Two-part candidates: each arc may switch to another curvature this far
    # ahead of the camera, by +/- count steps of this size. Count 0 = arcs only.
    switch_m: float = 4.0
    second_curvature_step: float = 0.04
    second_curvature_count: int = 2
    # The grid observes ground out to this range, independently of the gate ROI.
    max_range_m: float = 24.0
    # At a split, branches whose road reaches within this of the deepest one
    # count as equally deep; the closest of them is then taken.
    branch_depth_margin_m: float = 3.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"planning.grid.{f.name} must be finite and numeric")
        if not isinstance(self.second_curvature_count, int) or self.second_curvature_count < 0:
            raise ValueError("planning.grid.second_curvature_count must be an integer >= 0")
        if not isinstance(self.arc_count, int) or self.arc_count < 3 or self.arc_count % 2 == 0:
            raise ValueError("planning.grid.arc_count must be an odd integer >= 3 so 0 is an arc")
        for name in (
            "cell_m",
            "ahead_m",
            "behind_m",
            "half_width_m",
            "memory_s",
            "road_logodds",
            "logodds_limit",
            "lookahead_m",
            "step_m",
            "min_path_m",
            "clearance_cap_m",
            "switch_m",
            "max_range_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"planning.grid.{name} must be > 0")
        if self.offroad_logodds >= 0:
            raise ValueError("planning.grid.offroad_logodds must be < 0")
        if not self.blocked_threshold < self.road_threshold:
            raise ValueError("planning.grid.blocked_threshold must be below road_threshold")
        if self.step_m >= self.cell_m * 5:
            raise ValueError("planning.grid.step_m must be under 5 cells or arcs skip off-road")
        for name in (
            "start_m",
            "clearance_margin_m",
            "weight_length",
            "weight_clearance",
            "weight_consistency",
            "weight_curvature",
            "weight_centering",
            "weight_trajectory_distance",
            "second_curvature_step",
            "branch_depth_margin_m",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"planning.grid.{name} must be >= 0")
