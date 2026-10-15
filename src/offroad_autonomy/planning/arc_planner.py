"""Pick the best steerable path through the traversability grid.

This is the "tentacles" approach used on off-road research vehicles: a fan of
arcs the vehicle can actually steer, each scored on the grid, best one wins.
It never reports a path the grid does not support, and unlike a row-by-row
walk it cannot end early because one stretch of image was misleading: every
arc is judged on the whole corridor at once.

Besides the constant arcs, each arc also comes in two-part forms that
switch to a different curvature ``switch_m`` ahead of the camera, so a path
can run straight and then turn with the road, or come out of a bend and
straighten, instead of cutting across with one fixed curvature.

Arcs start at the rear axle, because that is the point a bicycle-model
vehicle turns about, and are expressed in the camera's ground frame that the
grid and Stanley use. No packaged library scores arcs on a custom
grid, so this part is written for this stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from offroad_autonomy.planning.bev_grid import TraversabilityGrid
from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.planning.trajectory_continuity import trajectory_distance, transform_path


def road_centre_offset(road: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """How far each road cell sits from the middle of its row's strip of road:
    0 on the centre line, approaching 1 at the edge.

    Only strips bounded by known non-road on both sides are scored; the rest
    are NaN. A strip that ends in unknown ground (off the side of the image,
    or under the hood before memory fills it) has no measurable middle, and
    treating its visible end as an edge would pull the path toward the side
    the camera happens to see.
    """
    rows, cols = road.shape
    idx = np.arange(cols)
    not_road = ~road
    left = np.maximum.accumulate(np.where(not_road, idx, -1), axis=1)
    right = np.minimum.accumulate(np.where(not_road, idx, cols)[:, ::-1], axis=1)[:, ::-1]
    row_index = np.arange(rows)[:, None]
    left_known = (left >= 0) & blocked[row_index, np.clip(left, 0, cols - 1)]
    right_known = (right < cols) & blocked[row_index, np.clip(right, 0, cols - 1)]
    centre = 0.5 * (left + right)
    half = np.maximum(0.5 * (right - left), 1.0)
    offset = np.clip(np.abs(idx - centre) / half, 0.0, 1.0)
    return np.where(road & left_known & right_known, offset, np.nan)


@dataclass
class ArcChoice:
    curvature: float
    forward_m: np.ndarray
    right_m: np.ndarray
    length_m: float
    min_clearance_m: float
    score: float
    continuity_distance_m: float = 0.0
    off_centre: float = 0.0
    #: Curvature at every point of the path (positive right); the first
    #: value equals ``curvature`` and a two-part path changes once.
    curvature_profile: np.ndarray | None = None


class ArcPlanner:
    def __init__(
        self,
        config: GridPlannerConfig,
        wheelbase_m: float,
        max_wheel_angle_deg: float,
        vehicle_half_width_m: float,
        rear_axle_behind_camera_m: float,
    ) -> None:
        self.config = config
        self._max_curvature = math.tan(math.radians(max_wheel_angle_deg)) / max(wheelbase_m, 0.1)
        # Positive curvature turns right, matching BeamNG steering.
        self.curvatures = np.linspace(-self._max_curvature, self._max_curvature, config.arc_count)
        self._required_clearance = float(vehicle_half_width_m) + config.clearance_margin_m

        s = np.arange(0.0, config.lookahead_m + rear_axle_behind_camera_m, config.step_m)
        first, second = self._curvature_pairs()
        switch = s[None, :] < rear_axle_behind_camera_m + config.switch_m
        profile = np.where(switch, first[:, None], second[:, None])
        # Integrate heading and position along the path (midpoint rule); the
        # first ``arc_count`` candidates are exactly the constant arcs.
        ds = config.step_m
        heading = np.concatenate(
            (np.zeros((len(first), 1)), np.cumsum(profile[:, :-1] * ds, axis=1)), axis=1
        )
        mid = heading + 0.5 * profile * ds
        start = np.zeros((len(first), 1))
        forward = np.concatenate((start, np.cumsum(np.cos(mid[:, :-1]) * ds, axis=1)), axis=1)
        right = np.concatenate((start, np.cumsum(np.sin(mid[:, :-1]) * ds, axis=1)), axis=1)
        self._forward = forward - rear_axle_behind_camera_m
        self._right = right
        self._profile = profile
        self._first = first
        self._previous: float | None = None
        self._previous_path: np.ndarray | None = None
        self._previous_pose = None
        self.branch_count = 0
        self.branch_locked = False

    def _curvature_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """(first, second) curvature of every candidate: the constant arcs
        first, in ``self.curvatures`` order, then the distinct two-part ones."""
        cfg = self.config
        arcs = len(self.curvatures)
        steps = cfg.second_curvature_step * np.arange(
            -cfg.second_curvature_count, cfg.second_curvature_count + 1
        )
        first = list(self.curvatures)
        second = list(self.curvatures)
        for k1 in self.curvatures:
            for step in steps:
                k2 = float(np.clip(k1 + step, -self._max_curvature, self._max_curvature))
                if abs(k2 - k1) > 1e-9:
                    first.append(float(k1))
                    second.append(k2)
        pairs = np.unique(np.column_stack((first[arcs:], second[arcs:])), axis=0)
        if len(pairs) == 0:
            return np.asarray(first), np.asarray(second)
        return (
            np.concatenate((self.curvatures, pairs[:, 0])),
            np.concatenate((self.curvatures, pairs[:, 1])),
        )

    def preview_arcs(self) -> tuple[np.ndarray, ...]:
        """Sparse fan of the constant arcs for display; not all are drivable."""
        arcs = len(self.curvatures)
        return tuple(
            np.column_stack((f, r)).copy()
            for f, r in zip(self._forward[:arcs:4], self._right[:arcs:4])
        )

    def reset(self) -> None:
        self._previous = None
        self._previous_path = None
        self._previous_pose = None
        self.branch_count = 0
        self.branch_locked = False

    def choose(self, grid: TraversabilityGrid, pose=None) -> ArcChoice | None:
        cfg = self.config
        clearance = grid.clearance_m()
        road = grid.road()
        rows, cols = grid.to_cell(self._forward, self._right)
        inside = (rows >= 0) & (rows < grid.rows) & (cols >= 0) & (cols < grid.cols)
        r = np.clip(rows, 0, grid.rows - 1)
        c = np.clip(cols, 0, grid.cols - 1)
        arc_clearance = np.where(inside, clearance[r, c], 0.0)
        arc_road = inside & road[r, c]
        arc_offset = np.where(inside, road_centre_offset(road, grid.blocked())[r, c], np.nan)
        ahead = self._forward >= cfg.start_m

        candidates = []
        reference = self._previous_path
        if reference is not None:
            if pose is not None and self._previous_pose is not None:
                reference = transform_path(reference, self._previous_pose, pose)
            else:
                # Without registration, do not pretend old ego-frame points
                # are still where the vehicle saw them.
                reference = None
                self.branch_locked = False
        for i, curvature in enumerate(self._first):
            candidate = self._score(
                i,
                float(curvature),
                arc_clearance[i],
                arc_road[i],
                ahead[i],
                inside[i],
                arc_offset[i],
            )
            if candidate is not None and reference is not None:
                candidate.continuity_distance_m = trajectory_distance(
                    np.column_stack((candidate.forward_m, candidate.right_m)), reference
                )
                candidate.score -= cfg.weight_trajectory_distance * candidate.continuity_distance_m
            if candidate is not None:
                candidates.append(candidate)
        candidates = self._keep_branch(candidates, grid, road, clearance, reference)
        if self.branch_count >= 2:
            # At a split the road has no single middle: a strip of road that
            # spans both branches puts its centre between them. The branch
            # logic has already chosen which way to go; centring sits out.
            for candidate in candidates:
                candidate.score += cfg.weight_centering * candidate.off_centre
        best = max(candidates, key=lambda candidate: candidate.score, default=None)
        if best is not None:
            self._previous = best.curvature
            self._previous_path = np.column_stack((best.forward_m, best.right_m))
            self._previous_pose = pose
        else:
            self.reset()
        return best

    def _split_options(self, candidates, grid, road, clearance):
        """Find distinct vehicle-width corridors reached by valid arcs.

        A fan of arcs on one wide road is one option, not dozens of branches.
        Arcs ending in the common trunk before the split cannot vote for a branch.
        """
        if len(candidates) < 2:
            return None
        best = None
        most = 1
        distances = np.arange(self.config.start_m, self.config.lookahead_m, self.config.step_m)
        # Lateral position of every candidate at every distance, once; a
        # per-distance, per-candidate loop cost tens of ms with two-part paths.
        lateral = np.full((len(candidates), len(distances)), np.nan)
        for j, candidate in enumerate(candidates):
            forward, indices = np.unique(candidate.forward_m, return_index=True)
            within = (distances >= forward[0]) & (distances <= forward[-1])
            lateral[j, within] = np.interp(distances[within], forward, candidate.right_m[indices])
        rows, _ = grid.to_cell(distances, np.zeros_like(distances))
        for d, distance in enumerate(distances):
            row = int(rows[d])
            if not 0 <= row < grid.rows:
                continue
            # Unseen ground (especially the hood exclusion) is not a divider.
            # Only a known obstacle/boundary can separate two road branches.
            safe = clearance[row] >= self._required_clearance
            starts = safe & ~np.r_[False, safe[:-1]]
            if np.count_nonzero(starts) < 2:
                continue
            labels = np.where(safe, np.cumsum(starts), 0)
            groups = {}
            present = np.flatnonzero(np.isfinite(lateral[:, d]))
            _, cols = grid.to_cell(np.full(len(present), distance), lateral[present, d])
            for j, col in zip(present, cols):
                if 0 <= col < grid.cols and labels[col] != 0 and road[row, col]:
                    groups.setdefault(int(labels[col]), []).append(candidates[j])
            if len(groups) > most:
                most = len(groups)
                best = distance, labels, groups
        return best

    def _keep_branch(self, candidates, grid, road, clearance, reference):
        split = self._split_options(candidates, grid, road, clearance)
        if split is None:
            self.branch_count = int(bool(candidates))
            self.branch_locked = False
            return candidates

        distance, labels, groups = split
        self.branch_count = len(groups)
        selected = None
        if self.branch_locked and reference is not None:
            forward, indices = np.unique(reference[:, 0], return_index=True)
            if forward[0] <= distance <= forward[-1]:
                lateral = np.interp(distance, forward, reference[indices, 1])
                _, col = grid.to_cell(np.asarray(distance), np.asarray(lateral))
                if 0 <= col < grid.cols:
                    # Match corridor membership, not its changing list index or
                    # score. Losing another branch must not change our branch.
                    selected = groups.get(int(labels[col]))

        if selected is None:
            # First split (or the chosen branch no longer has a safe candidate):
            # take the branch whose road goes furthest, so a short spur or a
            # dead end is not chosen just because it was nearer. Branches
            # within branch_depth_margin_m of the deepest count as equal, and
            # among those the closest wins, before arc-quality scores.
            fits = road & (clearance >= self._required_clearance)
            depth = self._branch_depths(grid, fits, distance, labels, groups)
            deepest = max(depth.values())
            groups = {
                label: group
                for label, group in groups.items()
                if depth[label] >= deepest - self.config.branch_depth_margin_m
            }
            target = reference
            if target is None or np.max(target[:, 0]) <= self.config.start_m:
                target = np.array([[0.0, 0.0], [self.config.lookahead_m, 0.0]])

            def nearest(group):
                separation = min(
                    trajectory_distance(
                        np.column_stack((candidate.forward_m, candidate.right_m)), target
                    )
                    for candidate in group
                )
                return separation, -max(candidate.score for candidate in group)

            selected = min(groups.values(), key=nearest)
        self.branch_locked = True
        return selected

    @staticmethod
    def _branch_depths(grid, fits, distance, labels, groups) -> dict[int, float]:
        """How far ahead each branch's road continues, through the whole grid
        (out to max_range_m), not just the planning look-ahead. ``fits`` is
        road the vehicle fits on (the split detector's rule): past the split
        the branches are separate patches of it, where plain road still joins
        them for the last metre or so before the median, and a patch that
        stops short is a spur or a dead end."""
        forward, _ = grid.cell_centres()
        beyond = (fits & (forward >= distance)).astype(np.uint8)
        _, patches = cv2.connectedComponents(beyond, connectivity=8)
        row, _ = grid.to_cell(np.asarray(distance), np.asarray(0.0))
        depth = {}
        for label in groups:
            ids = np.unique(patches[int(row), labels == label])
            ids = ids[ids != 0]
            depth[label] = float(distance)
            if len(ids):
                depth[label] = float(forward[np.isin(patches, ids)].max())
        return depth

    def _score(
        self,
        index: int,
        curvature: float,
        clearance: np.ndarray,
        road: np.ndarray,
        ahead: np.ndarray,
        inside: np.ndarray,
        offset: np.ndarray,
    ) -> ArcChoice | None:
        cfg = self.config
        # The body would leave the road, or the arc leaves the grid: the arc
        # ends there, however good the ground beyond might look.
        blocked = (ahead & (clearance < self._required_clearance)) | ~inside
        end = len(clearance)
        if blocked.any():
            end = int(np.argmax(blocked))
        drivable = ahead.copy()
        drivable[end:] = False
        supported = np.flatnonzero(drivable & road)
        if len(supported) == 0:
            return None
        first = int(np.argmax(drivable))
        last = int(supported[-1])
        # Length counts only ground the grid has seen as road, so the path
        # (and with it the controller's path-end speed) never runs into the unknown.
        length = (last - first) * cfg.step_m
        if length < cfg.min_path_m:
            return None

        span = slice(first, last + 1)
        # Reach is progress ahead on seen road, not path length: counting
        # samples let a path that wiggles and ends just inside the observed
        # range beat a straight one that pokes a sample past it. A straight
        # path into a bend still loses, because it runs off the road early.
        forward = self._forward[index]
        reach = (forward[last] - forward[first]) / max(cfg.lookahead_m - cfg.start_m, 1e-6)
        reach = min(max(reach, 0.0), 1.0)
        capped = np.minimum(clearance[span], cfg.clearance_cap_m) / cfg.clearance_cap_m
        bend = float(np.abs(self._profile[index, span]).mean()) / self._max_curvature
        change = 0.0
        if self._previous is not None:
            change = abs(curvature - self._previous) / (2.0 * self._max_curvature)
        # Clearance alone does not centre: it treats unseen ground as open, so
        # a side the camera cannot see looks endlessly clear.
        measured = offset[span]
        finite = np.isfinite(measured)
        off_centre = 0.0
        if finite.any():
            off_centre = float(measured[finite].mean())
        score = (
            cfg.weight_length * reach
            + cfg.weight_clearance * float(capped.mean())
            - cfg.weight_consistency * change
            - cfg.weight_curvature * bend
            - cfg.weight_centering * off_centre
        )
        return ArcChoice(
            curvature=curvature,
            forward_m=self._forward[index, span].copy(),
            right_m=self._right[index, span].copy(),
            length_m=length,
            min_clearance_m=float(clearance[span].min()),
            score=score,
            off_centre=off_centre,
            curvature_profile=self._profile[index, span].copy(),
        )
