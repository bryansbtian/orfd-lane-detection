from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from offroad_autonomy.planning.arc_planner import ArcChoice
from offroad_autonomy.planning.trajectory_continuity import trajectory_distance, transform_path
from tests.test_grid_planner import _arcs, _grid, _pose, _road_mask, _straight


def test_distance_is_metric_and_independent_of_point_order_and_density():
    f = np.linspace(1, 10, 6)
    reference = np.column_stack((f, f * 0.1))
    dense = np.linspace(1, 10, 50)
    same = np.column_stack((dense, dense * 0.1))
    shifted = same + [0, 1.5]
    assert trajectory_distance(same[::-1], reference) == pytest.approx(0, abs=1e-12)
    assert trajectory_distance(shifted, reference) == pytest.approx(1.5)
    assert trajectory_distance(same + [30, 0], reference) == 0


def test_motion_compensation_translates_and_rotates_the_previous_path():
    points = np.array([[5.0, 0.0], [10.0, 0.0]])
    moved = transform_path(points, _pose(), _pose(x=2, y=1))
    np.testing.assert_allclose(moved, [[3, 1], [8, 1]])
    turned = transform_path(points, _pose(), _pose(heading_rad=np.pi / 2))
    np.testing.assert_allclose(turned, [[0, 5], [0, 10]], atol=1e-12)


def test_closest_valid_candidate_wins_over_small_quality_fluctuations():
    arcs, grid = _arcs(), _grid()
    f = np.linspace(1, 10, 24)
    same = ArcChoice(0.02, f, f * 0 + 0.2, 9, 2, 1.0)
    switch = ArcChoice(-0.02, f, f * 0 - 1.0, 9, 2, 1.1)
    arcs._previous_path = np.column_stack((f, f * 0 + 0.2))
    arcs._previous_pose = _pose()
    with patch.object(
        arcs, "_score", side_effect=[same, switch] + [None] * (len(arcs.curvatures) - 2)
    ):
        choice = arcs.choose(grid, _pose())
    assert choice is same
    assert same.continuity_distance_m == pytest.approx(0)
    assert switch.continuity_distance_m == pytest.approx(1.2)


def test_unsafe_previous_path_never_overrides_current_clearance():
    arcs, grid = _arcs(), _grid()
    arcs.config = replace(arcs.config, weight_trajectory_distance=1000)
    f = np.linspace(1, 12, 24)
    arcs._previous_path = np.column_stack((f, f * 0 + 6))
    arcs._previous_pose = _pose()
    grid.update(_road_mask(_straight(2)), _pose(), dt_s=0)
    choice = arcs.choose(grid, _pose())
    assert choice is not None
    assert choice.min_clearance_m >= arcs._required_clearance
    assert np.abs(choice.right_m).max() < 2
    grid.logodds[:] = -4
    assert arcs.choose(grid, _pose()) is None
    assert arcs._previous_path is None


def test_missing_pose_and_reset_discard_geometric_preference():
    arcs, grid = _arcs(), _grid()
    grid.update(_road_mask(_straight(2)), _pose(), dt_s=0)
    arcs.choose(grid, _pose())
    assert arcs._previous_path is not None
    assert arcs.choose(grid).continuity_distance_m == 0
    arcs.reset()
    assert arcs._previous_path is None and arcs._previous_pose is None


def _fork_grid(branches=(-1, 0, 1)):
    """Three known traversable arms sharing a trunk, using the real BEV grid."""
    grid = _grid()
    forward, right = grid.cell_centres()
    bend = 0.072
    offset = (1 - np.sqrt(1 - np.clip(bend * (forward + 1.4), -1, 1) ** 2)) / bend
    road = np.zeros_like(forward, dtype=bool)
    for branch in branches:
        road |= np.abs(right - branch * offset) < 1.7
    grid.logodds[:] = np.where(road, 4.0, -4.0)
    return grid


def _prefer_right(arcs):
    index = int(np.argmin(np.abs(arcs.curvatures - 0.072)))
    ahead = arcs._forward[index] >= arcs.config.start_m
    arcs._previous_path = np.column_stack((arcs._forward[index, ahead], arcs._right[index, ahead]))
    arcs._previous_pose = _pose()


def test_three_way_split_picks_closest_branch_and_ignores_other_branch_scores():
    arcs, grid = _arcs(), _fork_grid()
    first = arcs.choose(grid, _pose())
    assert arcs.branch_count == 3 and arcs.branch_locked
    assert abs(first.curvature) < 0.02
    original = arcs._score

    def favor_right(*args):
        choice = original(*args)
        if choice is not None and choice.curvature > 0.05:
            choice.score += 1000
        return choice

    with patch.object(arcs, "_score", side_effect=favor_right):
        for _ in range(5):
            chosen = arcs.choose(grid, _pose())
            assert arcs.branch_count == 3 and arcs.branch_locked
            assert abs(chosen.curvature) < 0.02


def test_chosen_branch_survives_three_to_two_options_then_releases_at_one():
    arcs = _arcs()
    _prefer_right(arcs)
    original = arcs._score

    def favor_straight(*args):
        choice = original(*args)
        if choice is not None and abs(choice.curvature) < 0.02:
            choice.score += 1000
        return choice

    with patch.object(arcs, "_score", side_effect=favor_straight):
        chosen = arcs.choose(_fork_grid(), _pose())
    assert arcs.branch_count == 3 and arcs.branch_locked
    assert chosen.curvature > 0.05
    with patch.object(arcs, "_score", side_effect=favor_straight):
        chosen = arcs.choose(_fork_grid((0, 1)), _pose())
    assert arcs.branch_count == 2 and arcs.branch_locked
    assert chosen.curvature > 0.05
    chosen = arcs.choose(_fork_grid((0,)), _pose())
    assert arcs.branch_count == 1 and not arcs.branch_locked
    assert abs(chosen.curvature) < 0.02
    chosen = arcs.choose(_fork_grid(), _pose())
    assert arcs.branch_count == 3 and arcs.branch_locked
    assert abs(chosen.curvature) < 0.02  # The old right-branch lock was released.


def test_branch_lock_tracks_ego_motion_instead_of_a_fixed_curvature_index():
    arcs, grid = _arcs(), _fork_grid()
    _prefer_right(arcs)
    assert arcs.choose(grid, _pose()).curvature > 0.05
    moved = _pose(x=0.5, y=-0.1, heading_rad=-0.02)
    grid._shift(_pose(), moved)
    chosen = arcs.choose(grid, moved)
    assert arcs.branch_count == 3 and arcs.branch_locked
    assert chosen.curvature > 0.05


def test_blocked_chosen_branch_can_be_replaced_by_a_safe_branch():
    arcs = _arcs()
    _prefer_right(arcs)
    assert arcs.choose(_fork_grid(), _pose()).curvature > 0.05
    chosen = arcs.choose(_fork_grid((-1, 0)), _pose())
    assert arcs.branch_count == 2
    assert chosen.curvature < 0.02
    assert chosen.min_clearance_m >= arcs._required_clearance
    blocked = _fork_grid()
    blocked.logodds[:] = -4
    assert arcs.choose(blocked, _pose()) is None
    assert arcs.branch_count == 0 and not arcs.branch_locked


def test_many_arcs_on_one_road_are_not_mistaken_for_a_split():
    arcs, grid = _arcs(), _grid()
    grid.logodds[:] = 4.0
    assert arcs.choose(grid, _pose()) is not None
    assert arcs.branch_count == 1 and not arcs.branch_locked
    arcs.reset()
    assert arcs.branch_count == 0 and not arcs.branch_locked
