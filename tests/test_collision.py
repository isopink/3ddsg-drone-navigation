import itertools
import math

import numpy as np
import pytest

from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.graph.model import Box, VoxelState
from ddsg_nav.planner.collision import (
    check_path,
    check_points,
    check_trajectory,
    sample_path,
)
from ddsg_nav.planner.route import plan_route


@pytest.fixture(scope="module")
def graph():
    return build_fake_graph()


# ---- checker itself --------------------------------------------------------

def test_sample_path_spacing():
    pts = sample_path([(0, 0, 0), (1, 0, 0)], 0.25)
    assert pts[0] == (0, 0, 0) and pts[-1] == (1, 0, 0)
    assert len(pts) == 5
    assert max(abs(b[0] - a[0]) for a, b in zip(pts, pts[1:])) <= 0.25 + 1e-9


def test_check_points_classifies_violations(graph):
    free = (1.5, 1.5, 0.5)
    margin = (2.7, 0.5, 0.5)  # 0.2 m from the partition at x=2.9, inside the 0.25 m margin
    hard = (3.0, 0.5, 0.5)  # inside the partition wall
    outside = (20.0, 1.5, 0.5)
    r = check_points(graph, [free, margin, hard, outside])
    assert r.samples == 4
    assert r.hard == [hard]
    assert r.margin == [margin]
    assert r.outside == [outside]
    assert not r.ok and not r.collision_free


def test_check_path_detects_wall_crossing(graph):
    # straight line from room A into room B at y=0.5 goes through the wall
    r = check_path(graph, [(1.5, 0.5, 0.5), (4.5, 0.5, 0.5)])
    assert r.hard


def test_check_path_passes_through_door_center(graph):
    r = check_path(graph, [(1.5, 1.5, 0.5), (4.5, 1.5, 0.5)])
    assert r.ok, r.summary()


def test_unknown_voxels_count_as_hard():
    g = build_fake_graph()
    g.voxels.fill_box(Box((1.0, 1.0, 0.0), (2.0, 2.0, 1.0)), VoxelState.UNKNOWN)
    p = [(1.5, 1.5, 0.5)]
    assert check_points(g, p).hard
    assert check_points(g, p, unknown_is_obstacle=False).ok


def test_check_trajectory_array(graph):
    traj = np.array([[1.5, 1.5, 0.3], [2.0, 1.5, 0.3], [3.0, 1.5, 0.3]])
    assert check_trajectory(graph, traj).ok


# ---- many start/goal combinations (done criterion: zero collisions) ----

def test_goal_moves_when_room_center_is_blocked():
    g = build_fake_graph(obstacle=True)  # box in the middle of room B
    plan = plan_route(g, "A", "B", height=0.5)
    goal = plan.waypoints[-1]
    center = (4.5, 1.5, 0.5)
    assert goal != center and 0.25 < math.dist(goal, center) < 1.5
    assert g.room_at(goal).id == "B"
    assert check_path(g, plan.waypoints).ok


PAIRS = list(itertools.permutations("ABC", 2))  # 6 ordered pairs
HEIGHTS = [0.3, 0.5, 1.0]


@pytest.mark.parametrize("obstacle", [False, True])
@pytest.mark.parametrize("height", HEIGHTS)
def test_all_start_goal_pairs_collision_free(obstacle, height):
    g = build_fake_graph(obstacle=obstacle)
    for start, goal in PAIRS:
        plan = plan_route(g, start, goal, height=height)
        report = check_path(g, plan.waypoints)
        assert report.ok, f"{start}->{goal} h={height} obstacle={obstacle}: {report.summary()}"
