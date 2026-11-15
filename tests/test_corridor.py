"""Room -> L-shaped hallway -> room: planning, validation and a simulated flight."""
import itertools

import pytest

from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_corridor_graph
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph


@pytest.fixture(scope="module")
def graph():
    return build_corridor_graph()


def test_hallway_centroid_lies_outside_the_hallway(graph):
    # the reason this layout exists: an L-shape's centroid is not inside it
    h = graph.rooms["H"]
    assert not h.contains(h.center)


def test_graph_validates(graph):
    assert validate_graph(graph, 0.5, 0.25).ok


def test_route_turns_through_the_hallway(graph):
    goal = parse_goal("A방에서 C방으로 가줘", graph)
    plan = plan_route(graph, goal.start, goal.goal, height=0.5)
    assert plan.rooms == ["A", "H", "C"]
    assert plan.doors == ["door_AH", "door_HC"]
    xs = [p[0] for p in plan.waypoints]
    ys = [p[1] for p in plan.waypoints]
    assert max(xs) - min(xs) > 5 and max(ys) - min(ys) > 6  # east along the hallway, then north


@pytest.mark.parametrize("start,goal", list(itertools.permutations("AHC", 2)))
def test_every_pair_is_collision_free_and_ends_inside_the_goal(graph, start, goal):
    plan = plan_route(graph, start, goal, height=0.5)
    assert check_path(graph, plan.waypoints).ok
    assert graph.rooms[start].contains(plan.waypoints[0])
    assert graph.rooms[goal].contains(plan.waypoints[-1])


def test_hallway_alias(graph):
    assert parse_goal("A방에서 복도로", graph).goal == "H"


def test_simulated_flight_has_no_wall_contact(graph):
    pytest.importorskip("mujoco")
    import numpy as np

    from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend
    from ddsg_nav.flight.executor import follow_path
    from ddsg_nav.planner.collision import check_trajectory

    plan = plan_route(graph, "A", "C", height=0.5)
    start = plan.waypoints[0]
    with MuJoCoBackend(graph, (start[0], start[1], 0.0)) as drone:
        follow_path(drone, plan.waypoints, velocity=0.3)
        traj = np.array(drone.trajectory)[:, 1:]
        assert drone.collisions == 0
    assert check_trajectory(graph, traj).ok
