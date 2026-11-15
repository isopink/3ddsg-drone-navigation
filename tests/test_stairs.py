"""Room A (floor 1) -> stairwell -> room C (floor 2): heights are measured per floor."""
import itertools

import pytest

from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_fake_graph, build_stairs_graph
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

H = 0.5


@pytest.fixture(scope="module")
def graph():
    return build_stairs_graph()


def test_floor_levels(graph):
    assert graph.floor_z("A") == 0.0 and graph.floor_z("C") == pytest.approx(3.2)
    assert graph.door_floor_z(graph.doors["door_AS"]) == 0.0
    assert graph.door_floor_z(graph.doors["door_SC"]) == pytest.approx(3.2)


def test_graph_validates(graph):
    assert validate_graph(graph, H, 0.25).ok


def test_route_climbs_to_the_upper_floor(graph):
    goal = parse_goal("A방에서 C방으로 가줘", graph)
    plan = plan_route(graph, goal.start, goal.goal, height=H)
    assert plan.rooms == ["A", "S", "C"]
    assert plan.waypoints[0][2] == pytest.approx(H)  # 0.5 m above floor 1
    assert plan.waypoints[-1][2] == pytest.approx(3.2 + H)  # 0.5 m above floor 2
    zs = [p[2] for p in plan.waypoints]
    assert zs == sorted(zs)  # only climbs
    # it turns on the landing: around the end of the wall between the flights (x = 5.0)
    assert max(p[0] for p in plan.waypoints) > 5.0


@pytest.mark.parametrize("start,goal", list(itertools.permutations("ASC", 2)))
def test_every_pair_is_collision_free_and_ends_inside_the_goal(graph, start, goal):
    plan = plan_route(graph, start, goal, height=H)
    assert check_path(graph, plan.waypoints).ok
    assert graph.rooms[goal].contains(plan.waypoints[-1])


def test_stairwell_alias(graph):
    assert parse_goal("C방에서 계단으로", graph).goal == "S"


def test_single_floor_heights_unchanged():
    # floors at z = 0: height above the floor is the old absolute height
    g = build_fake_graph()
    assert all(p[2] == pytest.approx(H) for p in plan_route(g, "A", "C", height=H).keypoints)


def test_simulated_flight_up_the_stairs(graph):
    pytest.importorskip("mujoco")
    import numpy as np

    from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend
    from ddsg_nav.flight.executor import follow_path
    from ddsg_nav.planner.collision import check_trajectory

    plan = plan_route(graph, "A", "C", height=H)
    start = plan.waypoints[0]
    with MuJoCoBackend(graph, (start[0], start[1], 0.0)) as drone:
        follow_path(drone, plan.waypoints, velocity=0.3)
        traj = np.array(drone.trajectory)[:, 1:]
        assert drone.collisions == 0
    assert traj[:, 2].max() > 3.2  # it really reached floor 2
    # airborne part only: landing ends on the floor-2 slab, which is an obstacle voxel
    end = np.asarray(plan.waypoints[-1])
    airborne = traj[: np.flatnonzero(np.linalg.norm(traj - end, axis=1) < 0.1)[0] + 1]
    report = check_trajectory(graph, airborne, inflation_m=0.15)  # up to 10 cm tracking error
    assert report.ok, report.summary()
