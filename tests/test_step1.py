import numpy as np
import pytest

from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.graph.model import Box, VoxelGrid, VoxelState
from ddsg_nav.planner.global_planner import NoRoomRouteError, plan_rooms
from ddsg_nav.planner.local_planner import (
    NoPathError,
    blocked_mask,
    inflate,
    plan_voxel_path,
)
from ddsg_nav.planner.route import plan_route


@pytest.fixture(scope="module")
def graph():
    return build_fake_graph()


# ---- graph -----------------------------------------------------------------

def test_fake_graph_structure(graph):
    assert set(graph.rooms) == {"A", "B", "C"}
    assert len(graph.doors) == 2
    g = graph.room_graph()
    assert {frozenset(e) for e in g.edges} == {frozenset("AB"), frozenset("BC")}


def test_voxel_walls_and_doors(graph):
    v = graph.voxels
    assert v.state_at((1.5, 1.5, 0.5)) == VoxelState.FREE  # room A center
    assert v.state_at((3.0, 0.5, 0.5)) == VoxelState.OCCUPIED  # partition wall
    assert v.state_at((3.0, 1.5, 0.5)) == VoxelState.FREE  # door opening
    assert v.state_at((3.0, 1.5, 2.2)) == VoxelState.OCCUPIED  # lintel above the door
    assert v.state_at((-0.05, 1.5, 0.5)) == VoxelState.OCCUPIED  # outer wall


def test_find_room_by_name_and_id(graph):
    assert graph.find_room("room b").id == "B"
    assert graph.find_room("C").id == "C"
    with pytest.raises(KeyError):
        graph.find_room("Z")


# ---- goal parsing ----------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "go from A to C",
        "from Room A to Room C",
        "A -> C",
        "A→C",
        "A방에서 C방으로",
        "A방부터 C방까지 가줘",
        "to C from A",
        "A C",
    ],
)
def test_parse_goal_a_to_c(graph, text):
    goal = parse_goal(text, graph)
    assert (goal.start, goal.goal) == ("A", "C")


def test_parse_goal_current_room(graph):
    goal = parse_goal("C방으로 가줘", graph, current_room="B")
    assert (goal.start, goal.goal) == ("B", "C")


def test_parse_goal_errors(graph):
    with pytest.raises(GoalError):
        parse_goal("hello there", graph)
    with pytest.raises(GoalError):
        parse_goal("go to C", graph)  # no start, no current_room
    with pytest.raises(GoalError):
        parse_goal("from A to A", graph)


# ---- global planner --------------------------------------------------------

def test_global_route_through_b(graph):
    route = plan_rooms(graph, "A", "C")
    assert route.rooms == ["A", "B", "C"]
    assert route.doors == ["door_AB", "door_BC"]


def test_global_route_same_side(graph):
    assert plan_rooms(graph, "C", "B").rooms == ["C", "B"]


def test_global_no_route():
    g = build_fake_graph()
    g.doors.pop("door_BC")
    with pytest.raises(NoRoomRouteError):
        plan_rooms(g, "A", "C")


# ---- local planner ---------------------------------------------------------

def test_inflate_sphere():
    m = np.zeros((9, 9, 9), dtype=bool)
    m[4, 4, 4] = True
    out = inflate(m, 2)
    assert out[4, 4, 6] and out[6, 4, 4] and not out[6, 6, 4]
    assert not inflate(m, 0)[4, 4, 5]


def _open_grid(unknown=False):
    state = VoxelState.UNKNOWN if unknown else VoxelState.FREE
    return VoxelGrid.empty((0, 0, 0), (3, 3, 1), 0.1, state)


def test_local_straight_line_free_space():
    grid = _open_grid()
    path = plan_voxel_path(grid, (0.5, 0.5, 0.5), (2.5, 2.5, 0.5), inflation_m=0.1)
    assert path[0] == (0.5, 0.5, 0.5) and path[-1] == (2.5, 2.5, 0.5)
    assert len(path) == 2  # simplified to a single segment


def test_local_avoids_obstacle_with_margin():
    grid = _open_grid()
    wall = Box((1.4, 0.0, 0.0), (1.6, 2.0, 1.0))  # leaves a gap at y > 2
    grid.fill_box(wall, VoxelState.OCCUPIED)
    path = plan_voxel_path(grid, (0.5, 0.5, 0.5), (2.5, 0.5, 0.5), inflation_m=0.25)
    assert any(p[1] > 2.2 for p in path)  # detours through the gap, with margin
    blocked = blocked_mask(grid, 0.25)
    for p in path:
        assert not blocked[grid.index(p)]


def test_local_unknown_is_obstacle():
    grid = _open_grid()
    grid.fill_box(Box((1.4, 0.0, 0.0), (1.6, 3.0, 1.0)), VoxelState.UNKNOWN)
    with pytest.raises(NoPathError):
        plan_voxel_path(grid, (0.5, 1.5, 0.5), (2.5, 1.5, 0.5), inflation_m=0.1)
    path = plan_voxel_path(
        grid, (0.5, 1.5, 0.5), (2.5, 1.5, 0.5), inflation_m=0.1, unknown_is_obstacle=False
    )
    assert path[-1] == (2.5, 1.5, 0.5)


def test_local_start_inside_obstacle_raises():
    grid = _open_grid()
    grid.fill_box(Box((0.4, 0.4, 0.0), (0.6, 0.6, 1.0)), VoxelState.OCCUPIED)
    with pytest.raises(NoPathError):
        plan_voxel_path(grid, (0.5, 0.5, 0.5), (2.5, 2.5, 0.5))


# ---- full route ------------------------------------------------------------

def _assert_route_safe(graph, plan, inflation=0.25):
    blocked = blocked_mask(graph.voxels, inflation)
    # sample along every segment
    for a, b in zip(plan.waypoints, plan.waypoints[1:]):
        n = max(1, int(np.linalg.norm(np.subtract(b, a)) / 0.05))
        for i in range(n + 1):
            p = tuple(np.add(a, np.subtract(b, a) * (i / n)))
            assert not blocked[graph.voxels.index(p)], f"collision margin violated at {p}"


def test_route_a_to_c(graph):
    plan = plan_route(graph, "A", "C", height=0.5)
    assert plan.rooms == ["A", "B", "C"]
    assert plan.waypoints[0] == (1.5, 1.5, 0.5)
    assert plan.waypoints[-1] == (7.5, 1.5, 0.5)
    xs = [p[0] for p in plan.waypoints]
    assert xs == sorted(xs)  # moves monotonically toward C
    _assert_route_safe(graph, plan)


def test_route_with_obstacle_detours():
    g = build_fake_graph(obstacle=True)
    plan = plan_route(g, "A", "C", height=0.5)
    ys = [p[1] for p in plan.waypoints]
    zs = [p[2] for p in plan.waypoints]
    assert max(ys) > 1.9 or min(ys) < 1.1 or max(zs) > 1.2  # goes around or over the box
    _assert_route_safe(g, plan)
