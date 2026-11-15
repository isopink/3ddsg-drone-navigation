"""Exchange format, validation, frames, unknown-voxel policies, polygon rooms, aliases."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_fake_graph, build_large_graph
from ddsg_nav.graph.io import GraphFormatError, load_graph, save_graph
from ddsg_nav.graph.model import (
    Box,
    Door,
    Room,
    SceneGraph,
    VoxelGrid,
    VoxelState,
    points_in_polygon,
    polygon_centroid,
)
from ddsg_nav.graph.source import FakeGraphSource, FileGraphSource
from ddsg_nav.graph.transform import Transform
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.local_planner import NoPathError
from ddsg_nav.planner.policy import ElementPolicy, UnknownPolicy, raw_obstacles
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

ROOT = Path(__file__).resolve().parents[1]


# ---- exchange format -------------------------------------------------------

def test_io_roundtrip(tmp_path):
    g = build_fake_graph(obstacle=True)
    g.rooms["A"] = Room(
        "A", "Room A", g.rooms["A"].bounds, aliases=("401호", "2O2Fr$t4X7"),
        polygon=((0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)),
    )
    save_graph(g, tmp_path / "x")
    h = load_graph(tmp_path / "x")
    assert h.rooms == g.rooms and h.doors == g.doors and h.solids == g.solids
    assert h.voxels.origin == g.voxels.origin and h.voxels.resolution == g.voxels.resolution
    assert np.array_equal(h.voxels.states, g.voxels.states)
    assert h.rooms["A"].aliases == ("401호", "2O2Fr$t4X7")  # non-ASCII survives


def test_io_errors(tmp_path):
    with pytest.raises(GraphFormatError):
        load_graph(tmp_path / "missing")
    save_graph(build_fake_graph(), tmp_path / "g")
    doc = json.loads((tmp_path / "g" / "graph.json").read_text(encoding="utf-8"))

    def write(d):
        (tmp_path / "g" / "graph.json").write_text(json.dumps(d), encoding="utf-8")

    write({**doc, "version": 99})
    with pytest.raises(GraphFormatError, match="version"):
        load_graph(tmp_path / "g")
    write({**doc, "format": "other"})
    with pytest.raises(GraphFormatError, match="not a"):
        load_graph(tmp_path / "g")
    write({**doc, "voxels": {**doc["voxels"], "shape": [1, 2, 3]}})
    with pytest.raises(GraphFormatError, match="shape"):
        load_graph(tmp_path / "g")
    bad = {k: v for k, v in doc.items() if k != "rooms"}
    write(bad)
    with pytest.raises(GraphFormatError, match="missing field"):
        load_graph(tmp_path / "g")


def test_graph_sources(tmp_path):
    save_graph(build_fake_graph(), tmp_path / "s")
    a, b = FakeGraphSource().load(), FileGraphSource(tmp_path / "s").load()
    assert set(a.rooms) == set(b.rooms) and len(a.doors) == len(b.doors)
    assert FakeGraphSource(obstacle=True).load().solids != a.solids


# ---- frames ----------------------------------------------------------------

def test_transform_roundtrip_and_yaw():
    t = Transform(translation=(2.0, 3.0, 0.0), yaw_deg=90.0)
    x, y, z = t.to_graph((1.0, 0.0, 0.5))
    assert (round(x, 9), round(y, 9), z) == (2.0, 4.0, 0.5)  # nose along +y in the graph
    p = (1.3, -0.7, 0.4)
    q = t.to_drone(t.to_graph(p))
    assert np.allclose(q, p)
    assert Transform().to_graph((1, 2, 3)) == (1.0, 2.0, 3.0)


# ---- polygon rooms and aliases --------------------------------------------

L_SHAPE = ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0))


def test_polygon_helpers():
    inside = points_in_polygon(np.array([1.0, 3.0, 3.0]), np.array([1.0, 1.0, 3.0]), L_SHAPE)
    assert inside.tolist() == [True, True, False]  # (3, 3) is in the notch
    assert polygon_centroid(((0, 0), (2, 0), (2, 2), (0, 2))) == (1.0, 1.0)


def test_l_shaped_room():
    grid = VoxelGrid.empty((0, 0, 0), (4, 4, 1), 0.1, VoxelState.FREE)
    g = SceneGraph(voxels=grid)
    room = Room("L", "L room", Box((0, 0, 0), (4, 4, 1)), polygon=L_SHAPE)
    g.add_room(room)
    assert g.room_at((1.0, 3.0, 0.5)).id == "L"
    assert g.room_at((3.0, 3.0, 0.5)) is None  # inside the bounding box but outside the polygon
    mask = g.room_mask(room)
    assert mask[grid.index((1.0, 3.0, 0.5))] and not mask[grid.index((3.0, 3.0, 0.5))]
    assert 0.55 < mask.mean() < 0.95  # about 3/4 of the box


def test_aliases_resolve_rooms_and_goals():
    g = build_fake_graph()
    g.rooms["A"] = Room("A", "Room A", g.rooms["A"].bounds, aliases=("401호", "GUID-0xA1"))
    g.rooms["C"] = Room("C", "Room C", g.rooms["C"].bounds, aliases=("302호",))
    assert g.find_room("guid-0xa1").id == "A"
    goal = parse_goal("302호로 가줘 401호에서", g)
    assert (goal.start, goal.goal) == ("A", "C")


# ---- unknown-voxel policies ------------------------------------------------

def _graph_with_unknown_in_room_b():
    g = build_fake_graph()
    g.voxels.fill_box(Box((3.3, 0.2, 0.0), (5.7, 2.8, 2.5)), VoxelState.UNKNOWN)  # whole interior of B
    return g


def test_policy_coerce():
    assert UnknownPolicy.coerce(True) is UnknownPolicy.OBSTACLE
    assert UnknownPolicy.coerce(False) is UnknownPolicy.FREE
    assert UnknownPolicy.coerce("free_in_rooms") is UnknownPolicy.FREE_IN_ROOMS


def test_unknown_policies_change_what_is_blocked():
    g = _graph_with_unknown_in_room_b()
    p = (4.5, 1.5, 0.5)
    idx = g.voxels.index(p)
    assert raw_obstacles(g, UnknownPolicy.OBSTACLE)[idx]
    assert not raw_obstacles(g, UnknownPolicy.FREE)[idx]
    assert not raw_obstacles(g, UnknownPolicy.FREE_IN_ROOMS)[idx]  # unknown but inside a room


def test_route_fails_on_unknown_room_unless_policy_allows():
    g = _graph_with_unknown_in_room_b()
    with pytest.raises(NoPathError):
        plan_route(g, "A", "C", unknown_is_obstacle=UnknownPolicy.OBSTACLE)
    plan = plan_route(g, "A", "C", unknown_is_obstacle=UnknownPolicy.FREE_IN_ROOMS)
    assert plan.rooms == ["A", "B", "C"]
    assert check_path(g, plan.waypoints, unknown_is_obstacle=UnknownPolicy.FREE_IN_ROOMS).ok
    assert not check_path(g, plan.waypoints, unknown_is_obstacle=UnknownPolicy.OBSTACLE).ok


def test_free_in_rooms_keeps_unknown_outside_rooms_blocked():
    g = build_fake_graph()
    # unknown column in the open space north of the rooms (not inside any room)
    g.voxels.fill_box(Box((1.0, 3.0, 0.0), (2.0, 3.1, 2.5)), VoxelState.UNKNOWN)
    mask = raw_obstacles(g, UnknownPolicy.FREE_IN_ROOMS)
    assert mask[g.voxels.index((1.5, 3.05, 0.5))]


def _graph_with_unknown_room_b_and_doors():
    g = _graph_with_unknown_in_room_b()
    for x in (3.0, 6.0):  # both door openings of B
        g.voxels.fill_box(Box((x - 0.1, 1.0, 0.0), (x + 0.1, 2.0, 2.0)), VoxelState.UNKNOWN)
    return g


def test_element_policy_default_and_all_free_match_global_policies():
    g = _graph_with_unknown_room_b_and_doors()
    assert np.array_equal(raw_obstacles(g, ElementPolicy()), raw_obstacles(g, UnknownPolicy.OBSTACLE))
    all_free = ElementPolicy(wall="free", door="free", furniture="free")
    assert np.array_equal(raw_obstacles(g, all_free), raw_obstacles(g, UnknownPolicy.FREE))
    with pytest.raises(ValueError):
        ElementPolicy(door="free_in_rooms")


def test_element_policy_sets_each_type_separately():
    g = _graph_with_unknown_room_b_and_doors()
    g.voxels.fill_box(Box((1.0, 3.0, 0.0), (2.0, 3.1, 2.5)), VoxelState.UNKNOWN)  # in the north wall
    wall = g.voxels.index((1.5, 3.05, 0.5))
    door = g.voxels.index((3.05, 1.5, 0.5))
    furniture = g.voxels.index((4.5, 1.5, 0.5))
    for free in ("wall", "door", "furniture"):
        mask = raw_obstacles(g, ElementPolicy(**{free: "free"}))
        assert [bool(mask[i]) for i in (wall, door, furniture)] == [
            name != free for name in ("wall", "door", "furniture")
        ]


def test_route_needs_door_and_furniture_free():
    g = _graph_with_unknown_room_b_and_doors()
    for policy in (ElementPolicy(), ElementPolicy(door="free"), ElementPolicy(furniture="free")):
        with pytest.raises(NoPathError):
            plan_route(g, "A", "C", unknown_is_obstacle=policy)
    policy = ElementPolicy(door="free", furniture="free")
    plan = plan_route(g, "A", "C", unknown_is_obstacle=policy)
    assert plan.rooms == ["A", "B", "C"]
    assert check_path(g, plan.waypoints, unknown_is_obstacle=policy).ok
    assert not check_path(g, plan.waypoints).ok
    report = validate_graph(g, policy=policy)
    assert report.ok and "door=free" in report.format(), report.format()
    assert not validate_graph(g, policy=ElementPolicy(furniture="free")).ok


# ---- validation ------------------------------------------------------------

def test_validate_fake_graph_ok():
    r = validate_graph(build_fake_graph())
    assert r.ok and not r.warnings, r.format()
    assert r.stats["rooms"] == 3 and r.stats["doors"] == 2


def _codes(report):
    return {i.code for i in report.issues}


def test_validate_blocked_door():
    g = build_fake_graph()
    g.voxels.fill_box(Box((2.9, 1.0, 0.0), (3.1, 2.0, 2.0)), VoxelState.OCCUPIED)  # door filled in
    r = validate_graph(g)
    assert not r.ok and "door-blocked" in _codes(r)


def test_validate_unreachable_door_inside_a_room():
    g = build_fake_graph()
    g.voxels.fill_box(Box((2.0, 0.0, 0.0), (2.2, 3.0, 2.5)), VoxelState.OCCUPIED)  # wall across room A
    r = validate_graph(g)
    assert "door-unreachable" in _codes(r), r.format()


def test_validate_isolated_and_disconnected_rooms():
    g = build_fake_graph()
    g.doors.pop("door_BC")
    r = validate_graph(g)
    assert "isolated-room" in _codes(r) and "disconnected" in _codes(r)
    assert r.ok  # warnings only


def test_validate_mostly_unknown_and_room_outside_grid():
    g = build_fake_graph()
    g.voxels.states[:] = int(VoxelState.UNKNOWN)
    assert "mostly-unknown" in _codes(validate_graph(g, check_connectivity=False))
    g2 = build_fake_graph()
    g2.add_room(Room("Z", "Room Z", Box((50, 50, 0), (55, 55, 2.5))))
    r = validate_graph(g2, check_connectivity=False)
    assert "room-outside-grid" in _codes(r) and not r.ok


def test_validate_narrow_door_warning():
    g = build_fake_graph()
    g.doors["door_AB"] = Door("door_AB", "A", "B", (3.0, 1.5, 1.0), width=0.4)
    assert "door-narrower-than-margin" in _codes(validate_graph(g, check_connectivity=False))


def test_validate_flight_height_outside_grid():
    r = validate_graph(build_fake_graph(), flight_height=9.0)
    assert "height-outside-grid" in _codes(r) and not r.ok


# ---- larger building -------------------------------------------------------

def test_large_graph_routes_corner_to_corner():
    g = build_large_graph(rooms_x=3, rooms_y=2, resolution=0.2)
    assert len(g.rooms) == 6 and len(g.doors) == 7
    assert validate_graph(g).ok
    plan = plan_route(g, "R0_0", "R2_1", height=0.5)
    assert plan.rooms[0] == "R0_0" and plan.rooms[-1] == "R2_1"
    assert check_path(g, plan.waypoints).ok


def test_route_works_at_coarse_resolution():
    """0.2 m voxels must not close a 1 m door (the margin is exact, not rounded up)."""
    g = build_fake_graph(resolution=0.2)
    plan = plan_route(g, "A", "C", height=0.5)
    assert plan.rooms == ["A", "B", "C"]
    assert check_path(g, plan.waypoints).ok


# ---- command line tool -----------------------------------------------------

def _run(*args):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "graph_tools.py"), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env, timeout=300,
    )


def test_graph_tools_cli(tmp_path):
    folder = str(tmp_path / "fake")
    assert _run("export-fake", folder).returncode == 0
    v = _run("validate", folder)
    assert v.returncode == 0 and "RESULT: OK" in v.stdout, v.stdout + v.stderr
    r = _run("route", folder, "go from A to C")
    assert r.returncode == 0 and "A -> C" in r.stdout and "0 hard" in r.stdout, r.stdout + r.stderr
    bad = _run("route", folder, "go from A to Z")
    assert bad.returncode == 1 and "cannot plan" in bad.stdout
    assert _run("validate", str(tmp_path / "nothing")).returncode == 2
