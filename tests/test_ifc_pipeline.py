"""IFC pipeline on a synthetic building: convert, validate, plan, simulate, bad and messy input.

Skipped without ifcopenshell (`pip install -e ".[ifc]"`). The IFC is built at run time
by tests/ifc_fixture.py; nothing binary is committed.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ifcopenshell")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ifc_pipeline  # noqa: E402
from ifc_fixture import (  # noqa: E402
    BAD_SPACES, DOORS, EMPTY_STOREY, ORIGIN, ROOMS, SLAB_VOID, STAIR, STAIR_RISE, STOREY, WALLS, build_ifc, world,
)
from ifc_to_graph import (  # noqa: E402
    FLOOR_SKIP_M, ConversionError, convert, convert_with_report, list_storeys,
)

from ddsg_nav.graph.io import load_graph  # noqa: E402
from ddsg_nav.graph.model import VoxelState  # noqa: E402
from ddsg_nav.planner.collision import check_path  # noqa: E402
from ddsg_nav.planner.route import plan_route  # noqa: E402
from ddsg_nav.validate import validate_graph  # noqa: E402

OX, OY = ORIGIN


@pytest.fixture(scope="module")
def ifc(tmp_path_factory):
    return str(build_ifc(tmp_path_factory.mktemp("ifc") / "building.ifc"))


@pytest.fixture(scope="module")
def converted(ifc):
    return convert(ifc, STOREY)


@pytest.fixture(scope="module")
def graph(converted):
    return converted[0]


def state(graph, x, y, z):
    """Voxel state at storey coordinates (x, y), z above the floor."""
    return graph.voxels.state_at((OX + x, OY + y, z))


# ---- convert ---------------------------------------------------------------

def test_list_storeys(ifc):
    assert list_storeys(ifc) == [(EMPTY_STOREY, pytest.approx(3.6), 0), (STOREY, pytest.approx(7.2), 5)]


def test_rooms(graph):
    assert sorted(graph.rooms) == sorted(ROOMS)
    for rid, (name, x0, y0, x1, y1) in ROOMS.items():
        room = graph.rooms[rid]
        assert room.name == name
        assert room.bounds.min == pytest.approx((OX + x0, OY + y0, 0.0))  # z = 0 at the storey level
        assert room.bounds.max == pytest.approx((OX + x1, OY + y1, 2.6))
        assert graph.find_room(name) is room


def test_doors_and_open_connection(converted):
    graph, notes = converted
    assert sorted(graph.doors) == ["door_201_203", "door_202_203", "door_204_205", "open_203_205"]
    for name, (wall, x0, x1, rooms) in DOORS.items():
        if len(rooms) == 2:
            door = graph.doors[f"door_{rooms[0]}_{rooms[1]}"]
            assert door.position[:2] == pytest.approx((OX + (x0 + x1) / 2, OY + 2.1))
            assert door.width == pytest.approx(0.9)
    opening = graph.doors["open_203_205"]  # two spaces that touch without a wall
    assert opening.position[:2] == pytest.approx((OX + 6.0, OY + 1.0))
    assert opening.width == pytest.approx(2.0, abs=0.05)
    assert any("D4" in n and "outside" in n for n in notes)  # the exterior door is reported, not linked


def test_voxels(graph):
    for name, x0, y0, x1, y1 in ROOMS.values():
        assert state(graph, (x0 + x1) / 2, y0 + 0.3, 0.5) == VoxelState.FREE, name
        assert state(graph, (x0 + x1) / 2, y0 + 0.3, 0.05) == VoxelState.FREE, name  # just above the slab
    for name, (x0, y0, x1, y1) in WALLS.items():
        assert state(graph, x0 + 0.05, y0 + 0.05, 1.0) == VoxelState.OCCUPIED, name
    assert state(graph, 7.5, 1.0, 1.0) == VoxelState.OCCUPIED  # column
    assert state(graph, 4.0, 2.0, -0.1) == VoxelState.OCCUPIED  # slab
    for wall, x0, x1, rooms in DOORS.values():
        if len(rooms) == 2:  # the whole opening is free, the wall above it is not
            assert state(graph, x0 + 0.05, 2.1, 1.0) == VoxelState.FREE
            assert state(graph, x1 - 0.05, 2.1, 1.0) == VoxelState.FREE
            assert state(graph, (x0 + x1) / 2, 2.1, 2.3) == VoxelState.OCCUPIED


def test_walls_keep_their_thickness(graph):
    """A face on a voxel boundary must not mark the voxel on its far side."""
    assert state(graph, 0.05, 1.0, 1.0) == VoxelState.FREE  # next to the west wall
    assert state(graph, 2.85, 4.0, 1.0) == VoxelState.FREE and state(graph, 3.15, 4.0, 1.0) == VoxelState.FREE
    assert state(graph, 7.25, 1.0, 1.0) == VoxelState.FREE and state(graph, 7.75, 1.0, 1.0) == VoxelState.FREE


def test_solids_cover_the_occupied_voxels(graph):
    """The simulator's walls are exactly the occupied voxels from FLOOR_SKIP_M up."""
    grid = graph.voxels
    k = grid.index((0.0, 0.0, FLOOR_SKIP_M))[2]
    covered = np.zeros(grid.shape, dtype=bool)
    for box in graph.solids:
        covered[grid.box_slices(box)] = True
    occupied = grid.states == int(VoxelState.OCCUPIED)
    occupied[:, :, :k] = False
    assert np.array_equal(covered, occupied)
    assert len(graph.solids) < 50  # straight walls are merged, not one box per voxel


def test_millimeter_file_converts_to_meters(tmp_path, graph):
    g, notes = convert(str(build_ifc(tmp_path / "mm.ifc", unit="mm")), STOREY)
    assert any("converted to meters" in n for n in notes)
    assert g.doors.keys() == graph.doors.keys()
    for rid, room in graph.rooms.items():
        assert g.rooms[rid].bounds.min == pytest.approx(room.bounds.min)
        assert g.rooms[rid].bounds.max == pytest.approx(room.bounds.max)
    assert np.array_equal(g.voxels.states, graph.voxels.states)


# ---- validate, plan --------------------------------------------------------

def test_validate(graph):
    report = validate_graph(graph)
    assert report.ok and not report.warnings, report.format()


def test_plan(graph):
    plan = plan_route(graph, "Office", "Lab")
    assert plan.rooms == ["201", "203", "205", "204"]
    assert plan.doors == ["door_201_203", "open_203_205", "door_204_205"]
    assert check_path(graph, plan.waypoints).ok


def test_start_point_avoids_the_column(graph):
    """The column stands at the center of 205, so the route starts beside it."""
    plan = plan_route(graph, "205", "204")
    assert math.dist(plan.waypoints[0][:2], graph.rooms["205"].center[:2]) > 0.3
    assert check_path(graph, plan.waypoints).ok


def test_pick_pairs(graph):
    pairs = ifc_pipeline.pick_pairs(graph, 3)
    assert len(pairs) == 2  # five rooms, each used once
    assert len({r for p in pairs for r in p}) == 4
    assert ifc_pipeline.pick_pairs(graph, 1) == pairs[:1]


# ---- simulate, whole pipeline ----------------------------------------------

def test_pipeline_passes(ifc, tmp_path):
    pytest.importorskip("mujoco")
    result = ifc_pipeline.run(ifc, STOREY, tmp_path / "graph")
    assert result.ok, result.failures
    assert len(result.graph.rooms) == 5 and len(result.graph.doors) == 4
    assert len(result.routes) == 2 and all(r.flown for r in result.routes)
    assert result.wall_contacts == 0
    assert all(len(r.rooms) >= 3 and r.length_m > 4 for r in result.routes)
    saved = load_graph(result.out)  # the saved folder is a complete graph export
    assert saved.rooms == result.graph.rooms and saved.solids == result.graph.solids
    report = ifc_pipeline.format_report(result)
    assert "rooms: 5" in report and "wall contacts: 0" in report and "RESULT: PASS" in report


def test_pipeline_given_pair(ifc, tmp_path):
    pytest.importorskip("mujoco")
    result = ifc_pipeline.run(ifc, STOREY, tmp_path / "graph", start="Office", goal="Lab", velocity=0.5)
    assert result.ok, result.failures
    assert [r.rooms for r in result.routes] == [["201", "203", "205", "204"]]


def test_simulator_has_the_walls(graph):
    """Flying straight at a wall is counted, so 0 wall contacts means something."""
    pytest.importorskip("mujoco")
    from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend, SimTimeoutError
    from ddsg_nav.flight.executor import follow_path

    a, b = graph.rooms["201"].center, graph.rooms["202"].center
    sim = MuJoCoBackend(graph, (a[0], a[1], 0.0))
    try:
        follow_path(sim, [(a[0], a[1], 0.5), (b[0], b[1], 0.5)], velocity=0.5)
    except SimTimeoutError:
        pass
    assert sim.collisions > 0
    assert sim.position()[0] < OX + 2.9  # stopped by the partition wall


def _cli(script, *args, cwd):
    """Run a script; UTF-8 output, so a non-ASCII path does not break on a legacy code page."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", cwd=cwd, env=env, timeout=600)


def test_command_line(ifc, tmp_path):
    pytest.importorskip("mujoco")
    script = str(ROOT / "scripts" / "ifc_pipeline.py")
    listing = _cli(script, ifc, cwd=tmp_path)
    assert listing.returncode == 0 and STOREY in listing.stdout and "5 spaces" in listing.stdout
    assert not (tmp_path / "data").exists()  # listing writes nothing
    done = _cli(script, ifc, "--storey", STOREY, cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "wall contacts: 0" in done.stdout and "RESULT: PASS" in done.stdout
    assert (tmp_path / "data" / STOREY / "graph.json").is_file()  # default --out


# ---- messy input that still converts -----------------------------------------

def test_doors_without_space_boundaries_are_matched_by_geometry(tmp_path, graph):
    path = str(build_ifc(tmp_path / "x.ifc", space_boundaries=False))
    g, notes, report = convert_with_report(path, STOREY)
    assert g.doors == graph.doors  # same rooms, positions and widths as with boundaries
    assert np.array_equal(g.voxels.states, graph.voxels.states)
    assert (report.doors, report.doors_without_boundary) == (4, 4)
    assert (report.doors_by_geometry, report.doors_by_boundary) == (3, 0)
    assert any("4 of 4 doors have no space boundary" in n and "3 were matched" in n for n in notes)
    assert any("D4" in n and "outside" in n and "from geometry" in n for n in notes)  # exterior door
    assert validate_graph(g).ok


def test_pipeline_without_space_boundaries(tmp_path):
    pytest.importorskip("mujoco")
    path = str(build_ifc(tmp_path / "x.ifc", space_boundaries=False))
    result = ifc_pipeline.run(path, STOREY, tmp_path / "graph", start="Office", goal="Lab", velocity=0.5)
    assert result.ok, result.failures
    assert result.wall_contacts == 0 and result.routes[0].rooms == ["201", "203", "205", "204"]


def test_spaces_without_usable_geometry_are_skipped(tmp_path, graph):
    path = str(build_ifc(tmp_path / "x.ifc", bad_spaces=True))
    assert list_storeys(path)[-1] == (STOREY, pytest.approx(7.2), 7)
    g, notes, report = convert_with_report(path, STOREY)
    assert sorted(g.rooms) == sorted(ROOMS) and g.doors == graph.doors
    assert np.array_equal(g.voxels.states, graph.voxels.states)  # the bad spaces do not move the grid
    assert (report.spaces, report.rooms) == (7, 5)
    assert dict(report.skipped_spaces) == {BAD_SPACES[0]: "no 3D geometry", BAD_SPACES[1]: "floor area below 0.1 m2"}
    for rid in BAD_SPACES:
        assert any(n.startswith(f"warning: space {rid} skipped") for n in notes)


@pytest.mark.parametrize("origin,rotation", [(ORIGIN, 30.0), ((352_000.0, 4_151_000.0), 0.0),
                                             ((352_000.0, 4_151_000.0), -75.0)])
def test_rotated_building_and_large_offset_keep_the_ifc_frame(tmp_path, origin, rotation):
    """x, y stay as in the IFC world frame: no shift to the origin, no un-rotation."""
    g, _ = convert(str(build_ifc(tmp_path / "x.ifc", origin=origin, rotation_deg=rotation)), STOREY)
    assert sorted(g.rooms) == sorted(ROOMS)
    assert sorted(g.doors) == ["door_201_203", "door_202_203", "door_204_205", "open_203_205"]
    for rid, (name, x0, y0, x1, y1) in ROOMS.items():
        room = g.rooms[rid]
        corners = [world(x, y, origin, rotation) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        assert len(room.polygon) == 4
        for corner in corners:  # the footprint is the rotated rectangle, in world coordinates
            assert min(math.dist(corner, p) for p in room.polygon) < 2e-3
        assert room.bounds.min[:2] == pytest.approx((min(c[0] for c in corners), min(c[1] for c in corners)), abs=2e-3)
        assert (room.bounds.min[2], room.bounds.max[2]) == pytest.approx((0.0, 2.6))  # z = 0 at the storey
        cx, cy = world((x0 + x1) / 2, y0 + 0.5, origin, rotation)  # off the center: a column stands in 205
        assert room.contains((cx, cy, 0.5)) and g.voxels.state_at((cx, cy, 0.5)) == VoxelState.FREE
    for name, (wall, x0, x1, rooms) in DOORS.items():
        if len(rooms) == 2:
            door = g.doors[f"door_{rooms[0]}_{rooms[1]}"]
            assert door.position[:2] == pytest.approx(world((x0 + x1) / 2, 2.1, origin, rotation), abs=2e-3)
    wx, wy = world(4.5, 5.3, origin, rotation)  # inside the north wall
    assert g.voxels.state_at((wx, wy, 1.0)) == VoxelState.OCCUPIED
    assert validate_graph(g).ok, validate_graph(g).format()
    plan = plan_route(g, "Office", "Lab")
    assert plan.rooms == ["201", "203", "205", "204"]
    assert check_path(g, plan.waypoints).ok


def test_rotated_building_flies_without_wall_contact(tmp_path):
    pytest.importorskip("mujoco")
    path = str(build_ifc(tmp_path / "x.ifc", rotation_deg=30.0))
    result = ifc_pipeline.run(path, STOREY, tmp_path / "graph", start="Office", goal="Lab", velocity=0.5)
    assert result.ok, result.failures
    assert result.wall_contacts == 0 and result.routes[0].flown


def test_stair_is_blocked_and_slab_opening_is_not_flyable(tmp_path, graph):
    g, notes, report = convert_with_report(str(build_ifc(tmp_path / "x.ifc", stair_and_void=True)), STOREY)
    x0, y0, x1, y1 = STAIR
    tread = (x1 - x0) / 3
    for i in range(3):  # each step is occupied up to its top and free above it
        x, top = x0 + (i + 0.5) * tread, (i + 1) * STAIR_RISE
        assert state(g, x, (y0 + y1) / 2, top - 0.05) == VoxelState.OCCUPIED
        assert state(g, x, (y0 + y1) / 2, top + 0.05) == VoxelState.FREE
    vx0, vy0, vx1, vy1 = SLAB_VOID
    for z in (-0.1, 0.05, 0.5, 2.0):  # no floor: the whole column is unknown, not flyable
        assert state(g, (vx0 + vx1) / 2, (vy0 + vy1) / 2, z) == VoxelState.UNKNOWN
    assert state(g, vx0 - 0.15, (vy0 + vy1) / 2, 0.5) == VoxelState.FREE  # floor beside the hole
    assert report.voids == [("202", pytest.approx(1.0))]
    assert any("room 202" in n and "without a floor" in n and "not flyable" in n for n in notes)
    assert report.elements["IfcStair"] == 3
    different = g.voxels.states != graph.voxels.states  # nothing else changed
    lo, hi = g.voxels.index((OX + 3.1, OY + 2.2, -0.4)), g.voxels.index((OX + 6.0, OY + 5.2, 2.7))
    different[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, :] = False
    assert not different.any()
    assert validate_graph(g).ok, validate_graph(g).format()
    plan = plan_route(g, "Office", "Meeting")  # into the room with the stair and the hole
    assert check_path(g, plan.waypoints).ok


def test_pipeline_with_stair_and_void(tmp_path):
    pytest.importorskip("mujoco")
    path = str(build_ifc(tmp_path / "x.ifc", stair_and_void=True))
    result = ifc_pipeline.run(path, STOREY, tmp_path / "graph", start="Office", goal="Meeting", velocity=0.5)
    assert result.ok, result.failures
    assert result.wall_contacts == 0


def test_report(tmp_path):
    path = str(build_ifc(tmp_path / "x.ifc", space_boundaries=False, bad_spaces=True, stair_and_void=True))
    _, _, report = convert_with_report(path, STOREY)
    assert report.elements == {"IfcSlab": 1, "IfcColumn": 1, "IfcWall": 7, "IfcStair": 3}
    assert not report.skipped_elements and report.open_connections == 1
    text = report.format()
    for line in ("storey: 3F (elevation 7.20 m, length unit 1 m)", "obstacle elements: 12", "  IfcWall: 7",
                 "spaces: 7, rooms: 5, skipped: 2", "  206: no 3D geometry",
                 "doors: 4, connected by space boundary: 0, by geometry: 3 (of 4 without a boundary), skipped: 1",
                 "open connections (no wall): 1", "  202: 1.0 m2 not flyable"):
        assert line in text.splitlines(), line
    assert "D4" in text and "outside" in text


def test_report_command_line(ifc, tmp_path):
    script = ROOT / "scripts" / "ifc_to_graph.py"
    done = _cli(script, ifc, str(tmp_path / "g"), "--storey", STOREY, "--report", cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--- conversion report ---" in done.stdout and "  IfcWall: 7" in done.stdout
    assert "connected by space boundary: 3, by geometry: 0" in done.stdout
    assert load_graph(tmp_path / "g").rooms.keys() == set(ROOMS)
    plain = _cli(script, ifc, str(tmp_path / "g2"), "--storey", STOREY, cwd=tmp_path)
    assert plain.returncode == 0 and "conversion report" not in plain.stdout  # only on request


# ---- bad input: what is wrong, what to do ------------------------------------

def test_unknown_storey(ifc):
    with pytest.raises(ConversionError, match=r"'9F' is not in the file.*--storey: 2F \(0 spaces\), 3F \(5 spaces\)"):
        convert(ifc, "9F")


def test_storey_without_spaces(ifc):
    with pytest.raises(ConversionError, match=r"has no IfcSpace.*3F \(5 spaces\)"):
        convert(ifc, EMPTY_STOREY)


@pytest.mark.parametrize("unit,declared", [("mm", "m"), ("m", "mm")])
def test_wrong_unit(tmp_path, unit, declared):
    path = str(build_ifc(tmp_path / "x.ifc", unit=unit, declared_unit=declared))
    with pytest.raises(ConversionError, match=r"not plausible in meters.*Fix the project units"):
        convert(path, STOREY)


def test_missing_or_broken_file(tmp_path):
    with pytest.raises(ConversionError, match="not found"):
        convert(str(tmp_path / "none.ifc"), STOREY)
    (tmp_path / "broken.ifc").write_text("not an IFC file")
    with pytest.raises(ConversionError, match="cannot be read as IFC"):
        list_storeys(str(tmp_path / "broken.ifc"))


def test_pipeline_reports_a_failed_conversion(ifc, tmp_path, capsys):
    result = ifc_pipeline.run(ifc, "9F", tmp_path / "graph")
    assert not result.ok and result.failures[0].startswith("convert: storey '9F'")
    assert not (tmp_path / "graph").exists()
    assert ifc_pipeline.main([ifc, "--storey", "9F", "--out", str(tmp_path / "graph")]) == 1
    assert "FAILED convert: storey '9F'" in capsys.readouterr().out


def test_room_without_a_door(tmp_path):
    """A room no door leads to is reported; the other rooms still pass."""
    pytest.importorskip("mujoco")
    path = str(build_ifc(tmp_path / "x.ifc", lab_door=False))
    result = ifc_pipeline.run(path, STOREY, tmp_path / "graph", routes=1, velocity=0.5)
    assert result.ok, result.failures
    assert result.doorless == ["204"]
    assert "rooms without a door: 204" in ifc_pipeline.format_report(result)

    asked = ifc_pipeline.run(path, STOREY, tmp_path / "graph", start="201", goal="204")
    assert not asked.ok and "no route from 201 to 204" in asked.failures[0]
    unknown = ifc_pipeline.run(path, STOREY, tmp_path / "graph", start="201", goal="999")
    assert not unknown.ok and "unknown room '999'" in unknown.failures[0] and "201, 202" in unknown.failures[0]
