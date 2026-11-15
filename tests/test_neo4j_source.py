"""Neo4j source: mapping, row -> graph building, voxel merging. No database: rows come
from a fake driver, generated from the fake graph."""
import importlib.util

import numpy as np
import pytest

from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.graph.model import VoxelGrid, VoxelState
from ddsg_nav.graph.neo4j_source import (
    GraphRows,
    Neo4jGraphSource,
    Neo4jMapping,
    Neo4jMappingError,
    build_graph,
    build_queries,
    fetch_rows,
    solids_from_voxels,
)
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

FLOOR = "4"


def make_rows(scale: float = 1.0) -> GraphRows:
    """The 3-room fake graph (floor 4) as database rows, plus a room on floor 5, a door
    to it, and a door to a room that does not exist. `scale` = stored units per meter."""
    g = build_fake_graph()

    def s(values):
        return [v * scale for v in values]

    rooms = [
        {"props": {"id": r.id, "name": r.name, "number": f"40{i}", "floor": 4,
                   "bbox_min": s(r.bounds.min), "bbox_max": s(r.bounds.max)}}
        for i, r in enumerate(g.rooms.values(), start=1)
    ]
    rooms.append({"props": {"id": "U", "name": "Upstairs", "floor": 5,
                            "bbox_min": s((0, 0, 3)), "bbox_max": s((3, 3, 5.5))}})
    doors = [
        {"props": {"id": d.id, "position": s(d.position), "width": d.width * scale},
         "rooms": [d.room_a, d.room_b]}
        for d in g.doors.values()
    ]
    doors.append({"props": {"id": "door_up", "position": s((1, 1, 3))}, "rooms": ["A", "U"]})
    doors.append({"props": {"id": "door_ghost", "position": s((0, 1, 1))}, "rooms": ["A", "X9"]})
    grid = g.voxels
    voxels = [
        {"props": {"position": s(grid.center(tuple(idx))), "size": grid.resolution * scale,
                   "state": "occupied"}}
        for idx in np.argwhere(grid.states == int(VoxelState.OCCUPIED))
    ]
    return GraphRows(rooms, doors, voxels)


class FakeDriver:
    """Stands in for `neo4j.Driver`: answers each query from prepared rows."""

    def __init__(self, rows: GraphRows, mapping: Neo4jMapping | None = None) -> None:
        self.rows, self.mapping = rows, mapping or Neo4jMapping()
        self.calls: list[tuple[str, dict]] = []
        self.session_kwargs: dict = {}

    def session(self, **kwargs):
        self.session_kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query: str, **params):
        self.calls.append((query, params))
        m = self.mapping
        if f"`{m.voxel_label}`" in query:
            return self.rows.voxels
        if f"`{m.door_label}`" in query:
            return self.rows.doors
        return self.rows.rooms


def assert_same_map(graph, reference) -> None:
    """Same occupancy as the reference fake graph, wherever the reference is defined."""
    ref = reference.voxels
    for idx in np.ndindex(*ref.shape):
        if idx[2] % 6 or idx[1] % 3:  # sample: every 6th layer, every 3rd row
            continue
        assert graph.voxels.state_at(ref.center(idx)) == ref.state_at(ref.center(idx)), idx


@pytest.fixture(scope="module")
def rows() -> GraphRows:
    return make_rows()


@pytest.fixture(scope="module")
def graph(rows):
    return build_graph(rows, floor=FLOOR)


# ---- fetch -----------------------------------------------------------------

def test_load_through_fake_driver(rows):
    driver = FakeDriver(rows)
    source = Neo4jGraphSource("bolt://nowhere", "neo4j", "secret", FLOOR,
                              driver=driver, database="building")
    g = source.load()
    assert set(g.rooms) == {"A", "B", "C"}
    assert set(g.doors) == {"door_AB", "door_BC"}
    assert len(driver.calls) == 3
    assert all(params == {"floor": FLOOR} for _, params in driver.calls)
    assert driver.session_kwargs == {"database": "building"}
    assert len(source.notes) == 3  # other-floor room, other-floor door, ghost door


def test_queries_follow_the_mapping():
    q = build_queries(Neo4jMapping())
    assert "`Space`" in q["rooms"] and "`CONNECTS`" in q["doors"] and "`HAS_VOXEL`" in q["voxels"]
    m = Neo4jMapping(room_label="IfcSpace", door_label="IfcDoor", door_room_rel="BOUNDS",
                     voxel_label="Vox", element_label=None, room_id="GlobalId",
                     rooms_query="MATCH (s:Room {storey: $floor}) RETURN properties(s) AS props")
    q = build_queries(m)
    assert q["rooms"] == m.rooms_query
    assert "`IfcDoor`" in q["doors"] and "`BOUNDS`" in q["doors"] and "`GlobalId`" in q["doors"]
    assert q["voxels"] == "MATCH (v:`Vox`) RETURN properties(v) AS props"


def test_fetch_rows_returns_plain_dicts(rows):
    fetched = fetch_rows(FakeDriver(rows), floor=FLOOR)
    assert fetched.rooms == rows.rooms and fetched.doors == rows.doors
    assert len(fetched.voxels) == len(rows.voxels)


@pytest.mark.skipif(importlib.util.find_spec("neo4j") is not None, reason="neo4j is installed")
def test_missing_driver_package_is_explained():
    with pytest.raises(ImportError, match=r"\[neo4j\]"):
        Neo4jGraphSource("bolt://nowhere", "neo4j", "secret", FLOOR).load()


# ---- rooms, doors, floors --------------------------------------------------

def test_graph_matches_the_fake_graph(graph):
    ref = build_fake_graph()
    assert {r.id: r.bounds for r in graph.rooms.values()} == {
        r.id: r.bounds for r in ref.rooms.values()}
    assert graph.doors == ref.doors
    assert graph.rooms["A"].aliases == ("401",) and graph.find_room("402").id == "B"
    assert graph.voxels.resolution == pytest.approx(0.1)
    assert_same_map(graph, ref)


def test_graph_is_flyable(graph):
    report = validate_graph(graph, policy="free_in_rooms")
    assert report.ok, report.format()
    plan = plan_route(graph, "A", "C")
    assert plan.rooms == ["A", "B", "C"]


def test_other_floors_and_unknown_rooms(rows):
    notes: list[str] = []
    g = build_graph(rows, floor=FLOOR, notes=notes)
    assert "U" not in g.rooms and "door_up" not in g.doors and "door_ghost" not in g.doors
    text = "\n".join(notes)
    assert "1 room(s) on other floors" in text
    assert "'door_up' skipped" in text and "another floor" in text
    assert "'door_ghost' skipped" in text and "unknown room(s) ['X9']" in text

    upstairs = build_graph(rows, floor="5")
    assert set(upstairs.rooms) == {"U"} and not upstairs.doors

    everything = build_graph(rows, floor=None)
    assert set(everything.rooms) == {"A", "B", "C", "U"}
    assert "door_up" in everything.doors and "door_ghost" not in everything.doors


def test_door_with_one_room_is_skipped(rows):
    doors = rows.doors + [{"props": {"id": "exit", "position": [0, 1, 1]}, "rooms": ["A"]}]
    notes: list[str] = []
    g = build_graph(GraphRows(rows.rooms, doors, []), floor=FLOOR, notes=notes)
    assert "exit" not in g.doors
    assert any("'exit' skipped: joins 1 room(s)" in n for n in notes)


# ---- units and custom schemas ----------------------------------------------

def test_unit_conversion(graph):
    g = build_graph(make_rows(scale=1000.0), Neo4jMapping(length_unit="mm"), FLOOR)
    for rid, room in graph.rooms.items():
        assert g.rooms[rid].bounds.min == pytest.approx(room.bounds.min)
        assert g.rooms[rid].bounds.max == pytest.approx(room.bounds.max)
    for did, door in graph.doors.items():
        assert g.doors[did].position == pytest.approx(door.position)
        assert g.doors[did].width == pytest.approx(door.width)
    assert g.voxels.resolution == pytest.approx(0.1)
    assert g.voxels.shape == graph.voxels.shape
    assert np.array_equal(g.voxels.states, graph.voxels.states)
    assert Neo4jMapping(length_unit=0.5).scale == 0.5
    with pytest.raises(Neo4jMappingError, match="length_unit"):
        build_graph(make_rows(), Neo4jMapping(length_unit="yard"), FLOOR)


def test_wrong_unit_is_caught(rows):
    mm = GraphRows(make_rows(scale=1000.0).rooms, [], [])
    with pytest.raises(Neo4jMappingError, match="length_unit"):
        build_graph(mm, Neo4jMapping(grid_resolution=0.1), FLOOR)


def test_custom_schema(graph):
    """Other names, scalar coordinates, room ids on the door, constant voxel size."""
    base = make_rows()
    rooms = [
        {"props": {"GlobalId": p["id"], "LongName": p["name"], "storey": f"L{p['floor']}",
                   "x0": p["bbox_min"][0], "y0": p["bbox_min"][1], "z0": p["bbox_min"][2],
                   "x1": p["bbox_max"][0], "y1": p["bbox_max"][1], "z1": p["bbox_max"][2]}}
        for p in (r["props"] for r in base.rooms)
    ]
    doors = [
        {"props": {"GlobalId": d["props"]["id"], "from": d["rooms"][0], "to": d["rooms"][1],
                   "center": dict(zip("xyz", d["props"]["position"]))}}
        for d in base.doors
    ]
    voxels = [
        {"props": {"cx": v["props"]["position"][0] - 0.05, "cy": v["props"]["position"][1] - 0.05,
                   "cz": v["props"]["position"][2] - 0.05}}
        for v in base.voxels
    ]
    mapping = Neo4jMapping(
        room_id="GlobalId", room_name="LongName", room_aliases=(), room_floor="storey",
        room_min=("x0", "y0", "z0"), room_max=("x1", "y1", "z1"), room_polygon=None,
        door_id="GlobalId", door_position="center", door_width=None,
        door_room_props=("from", "to"), default_door_width=1.0,
        voxel_position=("cx", "cy", "cz"), voxel_anchor="corner", voxel_size=0.1,
        voxel_state=None,
    )
    g = build_graph(GraphRows(rooms, doors, voxels), mapping, floor="l4")
    assert g.rooms["B"].name == "Room B"
    assert g.doors == graph.doors
    assert np.array_equal(g.voxels.states, graph.voxels.states)
    assert "RETURN properties(d) AS props, [] AS rooms" in build_queries(mapping)["doors"]


def test_optional_properties_have_defaults():
    rows = GraphRows(
        rooms=[{"props": {"id": 7, "floor": FLOOR, "bbox_min": [0, 0, 0], "bbox_max": [2, 2, 2]}},
               {"props": {"id": 8, "floor": FLOOR, "bbox_min": [2, 0, 0], "bbox_max": [4, 2, 2],
                          "number": ["408", "408A"], "polygon": [2, 0, 4, 0, 4, 2, 2, 2]}}],
        doors=[{"props": {"id": "d", "position": [2, 1, 1]}, "rooms": [7, 8]}],
    )
    notes: list[str] = []
    g = build_graph(rows, floor=FLOOR, notes=notes)
    assert g.rooms["7"].name == "7" and g.rooms["7"].aliases == () and g.rooms["7"].polygon is None
    assert g.rooms["8"].aliases == ("408", "408A")
    assert g.rooms["8"].polygon == ((2.0, 0.0), (4.0, 0.0), (4.0, 2.0), (2.0, 2.0))
    assert g.doors["d"].width == 0.9
    assert g.voxels.resolution == 0.1 and not g.solids  # no voxels: default resolution
    assert any("no voxel found" in n for n in notes)


# ---- clear errors ----------------------------------------------------------

def _without(rows: GraphRows, table: str, prop: str) -> GraphRows:
    changed = [{**r, "props": {k: v for k, v in r["props"].items() if k != prop}}
               for r in getattr(rows, table)]
    return GraphRows(**{**rows.__dict__, table: changed})


@pytest.mark.parametrize("table, prop, label, mapping_field", [
    ("rooms", "id", "Space", "room_id"),
    ("rooms", "bbox_min", "Space", "room_min"),
    ("rooms", "bbox_max", "Space", "room_max"),
    ("rooms", "floor", "Space", "room_floor"),
    ("doors", "id", "Door", "door_id"),
    ("doors", "position", "Door", "door_position"),
    ("voxels", "position", "Voxel", "voxel_position"),
    ("voxels", "size", "Voxel", "voxel_size"),
])
def test_missing_property_names_the_mapping_field(rows, table, prop, label, mapping_field):
    with pytest.raises(Neo4jMappingError) as e:
        build_graph(_without(rows, table, prop), floor=FLOOR)
    message = str(e.value)
    assert repr(prop) in message and repr(label) in message
    assert f"Neo4jMapping.{mapping_field}" in message
    assert "properties found" in message


def test_missing_label_and_floor_errors(rows):
    with pytest.raises(Neo4jMappingError, match=r"label 'Space'.*Neo4jMapping\.room_label"):
        build_graph(GraphRows(), floor=FLOOR)
    with pytest.raises(Neo4jMappingError, match=r"no room on floor '9'.*\['4', '5'\].*room_floor"):
        build_graph(rows, floor="9")
    with pytest.raises(Neo4jMappingError, match="rooms_query"):
        build_graph(GraphRows(rooms=[{"r": {}}]), floor=FLOOR)


def test_bad_values_are_explained(rows):
    def one_room(**props):
        base = {"id": "A", "floor": FLOOR, "bbox_min": [0, 0, 0], "bbox_max": [2, 2, 2]}
        return GraphRows(rooms=[{"props": {**base, **props}}])

    with pytest.raises(Neo4jMappingError, match="expected 3 numbers.*room_min"):
        build_graph(one_room(bbox_min=[0, 0]), floor=FLOOR)
    with pytest.raises(Neo4jMappingError, match="bounds .* are empty"):
        build_graph(one_room(bbox_max=[0, 0, 0]), floor=FLOOR)
    with pytest.raises(Neo4jMappingError, match="room_polygon"):
        build_graph(one_room(polygon=[0, 0, 1]), floor=FLOOR)
    with pytest.raises(Neo4jMappingError, match="appears twice.*room_id"):
        build_graph(GraphRows(rooms=one_room().rooms * 2), floor=FLOOR)
    bad_state = one_room()
    bad_state.voxels = [{"props": {"position": [1, 1, 1], "size": 0.1, "state": "wall"}}]
    with pytest.raises(Neo4jMappingError, match="'wall'.*voxel_state_values"):
        build_graph(bad_state, floor=FLOOR)
    with pytest.raises(Neo4jMappingError, match="voxel_anchor"):
        build_graph(bad_state, Neo4jMapping(voxel_anchor="middle"), FLOOR)


# ---- voxels and solids -----------------------------------------------------

def _room_rows() -> GraphRows:
    return GraphRows(rooms=[
        {"props": {"id": "R", "floor": FLOOR, "bbox_min": [0, 0, 0], "bbox_max": [2, 2, 2]}}])


def _voxel(x, y, z, state=None, size=0.2):
    props = {"position": [x, y, z], "size": size}
    if state is not None:
        props["state"] = state
    return {"props": props}


def test_voxel_merge_states():
    rows = _room_rows()
    rows.voxels = [
        _voxel(1.1, 1.1, 0.1, "occupied"),
        _voxel(1.1, 1.1, 0.1, "free"),  # same cell: occupied wins
        _voxel(1.5, 1.1, 0.1),  # no state: occupied by default
        _voxel(2.1, 1.1, 0.1, 1),  # free, outside the room (in the margin)
        _voxel(0.5, 0.5, 0.1, 0),  # "unknown" rows change nothing
        _voxel(50.0, 1.0, 0.1, "occupied"),  # far outside the floor
    ]
    notes: list[str] = []
    g = build_graph(rows, floor=FLOOR, notes=notes)
    grid = g.voxels
    assert grid.resolution == pytest.approx(0.2)
    assert grid.origin == pytest.approx((-0.4, -0.4, 0.0))  # margin, aligned to the voxels
    assert grid.state_at((1.1, 1.1, 0.1)) == VoxelState.OCCUPIED
    assert grid.state_at((1.5, 1.1, 0.1)) == VoxelState.OCCUPIED
    assert grid.state_at((2.1, 1.1, 0.1)) == VoxelState.FREE
    assert grid.state_at((2.1, 1.5, 0.1)) == VoxelState.UNKNOWN  # margin, never observed
    assert grid.state_at((0.5, 0.5, 0.1)) == VoxelState.FREE  # room interior
    assert int((grid.states == int(VoxelState.OCCUPIED)).sum()) == 2
    assert notes == ["1 voxel(s) outside the floor grid skipped"]

    unknown_rooms = build_graph(rows, Neo4jMapping(rooms_are_free=False), FLOOR)
    assert unknown_rooms.voxels.state_at((0.5, 0.5, 0.1)) == VoxelState.UNKNOWN


def test_grid_resolution_can_differ_from_the_data():
    rows = _room_rows()
    rows.voxels = [_voxel(1.1, 1.1, 0.1, "occupied")]  # 0.2 m voxel at 1.0 .. 1.2
    fine = build_graph(rows, Neo4jMapping(grid_resolution=0.1), FLOOR).voxels
    assert fine.resolution == 0.1
    assert int((fine.states == int(VoxelState.OCCUPIED)).sum()) == 8
    assert fine.state_at((1.05, 1.15, 0.15)) == VoxelState.OCCUPIED
    coarse = build_graph(rows, Neo4jMapping(grid_resolution=0.5), FLOOR).voxels
    assert int((coarse.states == int(VoxelState.OCCUPIED)).sum()) == 1  # never lost
    assert coarse.state_at((1.1, 1.1, 0.1)) == VoxelState.OCCUPIED


def test_solids_cover_exactly_the_occupied_voxels(graph):
    occupied = graph.voxels.states == int(VoxelState.OCCUPIED)
    assert 0 < len(graph.solids) < int(occupied.sum()) / 20
    redrawn = VoxelGrid.empty(graph.voxels.origin, (0, 0, 0), graph.voxels.resolution)
    redrawn.states = np.zeros(graph.voxels.shape, dtype=np.uint8)
    for box in graph.solids:
        redrawn.fill_box(box, VoxelState.OCCUPIED)
    assert np.array_equal(redrawn.states == int(VoxelState.OCCUPIED), occupied)
    volume = sum(np.prod(b.size) for b in graph.solids)
    assert volume == pytest.approx(occupied.sum() * graph.voxels.resolution ** 3)  # no overlap

    assert solids_from_voxels(VoxelGrid.empty((0, 0, 0), (1, 1, 1), 0.5)) == []
    assert build_graph(make_rows(), Neo4jMapping(build_solids=False), FLOOR).solids == []
