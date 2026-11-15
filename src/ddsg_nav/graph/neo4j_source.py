"""Neo4j graph source: one floor of the 3DDSG -> `SceneGraph`, through a mapping.

The database schema is described by `Neo4jMapping` (labels, relationship types,
property names, units). When the real schema is known, fill in the mapping; no code
change should be needed. Fields that do not fit can be replaced by a Cypher query
(`rooms_query`, `doors_query`, `voxels_query`).

Two steps, so everything but the query can be tested without a database:

    fetch_rows(driver, mapping, floor) -> GraphRows     thin, needs a database
    build_graph(rows, mapping, floor)  -> SceneGraph    pure, plain dicts in

Row shapes (what the queries must return):

    rooms   {"props": {<room node properties>}}
    doors   {"props": {<door node properties>}, "rooms": [<room id>, <room id>]}
    voxels  {"props": {<voxel node properties>}}

What `build_graph` does:

* rooms    rooms of the requested floor -> `Room(id, name, bounds, aliases, polygon)`.
* doors    doors whose two rooms are both on the floor -> `Door(position, width)`.
           Doors to another floor or to an unknown room are skipped and reported.
* voxels   one regular grid over the floor's rooms plus `grid_margin`. Everything starts
           UNKNOWN, room interiors become FREE, then the per-element voxels are written:
           free first, occupied last (occupied wins).
* solids   boxes that cover the occupied voxels (only the simulator uses them).

All coordinates must be in one frame, z up. Lengths are converted to meters with
`length_unit`. See docs/graph_format.md and docs/3ddsg_request.md.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import Box, Door, Room, SceneGraph, VoxelGrid, VoxelState

# A 3D value is stored either in one list property ("bbox_min") or in three
# scalar properties (("min_x", "min_y", "min_z")).
Vec = str | tuple[str, str, str]

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "ft": 0.3048, "in": 0.0254}
MAX_CELLS = 200_000_000  # refuse grids larger than this (usually a unit mistake)
_EPS = 1e-6


class Neo4jMappingError(ValueError):
    """The data does not match the mapping. The message names the label or property
    and the `Neo4jMapping` field to change."""


def _default_state_values() -> dict[str, VoxelState]:
    return {
        "occupied": VoxelState.OCCUPIED, "2": VoxelState.OCCUPIED,
        "free": VoxelState.FREE, "1": VoxelState.FREE,
        "unknown": VoxelState.UNKNOWN, "0": VoxelState.UNKNOWN,
    }


@dataclass(frozen=True)
class Neo4jMapping:
    """How the 3DDSG is stored in Neo4j. The defaults are placeholders: the schema is not
    known yet. Property fields hold property NAMES, not values."""

    # --- node labels and relationship types ---
    room_label: str = "Space"
    door_label: str = "Door"
    voxel_label: str = "Voxel"
    # Voxels are reached as (element)-[voxel_rel]->(voxel). None: every voxel node is used.
    element_label: str | None = "Element"
    voxel_rel: str = "HAS_VOXEL"
    # A door is linked to its two rooms as (door)-[door_room_rel]-(room) ...
    door_room_rel: str = "CONNECTS"
    # ... or, if set, by two door properties that hold the room ids.
    door_room_props: tuple[str, str] | None = None

    # --- room properties ---
    room_id: str = "id"  # required, unique
    room_name: str | None = "name"  # optional, falls back to the id
    room_aliases: tuple[str, ...] = ("number", "global_id")  # optional, string or list
    room_floor: str | None = "floor"  # compared with the requested floor; None: no filter
    room_min: Vec = "bbox_min"  # required
    room_max: Vec = "bbox_max"  # required
    # Optional floor outline: [x0, y0, x1, y1, ...] or [[x, y], ...]. None: boxes only.
    room_polygon: str | None = "polygon"

    # --- door properties ---
    door_id: str = "id"  # required
    door_position: Vec = "position"  # required, center of the opening
    door_width: str | None = "width"  # optional
    default_door_width: float = 0.9  # meters, used when the width is missing

    # --- voxel properties ---
    voxel_position: Vec = "position"  # required
    voxel_anchor: str = "center"  # "center" or "corner" (minimum corner)
    voxel_size: str | float = "size"  # property name, or a constant edge length
    voxel_state: str | None = "state"  # None: every voxel is `voxel_default_state`
    voxel_state_values: Mapping[str, VoxelState] = field(default_factory=_default_state_values)
    voxel_default_state: VoxelState = VoxelState.OCCUPIED  # when the state is missing

    # --- units ---
    length_unit: str | float = "m"  # "m", "cm", "mm", "ft", "in", or meters per unit

    # --- output grid ---
    grid_resolution: float | None = None  # meters; None: the voxel size of the data, else 0.1
    grid_margin: float = 0.3  # meters added around the rooms in x and y (outer walls)
    rooms_are_free: bool = True  # room interiors start FREE (False: UNKNOWN)
    build_solids: bool = True

    # --- query overrides (Cypher). Each must return the row shape in the module
    # docstring; `$floor` is available as a parameter. ---
    rooms_query: str | None = None
    doors_query: str | None = None
    voxels_query: str | None = None

    @property
    def scale(self) -> float:
        """Meters per stored length unit."""
        if isinstance(self.length_unit, str):
            try:
                return UNITS[self.length_unit.lower()]
            except KeyError:
                raise Neo4jMappingError(
                    f"unknown length unit {self.length_unit!r}; set Neo4jMapping.length_unit "
                    f"to one of {sorted(UNITS)} or to meters per unit"
                ) from None
        return float(self.length_unit)


@dataclass
class GraphRows:
    """Plain query results (see the module docstring for the row shapes)."""

    rooms: list[dict[str, Any]] = field(default_factory=list)
    doors: list[dict[str, Any]] = field(default_factory=list)
    voxels: list[dict[str, Any]] = field(default_factory=list)


# ---- fetch (needs a database) ----------------------------------------------

def _q(name: str) -> str:
    """Quote a label, relationship type or property name for Cypher."""
    return "`" + name.replace("`", "``") + "`"


def build_queries(mapping: Neo4jMapping) -> dict[str, str]:
    """The three Cypher queries for a mapping (overrides applied)."""
    m = mapping
    rooms = f"MATCH (r:{_q(m.room_label)}) RETURN properties(r) AS props"
    if m.door_room_props:
        doors = f"MATCH (d:{_q(m.door_label)}) RETURN properties(d) AS props, [] AS rooms"
    else:
        doors = (
            f"MATCH (d:{_q(m.door_label)}) "
            f"OPTIONAL MATCH (d)-[:{_q(m.door_room_rel)}]-(r:{_q(m.room_label)}) "
            f"RETURN properties(d) AS props, collect(DISTINCT r.{_q(m.room_id)}) AS rooms"
        )
    if m.element_label:
        voxels = (
            f"MATCH (:{_q(m.element_label)})-[:{_q(m.voxel_rel)}]->(v:{_q(m.voxel_label)}) "
            "RETURN properties(v) AS props"
        )
    else:
        voxels = f"MATCH (v:{_q(m.voxel_label)}) RETURN properties(v) AS props"
    return {
        "rooms": m.rooms_query or rooms,
        "doors": m.doors_query or doors,
        "voxels": m.voxels_query or voxels,
    }


def fetch_rows(
    driver: Any,
    mapping: Neo4jMapping | None = None,
    floor: str | None = None,
    database: str | None = None,
) -> GraphRows:
    """Run the three queries and return plain rows. `driver` is a `neo4j.Driver`
    (or anything with the same `session().run()` interface)."""
    queries = build_queries(mapping or Neo4jMapping())
    kwargs = {"database": database} if database else {}
    with driver.session(**kwargs) as session:
        return GraphRows(**{
            name: [dict(record) for record in session.run(query, floor=floor)]
            for name, query in queries.items()
        })


# ---- build (pure) ----------------------------------------------------------

def _props(row: Mapping[str, Any], kind: str, query_field: str) -> Mapping[str, Any]:
    props = row.get("props")
    if not isinstance(props, Mapping):
        raise Neo4jMappingError(
            f"{kind} row has no 'props' map (got columns {sorted(row)}); "
            f"the query in Neo4jMapping.{query_field} must return properties(node) AS props"
        )
    return props


def _missing(kind: str, label: str, ident: object, prop: object, field_name: str,
             props: Mapping[str, Any], extra: str = "") -> Neo4jMappingError:
    return Neo4jMappingError(
        f"{kind} {ident!r} (label {label!r}): property {prop!r} is missing or empty; "
        f"set Neo4jMapping.{field_name} to the property that holds it{extra} "
        f"(properties found: {sorted(props)})"
    )


def _vec(props: Mapping[str, Any], spec: Vec, kind: str, label: str, ident: object,
         field_name: str) -> tuple[float, float, float]:
    if isinstance(spec, str):
        value = props.get(spec)
        if value is None:
            raise _missing(kind, label, ident, spec, field_name, props)
        if isinstance(value, Mapping):  # Neo4j point, returned as a map
            value = [value.get(k) for k in ("x", "y", "z")]
        elif all(hasattr(value, k) for k in ("x", "y", "z")):  # Neo4j point object
            value = [value.x, value.y, value.z]
    else:
        value = [props.get(name) for name in spec]
        for name, v in zip(spec, value):
            if v is None:
                raise _missing(kind, label, ident, name, field_name, props)
    try:
        x, y, z = (float(v) for v in value)
    except (TypeError, ValueError):
        raise Neo4jMappingError(
            f"{kind} {ident!r} (label {label!r}): property {spec!r} is {value!r}, expected "
            f"3 numbers; set Neo4jMapping.{field_name} to one list property or to three "
            "property names"
        ) from None
    return (x, y, z)


def _scaled(v: Iterable[float], scale: float) -> tuple[float, float, float]:
    return tuple(float(c) * scale for c in v)  # type: ignore[return-value]


def _ident(props: Mapping[str, Any], prop: str, kind: str, label: str, field_name: str) -> str:
    value = props.get(prop)
    if value is None or value == "":
        raise _missing(kind, label, "?", prop, field_name, props)
    return str(value)


def _floor_key(value: object) -> str:
    return str(value).strip().lower()


def _polygon(value: Any, scale: float, room_id: str,
             mapping: Neo4jMapping) -> tuple[tuple[float, float], ...] | None:
    if value is None or len(value) == 0:
        return None
    try:
        if isinstance(value[0], (int, float)):
            if len(value) % 2:
                raise ValueError
            pts = [(value[i], value[i + 1]) for i in range(0, len(value), 2)]
        else:
            pts = [(p[0], p[1]) for p in value]
        polygon = tuple((float(x) * scale, float(y) * scale) for x, y in pts)
    except (TypeError, ValueError, IndexError):
        polygon = ()
    if len(polygon) < 3:
        raise Neo4jMappingError(
            f"room {room_id!r} (label {mapping.room_label!r}): property "
            f"{mapping.room_polygon!r} is not a polygon; expected [x0, y0, x1, y1, ...] or "
            "[[x, y], ...] with at least 3 points. Set Neo4jMapping.room_polygon to the "
            "right property, or to None to use the bounding boxes"
        )
    return polygon


def _build_rooms(rows: GraphRows, m: Neo4jMapping, floor: str | None,
                 notes: list[str]) -> tuple[dict[str, Room], set[str]]:
    """Rooms of the floor, and the ids of the rooms on every floor."""
    if not rows.rooms:
        raise Neo4jMappingError(
            f"no room found: no node has the label {m.room_label!r}; "
            "set Neo4jMapping.room_label (or rooms_query)"
        )
    scale = m.scale
    rooms: dict[str, Room] = {}
    all_ids: set[str] = set()
    floors_seen: set[str] = set()
    for row in rows.rooms:
        props = _props(row, "room", "rooms_query")
        rid = _ident(props, m.room_id, "room", m.room_label, "room_id")
        if rid in all_ids:
            raise Neo4jMappingError(
                f"room id {rid!r} appears twice (property {m.room_id!r}); "
                "set Neo4jMapping.room_id to a property that is unique"
            )
        all_ids.add(rid)
        if m.room_floor is not None and floor is not None:
            value = props.get(m.room_floor)
            if value is None:
                raise _missing("room", m.room_label, rid, m.room_floor, "room_floor", props,
                               " (None loads every floor)")
            floors_seen.add(str(value))
            if _floor_key(value) != _floor_key(floor):
                continue
        lo = _scaled(_vec(props, m.room_min, "room", m.room_label, rid, "room_min"), scale)
        hi = _scaled(_vec(props, m.room_max, "room", m.room_label, rid, "room_max"), scale)
        if any(b <= a for a, b in zip(lo, hi)):
            raise Neo4jMappingError(
                f"room {rid!r} (label {m.room_label!r}): bounds {lo} .. {hi} are empty; check "
                "Neo4jMapping.room_min and room_max (minimum and maximum corner)"
            )
        name = props.get(m.room_name) if m.room_name else None
        aliases: list[str] = []
        for prop in m.room_aliases:
            value = props.get(prop)
            values = value if isinstance(value, (list, tuple)) else [value]
            for v in values:
                if v not in (None, "") and str(v) not in aliases and str(v) != rid:
                    aliases.append(str(v))
        polygon = None
        if m.room_polygon:
            polygon = _polygon(props.get(m.room_polygon), scale, rid, m)
        rooms[rid] = Room(rid, str(name) if name not in (None, "") else rid,
                          Box(lo, hi), tuple(aliases), polygon)
    if not rooms:
        raise Neo4jMappingError(
            f"no room on floor {floor!r}: property {m.room_floor!r} has the values "
            f"{sorted(floors_seen)}; pass one of them as `floor`, or set "
            "Neo4jMapping.room_floor to the property that holds the floor id"
        )
    skipped = len(all_ids) - len(rooms)
    if skipped:
        notes.append(f"{skipped} room(s) on other floors skipped")
    return rooms, all_ids


def _build_doors(rows: GraphRows, m: Neo4jMapping, rooms: dict[str, Room],
                 all_ids: set[str], notes: list[str]) -> list[Door]:
    scale = m.scale
    doors: list[Door] = []
    seen: set[str] = set()
    for row in rows.doors:
        props = _props(row, "door", "doors_query")
        did = _ident(props, m.door_id, "door", m.door_label, "door_id")
        if m.door_room_props:
            ends = [props.get(p) for p in m.door_room_props]
            where = f"properties {m.door_room_props!r} (Neo4jMapping.door_room_props)"
        else:
            ends = list(row.get("rooms") or [])
            where = f"relationship {m.door_room_rel!r} (Neo4jMapping.door_room_rel)"
        ends = list(dict.fromkeys(str(e) for e in ends if e not in (None, "")))
        if len(ends) != 2:
            notes.append(f"door {did!r} skipped: joins {len(ends)} room(s), expected 2 ({where})")
            continue
        unknown = [e for e in ends if e not in all_ids]
        if unknown:
            notes.append(f"door {did!r} skipped: joins unknown room(s) {unknown} ({where})")
            continue
        other = [e for e in ends if e not in rooms]
        if other:
            notes.append(f"door {did!r} skipped: room(s) {other} are on another floor")
            continue
        if did in seen:
            raise Neo4jMappingError(
                f"door id {did!r} appears twice (property {m.door_id!r}); "
                "set Neo4jMapping.door_id to a property that is unique"
            )
        seen.add(did)
        position = _scaled(
            _vec(props, m.door_position, "door", m.door_label, did, "door_position"), scale
        )
        width = props.get(m.door_width) if m.door_width else None
        width = float(width) * scale if width is not None else m.default_door_width
        doors.append(Door(did, ends[0], ends[1], position, width))
    return doors


def _voxel_arrays(rows: GraphRows, m: Neo4jMapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum corners (n, 3), edge lengths (n,) in meters, and states (n,)."""
    scale = m.scale
    n = len(rows.voxels)
    corners = np.empty((n, 3))
    sizes = np.empty(n)
    states = np.empty(n, dtype=np.uint8)
    values = {str(k).lower(): VoxelState(v) for k, v in m.voxel_state_values.items()}
    if m.voxel_anchor not in ("center", "corner"):
        raise Neo4jMappingError(
            f"Neo4jMapping.voxel_anchor is {m.voxel_anchor!r}; use 'center' or 'corner'"
        )
    for i, row in enumerate(rows.voxels):
        props = _props(row, "voxel", "voxels_query")
        ident = f"#{i}"
        corners[i] = _vec(props, m.voxel_position, "voxel", m.voxel_label, ident, "voxel_position")
        if isinstance(m.voxel_size, str):
            size = props.get(m.voxel_size)
            if size is None:
                raise _missing("voxel", m.voxel_label, ident, m.voxel_size, "voxel_size", props,
                               " (or to a constant edge length)")
        else:
            size = m.voxel_size
        sizes[i] = float(size)
        raw = props.get(m.voxel_state) if m.voxel_state else None
        if raw is None:
            states[i] = int(m.voxel_default_state)
        else:
            key = str(int(raw) if isinstance(raw, (bool, int, float)) else raw).strip().lower()
            if key not in values:
                raise Neo4jMappingError(
                    f"voxel {ident} (label {m.voxel_label!r}): state {raw!r} in property "
                    f"{m.voxel_state!r} is not known; add it to "
                    f"Neo4jMapping.voxel_state_values (known: {sorted(values)})"
                )
            states[i] = int(values[key])
    corners *= scale
    sizes *= scale
    if n and float(sizes.min()) <= 0:
        raise Neo4jMappingError(
            "voxel size must be positive; check Neo4jMapping.voxel_size and length_unit"
        )
    if m.voxel_anchor == "center":
        corners -= sizes[:, None] / 2
    return corners, sizes, states


def _write_voxels(grid: VoxelGrid, corners: np.ndarray, sizes: np.ndarray,
                  state: VoxelState) -> int:
    """Set every grid cell that overlaps one of the source voxels. Returns how many
    source voxels lie completely outside the grid."""
    if not len(corners):
        return 0
    origin = np.asarray(grid.origin)
    shape = np.asarray(grid.shape)
    res = grid.resolution
    lo = np.floor((corners - origin) / res + _EPS).astype(np.int64)
    hi = np.ceil((corners + sizes[:, None] - origin) / res - _EPS).astype(np.int64)
    hi = np.maximum(hi, lo + 1)
    outside = ((hi <= 0) | (lo >= shape)).any(axis=1)
    lo, hi = lo[~outside], hi[~outside]
    if len(lo):
        span = (hi - lo).max(axis=0)
        for dx in range(int(span[0])):
            for dy in range(int(span[1])):
                for dz in range(int(span[2])):
                    idx = lo + (dx, dy, dz)
                    ok = ((idx < hi) & (idx >= 0) & (idx < shape)).all(axis=1)
                    i, j, k = idx[ok].T
                    grid.states[i, j, k] = int(state)
    return int(outside.sum())


def solids_from_voxels(grid: VoxelGrid) -> list[Box]:
    """Boxes that cover exactly the occupied voxels: runs along x, merged along y, then z."""
    occupied = grid.states == int(VoxelState.OCCUPIED)
    nx = occupied.shape[0]
    padded = np.zeros((nx + 2, *occupied.shape[1:]), dtype=np.int8)
    padded[1:-1] = occupied
    edges = np.diff(padded, axis=0)

    def ordered(a: np.ndarray) -> np.ndarray:  # by (j, k), then i
        return a[np.lexsort((a[:, 0], a[:, 2], a[:, 1]))]

    starts, ends = ordered(np.argwhere(edges == 1)), ordered(np.argwhere(edges == -1))
    # runs (i0, i1, j, k) -> rectangles (i0, i1, j0, j1, k) -> boxes (i0, i1, j0, j1, k0, k1)
    runs = sorted((int(k), int(i0), int(i1), int(j))
                  for (i0, j, k), i1 in zip(starts, ends[:, 0]))
    rects: list[list[int]] = []
    for k, i0, i1, j in runs:
        last = rects[-1] if rects else None
        if last and last[:3] == [k, i0, i1] and last[4] == j:
            last[4] = j + 1
        else:
            rects.append([k, i0, i1, j, j + 1])
    boxes: list[list[int]] = []
    for k, i0, i1, j0, j1 in sorted(rects, key=lambda r: (r[1], r[2], r[3], r[4], r[0])):
        last = boxes[-1] if boxes else None
        if last and last[:4] == [i0, i1, j0, j1] and last[5] == k:
            last[5] = k + 1
        else:
            boxes.append([i0, i1, j0, j1, k, k + 1])
    o, r = grid.origin, grid.resolution
    return [
        Box((o[0] + i0 * r, o[1] + j0 * r, o[2] + k0 * r),
            (o[0] + i1 * r, o[1] + j1 * r, o[2] + k1 * r))
        for i0, i1, j0, j1, k0, k1 in boxes
    ]


def _make_grid(rooms: dict[str, Room], corners: np.ndarray, sizes: np.ndarray,
               m: Neo4jMapping) -> VoxelGrid:
    """Grid over the rooms plus the margin, aligned with the source voxels."""
    if m.grid_resolution is not None:
        res = float(m.grid_resolution)
    elif len(sizes):
        res = float(np.median(sizes))
    else:
        res = 0.1
    if res <= 0:
        raise Neo4jMappingError("Neo4jMapping.grid_resolution must be positive")
    pad = np.array([m.grid_margin, m.grid_margin, 0.0])
    lo = np.min([r.bounds.min for r in rooms.values()], axis=0) - pad
    hi = np.max([r.bounds.max for r in rooms.values()], axis=0) + pad
    anchor = corners[0] if len(corners) else lo
    origin = anchor + np.floor((lo - anchor) / res + _EPS) * res
    shape = np.maximum(np.ceil((hi - origin) / res - _EPS).astype(np.int64), 1)
    if int(shape.prod()) > MAX_CELLS:
        raise Neo4jMappingError(
            f"voxel grid would be {tuple(int(n) for n in shape)} cells at {res:g} m; check "
            "Neo4jMapping.length_unit (are the coordinates in mm?) or set grid_resolution"
        )
    states = np.full(tuple(int(n) for n in shape), int(VoxelState.UNKNOWN), dtype=np.uint8)
    return VoxelGrid(tuple(float(c) for c in origin), res, states)  # type: ignore[arg-type]


def build_graph(
    rows: GraphRows,
    mapping: Neo4jMapping | None = None,
    floor: str | None = None,
    notes: list[str] | None = None,
) -> SceneGraph:
    """Build the `SceneGraph` of one floor from plain rows. Pure: no database.

    `floor` is compared with the `room_floor` property (as text, case-insensitive);
    None loads every room. Skipped items are appended to `notes` (one line each).
    Raises `Neo4jMappingError` when the rows do not match the mapping.
    """
    m = mapping or Neo4jMapping()
    notes = notes if notes is not None else []
    rooms, all_ids = _build_rooms(rows, m, floor, notes)
    doors = _build_doors(rows, m, rooms, all_ids, notes)
    corners, sizes, states = _voxel_arrays(rows, m)

    graph = SceneGraph(voxels=_make_grid(rooms, corners, sizes, m))
    for room in rooms.values():
        graph.add_room(room)
        if m.rooms_are_free:
            graph.voxels.states[graph.room_mask(room)] = int(VoxelState.FREE)
    for door in doors:
        graph.add_door(door)

    outside = 0
    for state in (VoxelState.FREE, VoxelState.OCCUPIED):  # occupied wins
        pick = states == int(state)
        outside += _write_voxels(graph.voxels, corners[pick], sizes[pick], state)
    if outside:
        notes.append(f"{outside} voxel(s) outside the floor grid skipped")
    if not len(corners):
        notes.append(
            f"no voxel found (label {m.voxel_label!r}): the map has no walls; check "
            "Neo4jMapping.voxel_label, element_label and voxel_rel"
        )
    if m.build_solids:
        graph.solids = solids_from_voxels(graph.voxels)
    return graph


# ---- source ----------------------------------------------------------------

class Neo4jGraphSource:
    """`GraphSource` for one floor of the 3DDSG.

    `mapping` describes the schema (default: `Neo4jMapping()`). `driver` can be given
    for tests; otherwise a `neo4j` driver is opened for the call and closed after it.
    After `load()`, `notes` lists what was skipped.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        floor: str,
        mapping: Neo4jMapping | None = None,
        driver: Any = None,
        database: str | None = None,
    ) -> None:
        self.uri, self.user, self.floor = uri, user, floor
        self._password = password
        self.mapping = mapping or Neo4jMapping()
        self.database = database
        self._driver = driver
        self.notes: list[str] = []

    def fetch(self) -> GraphRows:
        """Query the database and return plain rows."""
        if self._driver is not None:
            return fetch_rows(self._driver, self.mapping, self.floor, self.database)
        try:
            from neo4j import GraphDatabase
        except ImportError as e:
            raise ImportError(
                'the neo4j driver is not installed: pip install -e ".[neo4j]"'
            ) from e
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self._password))
        try:
            return fetch_rows(driver, self.mapping, self.floor, self.database)
        finally:
            driver.close()

    def load(self) -> SceneGraph:
        self.notes = []
        return build_graph(self.fetch(), self.mapping, self.floor, self.notes)


__all__ = [
    "GraphRows",
    "Neo4jGraphSource",
    "Neo4jMapping",
    "Neo4jMappingError",
    "build_graph",
    "build_queries",
    "fetch_rows",
    "solids_from_voxels",
]
