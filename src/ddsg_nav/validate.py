"""Validate a SceneGraph before planning on it: turns "no path found" into a reason.

Run it on every new data export (`scripts/graph_tools.py validate <folder>`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from .graph.model import Box, SceneGraph, VoxelState
from .planner.local_planner import NoPathError, plan_voxel_path
from .planner.policy import UnknownPolicy, blocked_mask, raw_obstacles
from .planner.route import nearest_free_point


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.stats.items()]
        for sev in ("error", "warning", "info"):
            for i in self.issues:
                if i.severity == sev:
                    lines.append(f"[{sev.upper():7}] {i.code}: {i.message}")
        lines.append(
            f"RESULT: {'OK' if self.ok else 'NOT OK'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )
        return "\n".join(lines)


def _grid_box(graph: SceneGraph) -> Box:
    g = graph.voxels
    lo = g.origin
    hi = tuple(o + n * g.resolution for o, n in zip(lo, g.shape))
    return Box(lo, hi)  # type: ignore[arg-type]


def validate_graph(
    graph: SceneGraph,
    flight_height: float = 0.5,
    inflation_m: float = 0.25,
    policy: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE,
    check_connectivity: bool = True,
) -> ValidationReport:
    """Check structure, geometry and flyability at `flight_height` above each floor
    (a room's floor, or at a door the higher floor of its two rooms)."""
    rep = ValidationReport()
    grid = graph.voxels
    states = grid.states
    total = max(states.size, 1)
    rep.stats = {
        "rooms": len(graph.rooms),
        "doors": len(graph.doors),
        "grid_shape": tuple(grid.shape),
        "resolution_m": grid.resolution,
        "voxels": int(states.size),
        "occupied_ratio": round(float((states == int(VoxelState.OCCUPIED)).sum()) / total, 3),
        "unknown_ratio": round(float((states == int(VoxelState.UNKNOWN)).sum()) / total, 3),
        "free_ratio": round(float((states == int(VoxelState.FREE)).sum()) / total, 3),
    }

    if not graph.rooms:
        rep.add("error", "no-rooms", "the graph has no rooms")
        return rep
    if rep.stats["unknown_ratio"] > 0.5:
        rep.add(
            "warning",
            "mostly-unknown",
            f"{rep.stats['unknown_ratio']:.0%} of the voxels are unobserved; with policy "
            f"'{UnknownPolicy.coerce(policy).value}' most routes may be impossible",
        )

    extent = _grid_box(graph)
    for room in graph.rooms.values():
        lo, hi = room.bounds.min, room.bounds.max
        if any(a < b - 1e-9 for a, b in zip(lo[:2], extent.min[:2])) or any(
            a > b + 1e-9 for a, b in zip(hi[:2], extent.max[:2])
        ):
            rep.add("error", "room-outside-grid", f"room {room.id} extends outside the voxel grid")
        if room.polygon is not None and len(room.polygon) < 3:
            rep.add("error", "bad-polygon", f"room {room.id} polygon has fewer than 3 points")

    raw = raw_obstacles(graph, policy)
    blocked = blocked_mask(graph, inflation_m, policy)

    # room-level flyability at the flight height
    k_free: dict[str, bool] = {}
    for room in graph.rooms.values():
        k = grid.index((0.0, 0.0, graph.floor_z(room) + flight_height))[2]
        if not 0 <= k < grid.shape[2]:
            rep.add(
                "error",
                "height-outside-grid",
                f"flight height {flight_height} m above the floor of room {room.id} is outside the grid",
            )
            return rep
        # same rule as the planner: the flight height, or the lowest higher layer with space
        try:
            nearest_free_point(grid, blocked, room, graph.floor_z(room) + flight_height)
            k_free[room.id] = True
        except NoPathError:
            k_free[room.id] = False
            rep.add(
                "error",
                "room-not-flyable",
                f"room {room.id} has no flyable space at or above {flight_height} m "
                f"(after a {inflation_m} m safety margin)",
            )

    # doors
    for door in graph.doors.values():
        x, y, _ = door.position
        idx = grid.index((x, y, graph.door_floor_z(door) + flight_height))
        if not grid.in_bounds(idx):
            rep.add("error", "door-outside-grid", f"door {door.id} is outside the voxel grid")
            continue
        if raw[idx]:
            rep.add("error", "door-blocked", f"door {door.id} center is inside an obstacle at {flight_height} m")
        elif blocked[idx]:
            rep.add("warning", "door-too-tight", f"door {door.id} is within the safety margin of an obstacle")
        if door.width < 2 * inflation_m:
            rep.add(
                "warning",
                "door-narrower-than-margin",
                f"door {door.id} is {door.width} m wide, less than twice the {inflation_m} m margin",
            )
        for rid in (door.room_a, door.room_b):
            room = graph.rooms[rid]
            lo, hi = room.bounds.min, room.bounds.max
            dx = max(lo[0] - x, 0.0, x - hi[0])
            dy = max(lo[1] - y, 0.0, y - hi[1])
            if math.hypot(dx, dy) > max(door.width, 2 * grid.resolution):
                rep.add("warning", "door-far-from-room", f"door {door.id} is far from room {rid}")

    # connectivity of the room graph
    if len(graph.rooms) > 1:
        g = graph.room_graph()
        for room in graph.rooms.values():
            if g.degree[room.id] == 0:
                rep.add("warning", "isolated-room", f"room {room.id} has no door")
        comps = list(nx.connected_components(g))
        if len(comps) > 1:
            rep.add(
                "warning",
                "disconnected",
                f"{len(comps)} separate groups of rooms: "
                + "; ".join(",".join(sorted(c)) for c in comps),
            )

    # can the drone really get from each room to each of its doors?
    if check_connectivity and rep.ok:
        for door in graph.doors.values():
            x, y, _ = door.position
            target = (x, y, graph.door_floor_z(door) + flight_height)
            for rid in (door.room_a, door.room_b):
                if not k_free.get(rid):
                    continue
                try:
                    room = graph.rooms[rid]
                    start = nearest_free_point(grid, blocked, room, graph.floor_z(room) + flight_height)
                    plan_voxel_path(grid, start, target, blocked=blocked)
                except NoPathError as e:
                    rep.add("error", "door-unreachable", f"cannot fly from room {rid} to door {door.id}: {e}")

    rep.add("info", "checked", f"flight height {flight_height} m, margin {inflation_m} m, "
            f"unknown policy '{UnknownPolicy.coerce(policy).value}'")
    return rep
