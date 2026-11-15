"""Scene graph data model: rooms, doors (connections), voxel occupancy state."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import networkx as nx
import numpy as np

Point = tuple[float, float, float]

_EPS = 1e-9


class VoxelState(IntEnum):
    UNKNOWN = 0  # not observed
    FREE = 1
    OCCUPIED = 2


def dist(a: Point, b: Point) -> float:
    return math.dist(a, b)


Polygon = tuple[tuple[float, float], ...]


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon: Polygon) -> np.ndarray:
    """Vectorized even-odd (ray casting) point-in-polygon test for 2D points."""
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > ys) != (y2 > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= crosses & (xs < x_at)
    return inside


def polygon_centroid(polygon: Polygon) -> tuple[float, float]:
    """Area-weighted centroid (falls back to the vertex mean for degenerate polygons)."""
    a = cx = cy = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    return (cx / (3 * a), cy / (3 * a))


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in meters, from `min` corner to `max` corner."""

    min: Point
    max: Point

    @property
    def center(self) -> Point:
        return tuple((lo + hi) / 2 for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    @property
    def size(self) -> Point:
        return tuple(hi - lo for lo, hi in zip(self.min, self.max))  # type: ignore[return-value]

    def contains(self, p: Point) -> bool:
        return all(lo <= c <= hi for lo, c, hi in zip(self.min, p, self.max))


@dataclass(frozen=True)
class Room:
    """A room (IFC space). `bounds` is its bounding box; `polygon` is the exact floor
    footprint (x, y) when the room is not rectangular. `aliases` are extra names that
    should resolve to this room (for example an IFC GlobalId or "401호")."""

    id: str
    name: str
    bounds: Box
    aliases: tuple[str, ...] = ()
    polygon: Polygon | None = None

    @property
    def center(self) -> Point:
        if self.polygon:
            cx, cy = polygon_centroid(self.polygon)
            return (cx, cy, self.bounds.center[2])
        return self.bounds.center

    def contains(self, p: Point) -> bool:
        if not self.bounds.contains(p):
            return False
        if self.polygon is None:
            return True
        return bool(points_in_polygon(np.array([p[0]]), np.array([p[1]]), self.polygon)[0])


@dataclass(frozen=True)
class Door:
    """Connection between two rooms. `position` is the center of the opening."""

    id: str
    room_a: str
    room_b: str
    position: Point
    width: float = 0.9


@dataclass
class VoxelGrid:
    """Regular voxel grid. Voxel (i, j, k) covers origin + [i, i+1) * resolution."""

    origin: Point
    resolution: float
    states: np.ndarray  # (nx, ny, nz) uint8 of VoxelState

    @classmethod
    def empty(
        cls,
        origin: Point,
        size: Point,
        resolution: float,
        state: VoxelState = VoxelState.UNKNOWN,
    ) -> "VoxelGrid":
        shape = tuple(int(round(s / resolution)) for s in size)
        return cls(origin, resolution, np.full(shape, int(state), dtype=np.uint8))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.states.shape  # type: ignore[return-value]

    def index(self, p: Point) -> tuple[int, int, int]:
        return tuple(  # type: ignore[return-value]
            int(math.floor((c - o) / self.resolution + _EPS)) for c, o in zip(p, self.origin)
        )

    def center(self, idx: tuple[int, int, int]) -> Point:
        return tuple(o + (i + 0.5) * self.resolution for i, o in zip(idx, self.origin))  # type: ignore[return-value]

    def in_bounds(self, idx: tuple[int, int, int]) -> bool:
        return all(0 <= i < n for i, n in zip(idx, self.shape))

    def state_at(self, p: Point) -> VoxelState:
        idx = self.index(p)
        if not self.in_bounds(idx):
            return VoxelState.UNKNOWN
        return VoxelState(int(self.states[idx]))

    def box_slices(self, box: Box) -> tuple[slice, slice, slice]:
        """Index slices of the voxels whose center lies in [box.min, box.max), clipped to the grid."""
        slices = []
        for lo, hi, o, n in zip(box.min, box.max, self.origin, self.shape):
            i0 = math.ceil((lo - o) / self.resolution - 0.5 - _EPS)
            i1 = math.ceil((hi - o) / self.resolution - 0.5 - _EPS)
            slices.append(slice(min(max(i0, 0), n), min(max(i1, 0), n)))
        return tuple(slices)  # type: ignore[return-value]

    def fill_box(self, box: Box, state: VoxelState) -> None:
        """Set every voxel whose center lies in [box.min, box.max)."""
        self.states[self.box_slices(box)] = int(state)


@dataclass
class SceneGraph:
    """Rooms and doors plus the voxel map and the solid geometry they came from."""

    voxels: VoxelGrid
    rooms: dict[str, Room] = field(default_factory=dict)
    doors: dict[str, Door] = field(default_factory=dict)
    solids: list[Box] = field(default_factory=list)  # walls/obstacles (used by the simulator)

    def add_room(self, room: Room) -> None:
        self.rooms[room.id] = room

    def add_door(self, door: Door) -> None:
        for rid in (door.room_a, door.room_b):
            if rid not in self.rooms:
                raise KeyError(f"door {door.id}: unknown room {rid!r}")
        self.doors[door.id] = door

    def add_solid(self, box: Box) -> None:
        self.solids.append(box)
        self.voxels.fill_box(box, VoxelState.OCCUPIED)

    def find_room(self, ref: str) -> Room:
        """Find a room by id, name or alias (case-insensitive)."""
        if ref in self.rooms:
            return self.rooms[ref]
        wanted = ref.lower()
        for room in self.rooms.values():
            if wanted in {room.id.lower(), room.name.lower(), *(a.lower() for a in room.aliases)}:
                return room
        raise KeyError(f"unknown room {ref!r}")

    def room_at(self, p: Point) -> Room | None:
        for room in self.rooms.values():
            if room.contains(p):
                return room
        return None

    def room_mask(self, room: Room) -> np.ndarray:
        """Boolean mask of the voxels inside the room (its polygon footprint, if any)."""
        grid = self.voxels
        mask = np.zeros(grid.shape, dtype=bool)
        sx, sy, sz = grid.box_slices(room.bounds)
        if sx.stop <= sx.start or sy.stop <= sy.start or sz.stop <= sz.start:
            return mask
        xs = grid.origin[0] + (np.arange(sx.start, sx.stop) + 0.5) * grid.resolution
        ys = grid.origin[1] + (np.arange(sy.start, sy.stop) + 0.5) * grid.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        if room.polygon is not None:
            inside = points_in_polygon(gx, gy, room.polygon)
        else:
            inside = np.ones(gx.shape, dtype=bool)
        mask[sx, sy, sz] = inside[:, :, None]
        return mask

    def floor_z(self, room: Room | str) -> float:
        """Floor level of a room (the bottom of its bounds)."""
        r = self.rooms[room] if isinstance(room, str) else room
        return r.bounds.min[2]

    def door_floor_z(self, door: Door) -> float:
        """Floor level at a door: the higher floor of its two rooms. A stairwell spans
        both floors, so its door to an upper room sits on that room's floor."""
        return max(self.floor_z(door.room_a), self.floor_z(door.room_b))

    def room_graph(self) -> nx.Graph:
        """Room connectivity graph. Edge weight = path length through the door."""
        g = nx.Graph()
        for room in self.rooms.values():
            g.add_node(room.id)
        for door in self.doors.values():
            a, b = self.rooms[door.room_a], self.rooms[door.room_b]
            weight = dist(a.center, door.position) + dist(door.position, b.center)
            g.add_edge(door.room_a, door.room_b, door=door.id, weight=weight)
        return g
