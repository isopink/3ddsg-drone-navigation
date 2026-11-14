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
    id: str
    name: str
    bounds: Box

    @property
    def center(self) -> Point:
        return self.bounds.center


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

    def fill_box(self, box: Box, state: VoxelState) -> None:
        """Set every voxel whose center lies in [box.min, box.max)."""
        slices = []
        for lo, hi, o, n in zip(box.min, box.max, self.origin, self.shape):
            i0 = math.ceil((lo - o) / self.resolution - 0.5 - _EPS)
            i1 = math.ceil((hi - o) / self.resolution - 0.5 - _EPS)
            slices.append(slice(max(i0, 0), min(i1, n)))
        self.states[tuple(slices)] = int(state)


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
        """Find a room by id or by name (case-insensitive)."""
        if ref in self.rooms:
            return self.rooms[ref]
        for room in self.rooms.values():
            if room.name.lower() == ref.lower() or room.id.lower() == ref.lower():
                return room
        raise KeyError(f"unknown room {ref!r}")

    def room_at(self, p: Point) -> Room | None:
        for room in self.rooms.values():
            if room.bounds.contains(p):
                return room
        return None

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
