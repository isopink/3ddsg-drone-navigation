"""How unobserved (UNKNOWN) voxels are treated when planning and checking paths.

OBSTACLE        unknown is blocked. Safest; with real, partly observed data this can
                make most routes impossible.
FREE            unknown is flyable. Only for tests and for fully trusted maps.
FREE_IN_ROOMS   unknown inside a room's footprint is flyable, unknown elsewhere stays
                blocked. Assumes the drawing's walls are right and the room interior is
                empty; it can hide furniture or people that were never observed.

`ElementPolicy` sets OBSTACLE or FREE per element type instead of one policy for the
whole map. The map has no element labels, so the type of an unknown voxel is guessed
from the graph geometry:

door        within half the door width (in x, y) of a door center, over the door height
furniture   inside a room's footprint, and not door
wall        everything else (between rooms, outside the building)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..graph.model import Box, SceneGraph, VoxelState
from .local_planner import inflate_by_distance


class UnknownPolicy(str, Enum):
    OBSTACLE = "obstacle"
    FREE = "free"
    FREE_IN_ROOMS = "free_in_rooms"

    @classmethod
    def coerce(cls, value: "bool | str | UnknownPolicy") -> "UnknownPolicy":
        """Accept the old boolean `unknown_is_obstacle` as well as names and members."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.OBSTACLE if value else cls.FREE
        return cls(value)


@dataclass(frozen=True)
class ElementPolicy:
    """OBSTACLE or FREE for unknown voxels, per element type. The default blocks all
    three, the same as `UnknownPolicy.OBSTACLE`."""

    wall: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE
    door: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE
    furniture: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE

    def __post_init__(self) -> None:
        for name in ("wall", "door", "furniture"):
            value = UnknownPolicy.coerce(getattr(self, name))
            if value is UnknownPolicy.FREE_IN_ROOMS:
                raise ValueError(f"{name}: an element policy must be 'obstacle' or 'free'")
            object.__setattr__(self, name, value)

    @property
    def label(self) -> str:
        return f"wall={self.wall.value},door={self.door.value},furniture={self.furniture.value}"


PolicyLike = "bool | str | UnknownPolicy | ElementPolicy"


def policy_label(policy: PolicyLike) -> str:
    """Short name of a policy, for messages."""
    if isinstance(policy, ElementPolicy):
        return policy.label
    return UnknownPolicy.coerce(policy).value


def rooms_mask(graph: SceneGraph) -> np.ndarray:
    """Voxels inside any room's footprint."""
    mask = np.zeros(graph.voxels.shape, dtype=bool)
    for room in graph.rooms.values():
        mask |= graph.room_mask(room)
    return mask


def doors_mask(graph: SceneGraph) -> np.ndarray:
    """Voxels in a door opening: within half the door width (in x, y) of the door center,
    from the door's floor up to the door height (`position` is the center of the opening)."""
    grid = graph.voxels
    mask = np.zeros(grid.shape, dtype=bool)
    for door in graph.doors.values():
        x, y, z = door.position
        r = door.width / 2
        z0 = graph.door_floor_z(door)
        sx, sy, sz = grid.box_slices(Box((x - r, y - r, z0), (x + r, y + r, 2 * z - z0)))
        xs = grid.origin[0] + (np.arange(sx.start, sx.stop) + 0.5) * grid.resolution
        ys = grid.origin[1] + (np.arange(sy.start, sy.stop) + 0.5) * grid.resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        near = (gx - x) ** 2 + (gy - y) ** 2 <= r * r
        mask[sx, sy, sz] |= near[:, :, None]
    return mask


def raw_obstacles(graph: SceneGraph, policy: PolicyLike = UnknownPolicy.OBSTACLE) -> np.ndarray:
    """Boolean mask of voxels that count as obstacles (not inflated)."""
    states = graph.voxels.states
    occupied = states == int(VoxelState.OCCUPIED)
    unknown = states == int(VoxelState.UNKNOWN)
    if isinstance(policy, ElementPolicy):
        door = doors_mask(graph)
        furniture = rooms_mask(graph) & ~door
        wall = ~(door | furniture)
        blocked = occupied
        for mask, element in ((wall, policy.wall), (door, policy.door), (furniture, policy.furniture)):
            if element is UnknownPolicy.OBSTACLE:
                blocked = blocked | (unknown & mask)
        return blocked
    policy = UnknownPolicy.coerce(policy)
    if policy is UnknownPolicy.OBSTACLE:
        return occupied | unknown
    if policy is UnknownPolicy.FREE:
        return occupied
    return occupied | (unknown & ~rooms_mask(graph))


def blocked_mask(
    graph: SceneGraph,
    inflation_m: float = 0.25,
    policy: PolicyLike = UnknownPolicy.OBSTACLE,
) -> np.ndarray:
    """Obstacles inflated by the safety margin: the voxels the drone center must avoid."""
    raw = raw_obstacles(graph, policy)
    return inflate_by_distance(raw, inflation_m, graph.voxels.resolution)
