"""How unobserved (UNKNOWN) voxels are treated when planning and checking paths.

OBSTACLE        unknown is blocked. Safest; with real, partly observed data this can
                make most routes impossible.
FREE            unknown is flyable. Only for tests and for fully trusted maps.
FREE_IN_ROOMS   unknown inside a room's footprint is flyable, unknown elsewhere stays
                blocked. Assumes the drawing's walls are right and the room interior is
                empty; it can hide furniture or people that were never observed.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from ..graph.model import SceneGraph, VoxelState
from .local_planner import inflate, margin_voxels


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


def raw_obstacles(graph: SceneGraph, policy: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE) -> np.ndarray:
    """Boolean mask of voxels that count as obstacles (not inflated)."""
    policy = UnknownPolicy.coerce(policy)
    states = graph.voxels.states
    occupied = states == int(VoxelState.OCCUPIED)
    unknown = states == int(VoxelState.UNKNOWN)
    if policy is UnknownPolicy.OBSTACLE:
        return occupied | unknown
    if policy is UnknownPolicy.FREE:
        return occupied
    in_rooms = np.zeros(states.shape, dtype=bool)
    for room in graph.rooms.values():
        in_rooms |= graph.room_mask(room)
    return occupied | (unknown & ~in_rooms)


def blocked_mask(
    graph: SceneGraph,
    inflation_m: float = 0.25,
    policy: "bool | str | UnknownPolicy" = UnknownPolicy.OBSTACLE,
) -> np.ndarray:
    """Obstacles inflated by the safety margin: the voxels the drone center must avoid."""
    raw = raw_obstacles(graph, policy)
    return inflate(raw, margin_voxels(inflation_m, graph.voxels.resolution))
