"""Full route: global room route + local voxel paths between rooms."""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from ..graph.model import Point, Room, SceneGraph, VoxelGrid
from .global_planner import RoomRoute, plan_rooms
from .local_planner import NoPathError, plan_voxel_path
from .policy import UnknownPolicy, blocked_mask


@dataclass(frozen=True)
class RoutePlan:
    rooms: list[str]
    doors: list[str]
    keypoints: list[Point]  # start, door centers, goal (at flight height)
    waypoints: list[Point]  # collision-free path through all keypoints


def nearest_free_point(
    grid: VoxelGrid, blocked: np.ndarray, room: Room, height: float
) -> Point:
    """The room's center at `height`, or the closest free point inside the room if the
    center is blocked or lies outside the room (the centroid of an L-shaped hallway does).

    If the room has no free space at `height` (a stairwell: the steps fill it), the
    lowest higher layer with free space is used instead.
    """
    center = (room.center[0], room.center[1], height)
    c_idx = grid.index(center)
    if room.contains(center) and grid.in_bounds(c_idx) and not blocked[c_idx]:
        return center
    lo, hi = room.bounds.min, room.bounds.max
    i0, j0, k0 = grid.index((lo[0], lo[1], height))
    i1, j1, _ = grid.index((hi[0], hi[1], height))
    k_top = min(grid.index((lo[0], lo[1], hi[2]))[2], grid.shape[2])
    for k in range(k0, max(k0 + 1, k_top)):
        z = height if k == k0 else grid.center((0, 0, k))[2]
        best: tuple[float, Point] | None = None
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if not grid.in_bounds((i, j, k)) or blocked[i, j, k]:
                    continue
                c = grid.center((i, j, k))
                p = (c[0], c[1], z)
                if not room.contains(p):
                    continue
                d = math.dist(p[:2], center[:2])
                if best is None or d < best[0]:
                    best = (d, p)
        if best is not None:
            return best[1]
    raise NoPathError(f"no free space in room {room.id} at or above height {height} m")


def plan_route(
    graph: SceneGraph,
    start: str,
    goal: str,
    height: float = 0.5,
    inflation_m: float = 0.25,
    unknown_is_obstacle: "bool | str | UnknownPolicy" = True,
) -> RoutePlan:
    """Plan from room `start` to room `goal`, `height` above the floor.

    `height` is measured from each room's floor (and, at a door, from the higher floor of
    its two rooms), so a route through a stairwell climbs to the upper floor. On a single
    floor at z = 0 this is the same as an absolute height.
    Starts/ends at the room centers, or at the closest free point if a center is blocked.
    `unknown_is_obstacle` is a bool or an UnknownPolicy (see planner/policy.py).
    """
    route: RoomRoute = plan_rooms(graph, start, goal)
    blocked = blocked_mask(graph, inflation_m, unknown_is_obstacle)

    def room_point(rid: str) -> Point:
        room = graph.rooms[rid]
        return nearest_free_point(graph.voxels, blocked, room, graph.floor_z(room) + height)

    def door_point(did: str) -> Point:
        door = graph.doors[did]
        return (door.position[0], door.position[1], graph.door_floor_z(door) + height)

    keypoints = [room_point(route.rooms[0])]
    keypoints += [door_point(d) for d in route.doors]
    keypoints.append(room_point(route.rooms[-1]))

    waypoints: list[Point] = [keypoints[0]]
    for a, b in zip(keypoints, keypoints[1:]):
        leg = plan_voxel_path(graph.voxels, a, b, blocked=blocked)
        waypoints.extend(leg[1:])
    return RoutePlan(route.rooms, route.doors, keypoints, waypoints)
