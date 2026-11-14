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
    """The room's center at `height`, or the closest free point inside the room if it is blocked."""
    center = (room.center[0], room.center[1], height)
    if not blocked[grid.index(center)]:
        return center
    lo, hi = room.bounds.min, room.bounds.max
    i0, j0, _ = grid.index((lo[0], lo[1], height))
    i1, j1, k = grid.index((hi[0], hi[1], height))
    best: tuple[float, Point] | None = None
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if not grid.in_bounds((i, j, k)) or blocked[i, j, k]:
                continue
            c = grid.center((i, j, k))
            p = (c[0], c[1], height)
            d = math.dist(p, center)
            if best is None or d < best[0]:
                best = (d, p)
    if best is None:
        raise NoPathError(f"no free space in room {room.id} at height {height} m")
    return best[1]


def plan_route(
    graph: SceneGraph,
    start: str,
    goal: str,
    height: float = 0.5,
    inflation_m: float = 0.25,
    unknown_is_obstacle: "bool | str | UnknownPolicy" = True,
) -> RoutePlan:
    """Plan from room `start` to room `goal` at `height`.

    Starts/ends at the room centers, or at the closest free point if a center is blocked.
    `unknown_is_obstacle` is a bool or an UnknownPolicy (see planner/policy.py).
    """
    route: RoomRoute = plan_rooms(graph, start, goal)
    blocked = blocked_mask(graph, inflation_m, unknown_is_obstacle)

    def at_height(p: Point) -> Point:
        return (p[0], p[1], height)

    keypoints = [nearest_free_point(graph.voxels, blocked, graph.rooms[route.rooms[0]], height)]
    keypoints += [at_height(graph.doors[d].position) for d in route.doors]
    keypoints.append(
        nearest_free_point(graph.voxels, blocked, graph.rooms[route.rooms[-1]], height)
    )

    waypoints: list[Point] = [keypoints[0]]
    for a, b in zip(keypoints, keypoints[1:]):
        leg = plan_voxel_path(graph.voxels, a, b, blocked=blocked)
        waypoints.extend(leg[1:])
    return RoutePlan(route.rooms, route.doors, keypoints, waypoints)
