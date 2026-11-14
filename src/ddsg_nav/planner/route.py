"""Full route: global room route + local voxel paths between rooms."""
from __future__ import annotations

from dataclasses import dataclass

from ..graph.model import Point, SceneGraph
from .global_planner import RoomRoute, plan_rooms
from .local_planner import blocked_mask, plan_voxel_path


@dataclass(frozen=True)
class RoutePlan:
    rooms: list[str]
    doors: list[str]
    keypoints: list[Point]  # start, door centers, goal (at flight height)
    waypoints: list[Point]  # collision-free path through all keypoints


def plan_route(
    graph: SceneGraph,
    start: str,
    goal: str,
    height: float = 0.5,
    inflation_m: float = 0.25,
    unknown_is_obstacle: bool = True,
) -> RoutePlan:
    """Plan from the center of room `start` to the center of room `goal` at `height`."""
    route: RoomRoute = plan_rooms(graph, start, goal)

    def at_height(p: Point) -> Point:
        return (p[0], p[1], height)

    keypoints = [at_height(graph.rooms[route.rooms[0]].center)]
    keypoints += [at_height(graph.doors[d].position) for d in route.doors]
    keypoints.append(at_height(graph.rooms[route.rooms[-1]].center))

    blocked = blocked_mask(graph.voxels, inflation_m, unknown_is_obstacle)
    waypoints: list[Point] = [keypoints[0]]
    for a, b in zip(keypoints, keypoints[1:]):
        leg = plan_voxel_path(graph.voxels, a, b, blocked=blocked)
        waypoints.extend(leg[1:])
    return RoutePlan(route.rooms, route.doors, keypoints, waypoints)
