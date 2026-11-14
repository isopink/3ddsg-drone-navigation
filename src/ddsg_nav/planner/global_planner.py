"""Global planner: shortest route over the room connectivity graph."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from ..graph.model import SceneGraph


class NoRoomRouteError(RuntimeError):
    """No chain of doors connects the two rooms."""


@dataclass(frozen=True)
class RoomRoute:
    rooms: list[str]  # room ids from start to goal
    doors: list[str]  # door ids passed through, len(rooms) - 1


def plan_rooms(graph: SceneGraph, start: str, goal: str) -> RoomRoute:
    start_id, goal_id = graph.find_room(start).id, graph.find_room(goal).id
    g = graph.room_graph()
    try:
        rooms = nx.shortest_path(g, start_id, goal_id, weight="weight")
    except nx.NetworkXNoPath as e:
        raise NoRoomRouteError(f"no route from {start_id} to {goal_id}") from e
    doors = [g.edges[a, b]["door"] for a, b in zip(rooms, rooms[1:])]
    return RoomRoute(rooms=list(rooms), doors=doors)
