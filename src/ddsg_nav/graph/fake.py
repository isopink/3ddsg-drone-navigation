"""Fake graph for development: 3 rooms in a row (A - B - C) connected by 2 doors.

Layout (meters, top view, x to the right):

    y=3 +-------+-------+-------+
        |   A   |   B   |   C   |
    y=0 +-------+-------+-------+
        x=0     3       6       9      doors at x=3 and x=6 (y=1..2, height 2 m)
"""
from __future__ import annotations

from .model import Box, Door, Room, SceneGraph, VoxelGrid, VoxelState

ROOM_SIZE = 3.0
HEIGHT = 2.5
WALL = 0.1  # outer wall thickness
PARTITION = 0.2  # inner wall thickness
DOOR_WIDTH = 1.0
DOOR_HEIGHT = 2.0


def build_fake_graph(resolution: float = 0.1, obstacle: bool = False) -> SceneGraph:
    """Build the 3-room fake graph. `obstacle=True` adds a box in the middle of room B."""
    width = 3 * ROOM_SIZE
    grid = VoxelGrid.empty(
        origin=(-WALL, -WALL, 0.0),
        size=(width + 2 * WALL, ROOM_SIZE + 2 * WALL, HEIGHT),
        resolution=resolution,
        state=VoxelState.FREE,
    )
    graph = SceneGraph(voxels=grid)

    for i, rid in enumerate("ABC"):
        x0 = i * ROOM_SIZE
        graph.add_room(
            Room(rid, f"Room {rid}", Box((x0, 0.0, 0.0), (x0 + ROOM_SIZE, ROOM_SIZE, HEIGHT)))
        )

    # outer walls
    graph.add_solid(Box((-WALL, -WALL, 0.0), (width + WALL, 0.0, HEIGHT)))  # south
    graph.add_solid(Box((-WALL, ROOM_SIZE, 0.0), (width + WALL, ROOM_SIZE + WALL, HEIGHT)))  # north
    graph.add_solid(Box((-WALL, 0.0, 0.0), (0.0, ROOM_SIZE, HEIGHT)))  # west
    graph.add_solid(Box((width, 0.0, 0.0), (width + WALL, ROOM_SIZE, HEIGHT)))  # east

    # partitions with a door opening each
    y_lo = (ROOM_SIZE - DOOR_WIDTH) / 2
    y_hi = y_lo + DOOR_WIDTH
    for i, (a, b) in enumerate([("A", "B"), ("B", "C")], start=1):
        x = i * ROOM_SIZE
        x0, x1 = x - PARTITION / 2, x + PARTITION / 2
        graph.add_solid(Box((x0, 0.0, 0.0), (x1, y_lo, HEIGHT)))
        graph.add_solid(Box((x0, y_hi, 0.0), (x1, ROOM_SIZE, HEIGHT)))
        graph.add_solid(Box((x0, y_lo, DOOR_HEIGHT), (x1, y_hi, HEIGHT)))  # lintel
        graph.add_door(
            Door(f"door_{a}{b}", a, b, (x, ROOM_SIZE / 2, DOOR_HEIGHT / 2), DOOR_WIDTH)
        )

    if obstacle:
        cx, cy = 1.5 * ROOM_SIZE, ROOM_SIZE / 2
        graph.add_solid(Box((cx - 0.2, cy - 0.2, 0.0), (cx + 0.2, cy + 0.2, 1.0)))

    return graph
