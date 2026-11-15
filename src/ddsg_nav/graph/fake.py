"""Fake graph for development: 3 rooms in a row (A - B - C) connected by 2 doors.

Layout (meters, top view, x to the right):

    y=3 +-------+-------+-------+
        |   A   |   B   |   C   |
    y=0 +-------+-------+-------+
        x=0     3       6       9      doors at x=3 and x=6 (y=1..2, height 2 m)
"""
from __future__ import annotations

from collections.abc import Sequence

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


def build_row_graph(
    room_sizes: Sequence[float] = (1.5, 1.5, 1.5),
    depth: float = 1.5,
    door_width: float = 0.8,
    height: float = HEIGHT,
    resolution: float = 0.1,
) -> SceneGraph:
    """Virtual rooms in a row, sized to the space you can really fly in.

    `room_sizes` are the widths (x) of the rooms, `depth` their size in y. Rooms are named
    A, B, C ... and every shared wall has a door of `door_width`. The walls only exist in
    the map, nothing is built: the drone flies in an empty space and the planner keeps it
    inside the virtual rooms and routes it through the virtual doors.

    The usable door width is `door_width` minus twice the safety margin (0.25 m), so
    keep doors at least 0.8 m wide, and leave room for the Flow deck's position drift.
    """
    if not room_sizes or len(room_sizes) > 26:
        raise ValueError("need 1 to 26 rooms")
    width = sum(room_sizes)
    grid = VoxelGrid.empty(
        origin=(-WALL, -WALL, 0.0),
        size=(width + 2 * WALL, depth + 2 * WALL, height),
        resolution=resolution,
        state=VoxelState.FREE,
    )
    graph = SceneGraph(voxels=grid)
    ids = [chr(ord("A") + i) for i in range(len(room_sizes))]

    x0 = 0.0
    edges = []
    for rid, w in zip(ids, room_sizes):
        graph.add_room(Room(rid, f"Room {rid}", Box((x0, 0.0, 0.0), (x0 + w, depth, height))))
        x0 += w
        edges.append(x0)

    graph.add_solid(Box((-WALL, -WALL, 0.0), (width + WALL, 0.0, height)))
    graph.add_solid(Box((-WALL, depth, 0.0), (width + WALL, depth + WALL, height)))
    graph.add_solid(Box((-WALL, 0.0, 0.0), (0.0, depth, height)))
    graph.add_solid(Box((width, 0.0, 0.0), (width + WALL, depth, height)))

    yc, half, t = depth / 2, door_width / 2, PARTITION / 2
    for i, x in enumerate(edges[:-1]):
        graph.add_solid(Box((x - t, 0.0, 0.0), (x + t, yc - half, height)))
        graph.add_solid(Box((x - t, yc + half, 0.0), (x + t, depth, height)))
        graph.add_solid(Box((x - t, yc - half, DOOR_HEIGHT), (x + t, yc + half, height)))
        a, b = ids[i], ids[i + 1]
        graph.add_door(Door(f"door_{a}{b}", a, b, (x, yc, DOOR_HEIGHT / 2), door_width))
    return graph


def build_large_graph(
    rooms_x: int = 6,
    rooms_y: int = 4,
    room_size: float = 5.0,
    resolution: float = 0.1,
    height: float = HEIGHT,
) -> SceneGraph:
    """A rooms_x by rooms_y block of square rooms, a door in every shared wall.

    Default is 30 m x 20 m (1.5 million voxels at 0.1 m): a rough size for one real floor,
    used to check planning speed. Room ids are "R<ix>_<iy>".
    """
    width, depth = rooms_x * room_size, rooms_y * room_size
    grid = VoxelGrid.empty(
        origin=(-WALL, -WALL, 0.0),
        size=(width + 2 * WALL, depth + 2 * WALL, height),
        resolution=resolution,
        state=VoxelState.FREE,
    )
    graph = SceneGraph(voxels=grid)
    rid = lambda ix, iy: f"R{ix}_{iy}"  # noqa: E731

    for ix in range(rooms_x):
        for iy in range(rooms_y):
            x0, y0 = ix * room_size, iy * room_size
            graph.add_room(
                Room(rid(ix, iy), f"Room {ix}-{iy}",
                     Box((x0, y0, 0.0), (x0 + room_size, y0 + room_size, height)))
            )

    graph.add_solid(Box((-WALL, -WALL, 0.0), (width + WALL, 0.0, height)))
    graph.add_solid(Box((-WALL, depth, 0.0), (width + WALL, depth + WALL, height)))
    graph.add_solid(Box((-WALL, 0.0, 0.0), (0.0, depth, height)))
    graph.add_solid(Box((width, 0.0, 0.0), (width + WALL, depth, height)))

    half, t = DOOR_WIDTH / 2, PARTITION / 2
    for ix in range(1, rooms_x):  # walls between neighbours in x
        x = ix * room_size
        for iy in range(rooms_y):
            y0, y1 = iy * room_size, (iy + 1) * room_size
            yc = (y0 + y1) / 2
            graph.add_solid(Box((x - t, y0, 0.0), (x + t, yc - half, height)))
            graph.add_solid(Box((x - t, yc + half, 0.0), (x + t, y1, height)))
            graph.add_solid(Box((x - t, yc - half, DOOR_HEIGHT), (x + t, yc + half, height)))
            a, b = rid(ix - 1, iy), rid(ix, iy)
            graph.add_door(Door(f"door_{a}__{b}", a, b, (x, yc, DOOR_HEIGHT / 2), DOOR_WIDTH))
    for iy in range(1, rooms_y):  # walls between neighbours in y
        y = iy * room_size
        for ix in range(rooms_x):
            x0, x1 = ix * room_size, (ix + 1) * room_size
            xc = (x0 + x1) / 2
            graph.add_solid(Box((x0, y - t, 0.0), (xc - half, y + t, height)))
            graph.add_solid(Box((xc + half, y - t, 0.0), (x1, y + t, height)))
            graph.add_solid(Box((xc - half, y - t, DOOR_HEIGHT), (xc + half, y + t, height)))
            a, b = rid(ix, iy - 1), rid(ix, iy)
            graph.add_door(Door(f"door_{a}__{b}", a, b, (xc, y, DOOR_HEIGHT / 2), DOOR_WIDTH))
    return graph


def build_corridor_graph(resolution: float = 0.1, height: float = HEIGHT) -> SceneGraph:
    """Room A -> L-shaped hallway H -> room C: a route with two turns and two doors.

    Layout (meters, top view, x to the right, y up):

        y=10.2      +--------+
                    |   C    |
        y=7.2       +--[ ]---+         door H-C at y=7.1, x=6.9..7.9
        y=7.0          |  |
                       |H |            hallway: 1.2 m wide, L-shaped
        y=2.2  ---+----+  |
        A    [ ]   H      |            door A-H at x=3.1, y=1.1..2.1
        y=1.0  ---+-------+
              x=3.2  x=6.8 x=8.0

    The hallway is one room with an L-shaped polygon. Walls are 0.2 m thick, doors 1 m
    wide and 2 m high. Room ids: "A", "H" (name "Hallway", alias "복도"), "C".
    """
    t = PARTITION
    grid = VoxelGrid.empty(
        origin=(-t, -t, 0.0),
        size=(9.1 + 2 * t, 10.2 + 2 * t, height),
        resolution=resolution,
        state=VoxelState.FREE,
    )
    graph = SceneGraph(voxels=grid)

    hall = ((3.2, 1.0), (8.0, 1.0), (8.0, 7.0), (6.8, 7.0), (6.8, 2.2), (3.2, 2.2))
    graph.add_room(Room("A", "Room A", Box((0.0, 0.0, 0.0), (3.0, 3.0, height))))
    graph.add_room(
        Room("H", "Hallway", Box((3.2, 1.0, 0.0), (8.0, 7.0, height)), ("복도", "corridor"), hall)
    )
    graph.add_room(Room("C", "Room C", Box((5.9, 7.2, 0.0), (8.9, 10.2, height))))

    def wall(x0: float, y0: float, x1: float, y1: float) -> None:
        graph.add_solid(Box((x0, y0, 0.0), (x1, y1, height)))

    def lintel(x0: float, y0: float, x1: float, y1: float) -> None:
        graph.add_solid(Box((x0, y0, DOOR_HEIGHT), (x1, y1, height)))

    # room A
    wall(-t, -t, 3.0 + t, 0.0)  # south
    wall(-t, 3.0, 3.0 + t, 3.0 + t)  # north
    wall(-t, 0.0, 0.0, 3.0)  # west
    wall(3.0, 0.0, 3.0 + t, 1.1)  # east, below the door
    wall(3.0, 2.1, 3.0 + t, 3.0)  # east, above the door
    lintel(3.0, 1.1, 3.0 + t, 2.1)
    # hallway
    wall(3.2, 1.0 - t, 8.0 + t, 1.0)  # south
    wall(3.2, 2.2, 6.8, 2.2 + t)  # north side of the horizontal leg
    wall(6.8 - t, 2.2, 6.8, 7.0)  # west side of the vertical leg
    wall(8.0, 1.0, 8.0 + t, 7.0)  # east
    # wall between hallway and room C, with the door
    wall(5.9 - t, 7.0, 6.9, 7.2)
    wall(7.9, 7.0, 8.9 + t, 7.2)
    lintel(6.9, 7.0, 7.9, 7.2)
    # room C
    wall(5.9 - t, 10.2, 8.9 + t, 10.2 + t)  # north
    wall(5.9 - t, 7.2, 5.9, 10.2)  # west
    wall(8.9, 7.2, 8.9 + t, 10.2)  # east

    graph.add_door(Door("door_AH", "A", "H", (3.1, 1.6, DOOR_HEIGHT / 2), DOOR_WIDTH))
    graph.add_door(Door("door_HC", "H", "C", (7.4, 7.1, DOOR_HEIGHT / 2), DOOR_WIDTH))
    return graph


def build_stairs_graph(resolution: float = 0.1) -> SceneGraph:
    """Room A (floor 1) -> stairwell S -> room C (floor 2, right above A).

    Side view of the stairwell (x to the right, z up), dog-leg stairs:

        z=3.2  [C door]=  top landing <- north flight (y=1.7..3.0) <-+
                                                                     | landing z=1.6
        z=0    [A door]=  south flight (y=0..1.3) ------------------>+  (x=5.0..6.2)
              x=3.2                                           x=5.0  x=6.2

    A full-height wall (y=1.3..1.7, x=3.2..5.0) separates the two flights, so the drone
    climbs along the south flight, turns on the landing and climbs along the north
    flight to the door of room C. Floors: 1 at z=0, 2 at z=3.2 (0.2 m slab). Doors are
    1 m wide and 2 m high. Room ids: "A", "S" (name "Stairwell", alias "계단"), "C".

    The simulator has no sensor model. On the real drone the Flow deck measures height
    over the steps, not over the floor, so this layout is a planning test only.
    """
    t = PARTITION
    f2 = 3.2  # floor 2 level
    top = 6.2  # ceiling of floor 2
    grid = VoxelGrid.empty(
        origin=(-t, -t, 0.0),
        size=(6.2 + 2 * t, 3.0 + 2 * t, top),
        resolution=resolution,
        state=VoxelState.FREE,
    )
    graph = SceneGraph(voxels=grid)
    graph.add_room(Room("A", "Room A", Box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))))
    graph.add_room(
        Room("S", "Stairwell", Box((3.2, 0.0, 0.0), (6.2, 3.0, top)), ("계단", "계단실", "stairs"))
    )
    graph.add_room(Room("C", "Room C", Box((0.0, 0.0, f2), (3.0, 3.0, top))))

    def solid(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> None:
        graph.add_solid(Box((x0, y0, z0), (x1, y1, z1)))

    # outer walls, full height
    solid(-t, -t, 0.0, 6.2 + t, 0.0, top)  # south
    solid(-t, 3.0, 0.0, 6.2 + t, 3.0 + t, top)  # north
    solid(-t, 0.0, 0.0, 0.0, 3.0, top)  # west
    solid(6.2, 0.0, 0.0, 6.2 + t, 3.0, top)  # east
    # slab between room A and room C
    solid(0.0, 0.0, 3.0, 3.0, 3.0, f2)
    # wall between the rooms and the stairwell (x=3.0..3.2) with one door per floor
    solid(3.0, 0.0, 0.0, 3.2, 0.15, DOOR_HEIGHT)  # floor 1, beside door A-S
    solid(3.0, 1.15, 0.0, 3.2, 3.0, DOOR_HEIGHT)
    solid(3.0, 0.0, DOOR_HEIGHT, 3.2, 3.0, f2)  # lintel and slab level
    solid(3.0, 0.0, f2, 3.2, 1.85, f2 + DOOR_HEIGHT)  # floor 2, beside door S-C
    solid(3.0, 2.85, f2, 3.2, 3.0, f2 + DOOR_HEIGHT)
    solid(3.0, 0.0, f2 + DOOR_HEIGHT, 3.2, 3.0, top)
    # stairwell: wall between the flights, landing, steps (solid down to the floor)
    solid(3.2, 1.3, 0.0, 5.0, 1.7, top)
    solid(5.0, 0.0, 0.0, 6.2, 3.0, 1.6)  # landing
    rise, n = 0.2, 8
    w1 = (5.0 - 3.2) / n
    for i in range(n):  # south flight, climbing in +x
        solid(3.2 + i * w1, 0.0, 0.0, 3.2 + (i + 1) * w1, 1.3, (i + 1) * rise)
    w2 = (5.0 - 3.4) / n
    for j in range(1, n + 1):  # north flight, climbing in -x from the landing
        solid(5.0 - j * w2, 1.7, 0.0, 5.0 - (j - 1) * w2, 3.0, 1.6 + j * rise)
    solid(3.2, 1.7, 0.0, 3.4, 3.0, f2)  # top landing in front of door S-C

    graph.add_door(Door("door_AS", "A", "S", (3.1, 0.65, DOOR_HEIGHT / 2), DOOR_WIDTH))
    graph.add_door(Door("door_SC", "S", "C", (3.1, 2.35, f2 + DOOR_HEIGHT / 2), DOOR_WIDTH))
    return graph
