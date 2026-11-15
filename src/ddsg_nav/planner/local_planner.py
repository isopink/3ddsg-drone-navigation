"""Local planner: A* on the voxel grid with obstacle inflation. Axis-aligned moves only.

Unobserved (UNKNOWN) voxels are treated as obstacles by default.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from ..graph.model import Point, VoxelGrid, VoxelState


class NoPathError(RuntimeError):
    """No collision-free path exists between the two points."""


def _dilate(blocked: np.ndarray, offsets) -> np.ndarray:
    out = blocked.copy()
    for dx, dy, dz in offsets:
        dst, src = [], []
        for d, n in zip((dx, dy, dz), blocked.shape):
            dst.append(slice(max(0, d), n + min(0, d)))
            src.append(slice(max(0, -d), n - max(0, d)))
        out[tuple(dst)] |= blocked[tuple(src)]
    return out


def inflate(blocked: np.ndarray, radius_voxels: int) -> np.ndarray:
    """Dilate a boolean mask by a sphere of `radius_voxels` (center-to-center distance)."""
    if radius_voxels <= 0:
        return blocked.copy()
    r = radius_voxels
    return _dilate(
        blocked,
        [
            d
            for d in itertools.product(range(-r, r + 1), repeat=3)
            if d != (0, 0, 0) and d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r
        ],
    )


def margin_offsets(inflation_m: float, resolution: float) -> list[tuple[int, int, int]]:
    """Voxel offsets whose center is closer than `inflation_m` to the surface of an
    obstacle voxel at the origin.

    The distance from a voxel center at offset d to the obstacle voxel's cube is
    resolution * |max(|d| - 0.5, 0)| per axis. Along an axis this gives the same radius as
    margin_voxels(); around a convex corner (a wall end, a door frame) it also blocks the
    diagonal voxels that a center-to-center sphere misses, where the margin would
    otherwise shrink to about 0.16 m for a 0.25 m margin at 0.1 m voxels.
    """
    r = math.ceil(inflation_m / resolution + 0.5)
    limit = (inflation_m / resolution) ** 2 - 1e-9
    out = []
    for d in itertools.product(range(-r, r + 1), repeat=3):
        if d == (0, 0, 0):
            continue
        if sum(max(abs(c) - 0.5, 0.0) ** 2 for c in d) < limit:
            out.append(d)
    return out


def inflate_by_distance(blocked: np.ndarray, inflation_m: float, resolution: float) -> np.ndarray:
    """Block every voxel whose center is closer than `inflation_m` to an obstacle surface."""
    if inflation_m <= 0:
        return blocked.copy()
    return _dilate(blocked, margin_offsets(inflation_m, resolution))


def margin_voxels(inflation_m: float, resolution: float) -> int:
    """Smallest inflation radius (in voxels) that keeps a free voxel center at least
    `inflation_m` from an obstacle surface.

    A free voxel center that is r voxels from an obstacle voxel center is (r + 0.5) *
    resolution from its surface, so r >= inflation_m / resolution - 0.5. (Rounding
    inflation_m / resolution up instead would give 0.35 m for a 0.25 m margin at 0.1 m
    voxels and 0.4 m at 0.2 m, which closes 1 m wide doors on coarse grids.)
    """
    return max(0, math.ceil(inflation_m / resolution - 0.5 - 1e-9))


def blocked_mask(
    grid: VoxelGrid, inflation_m: float = 0.25, unknown_is_obstacle: bool = True
) -> np.ndarray:
    blocked = grid.states == int(VoxelState.OCCUPIED)
    if unknown_is_obstacle:
        blocked |= grid.states == int(VoxelState.UNKNOWN)
    return inflate_by_distance(blocked, inflation_m, grid.resolution)


_MOVES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
TURN_COST = 0.5  # in voxels: prefer long straight runs over staircase zigzags


def _astar(blocked: np.ndarray, start: tuple, goal: tuple) -> list[tuple]:
    """A* over the 6 axis moves (no diagonals), with a small cost per change of direction."""
    shape = blocked.shape
    counter = itertools.count()
    first = (start, -1)
    open_heap = [(0.0, next(counter), first)]
    g_score = {first: 0.0}
    parent: dict[tuple, tuple] = {}
    closed: set[tuple] = set()
    while open_heap:
        _, _, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        cur, d = state
        if cur == goal:
            path = [cur]
            while state in parent:
                state = parent[state]
                path.append(state[0])
            return path[::-1]
        closed.add(state)
        for k, (dx, dy, dz) in enumerate(_MOVES):
            nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
            if not all(0 <= i < n for i, n in zip(nxt, shape)) or blocked[nxt]:
                continue
            ns = (nxt, k)
            if ns in closed:
                continue
            ng = g_score[state] + 1.0 + (TURN_COST if d not in (-1, k) else 0.0)
            if ng < g_score.get(ns, math.inf):
                g_score[ns] = ng
                parent[ns] = state
                h = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1]) + abs(nxt[2] - goal[2])
                heapq.heappush(open_heap, (ng + h, next(counter), ns))
    raise NoPathError("no collision-free path")


def _corners(grid: VoxelGrid, cells: list[tuple], start: Point, goal: Point) -> list[Point]:
    """Waypoints at the corners of an axis-aligned cell path, every segment along one axis.

    The path starts and ends at the exact `start` and `goal`. Their offsets from the voxel
    centers (under half a voxel) are absorbed on the axes where they apply, so no segment
    becomes diagonal.
    """
    axis_of = []  # axis of each step, None for a repeated cell
    for a, b in zip(cells, cells[1:]):
        axis_of.append(next(i for i in range(3) if a[i] != b[i]))
    last_move = {}
    for k, ax in enumerate(axis_of):
        last_move[ax] = k  # last step along each axis
    cur = list(start)
    out: list[Point] = [tuple(cur)]
    for k, ax in enumerate(axis_of):
        if k + 1 < len(axis_of) and axis_of[k + 1] == ax:
            continue  # same direction continues: not a corner
        cur[ax] = goal[ax] if last_move[ax] == k else grid.center(cells[k + 1])[ax]
        out.append(tuple(cur))
    for ax in range(3):  # leftover sub-voxel offsets, one axis at a time
        if cur[ax] != goal[ax]:
            cur[ax] = goal[ax]
            out.append(tuple(cur))
    return out


def plan_voxel_path(
    grid: VoxelGrid,
    start: Point,
    goal: Point,
    inflation_m: float = 0.25,
    unknown_is_obstacle: bool = True,
    blocked: np.ndarray | None = None,
) -> list[Point]:
    """Plan a collision-free path from `start` to `goal` (meters). Returns the waypoints.

    Every segment runs along one axis (x, y or z): the path never goes diagonally.
    """
    if blocked is None:
        blocked = blocked_mask(grid, inflation_m, unknown_is_obstacle)
    s_idx, g_idx = grid.index(start), grid.index(goal)
    for name, idx in (("start", s_idx), ("goal", g_idx)):
        if not grid.in_bounds(idx):
            raise NoPathError(f"{name} {idx} is outside the grid")
        if blocked[idx]:
            raise NoPathError(f"{name} {idx} is inside an obstacle or its safety margin")
    return _corners(grid, _astar(blocked, s_idx, g_idx), start, goal)
