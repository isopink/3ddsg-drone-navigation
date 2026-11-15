"""Local planner: A* on the voxel grid with obstacle inflation.

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


def _intermediate_cells(d: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Cells a diagonal move squeezes past. They must be free too, otherwise the move
    would cut the corner of an obstacle (keeping the safety margin only on paper)."""
    axes = [i for i in range(3) if d[i] != 0]
    out = []
    for mask in range(1, (1 << len(axes)) - 1):
        v = [0, 0, 0]
        for bit, axis in enumerate(axes):
            if mask >> bit & 1:
                v[axis] = d[axis]
        out.append((v[0], v[1], v[2]))
    return out


_NEIGHBORS = [
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz), _intermediate_cells((dx, dy, dz)))
    for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
    if (dx, dy, dz) != (0, 0, 0)
]


def _astar(blocked: np.ndarray, start: tuple, goal: tuple) -> list[tuple]:
    shape = blocked.shape
    counter = itertools.count()
    open_heap = [(0.0, next(counter), start)]
    g_score = {start: 0.0}
    parent: dict[tuple, tuple] = {}
    closed: set[tuple] = set()
    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        for dx, dy, dz, cost, via in _NEIGHBORS:
            nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
            if not all(0 <= i < n for i, n in zip(nxt, shape)) or blocked[nxt] or nxt in closed:
                continue
            if via and any(blocked[cur[0] + a, cur[1] + b, cur[2] + c] for a, b, c in via):
                continue
            ng = g_score[cur] + cost
            if ng < g_score.get(nxt, math.inf):
                g_score[nxt] = ng
                parent[nxt] = cur
                h = math.dist(nxt, goal)
                heapq.heappush(open_heap, (ng + h, next(counter), nxt))
    raise NoPathError("no collision-free path")


def _segment_free(grid: VoxelGrid, blocked: np.ndarray, a: Point, b: Point) -> bool:
    """True if the straight segment a-b only passes through free voxels. Sampled every
    1/8 voxel (vectorized), so it cannot skip the corner of a blocked voxel by more than
    that."""
    n = max(1, math.ceil(math.dist(a, b) / (grid.resolution / 8)))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = np.asarray(a, dtype=float) + (np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) * t
    idx = np.floor((pts - np.asarray(grid.origin)) / grid.resolution + 1e-9).astype(int)
    shape = np.asarray(blocked.shape)
    if (idx < 0).any() or (idx >= shape).any():
        return False
    return not blocked[idx[:, 0], idx[:, 1], idx[:, 2]].any()


def _simplify(grid: VoxelGrid, blocked: np.ndarray, pts: list[Point]) -> list[Point]:
    """Greedy line-of-sight shortcut ("string pulling")."""
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _segment_free(grid, blocked, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def plan_voxel_path(
    grid: VoxelGrid,
    start: Point,
    goal: Point,
    inflation_m: float = 0.25,
    unknown_is_obstacle: bool = True,
    simplify: bool = True,
    blocked: np.ndarray | None = None,
) -> list[Point]:
    """Plan a collision-free path from `start` to `goal` (meters). Returns waypoints."""
    if blocked is None:
        blocked = blocked_mask(grid, inflation_m, unknown_is_obstacle)
    s_idx, g_idx = grid.index(start), grid.index(goal)
    for name, idx in (("start", s_idx), ("goal", g_idx)):
        if not grid.in_bounds(idx):
            raise NoPathError(f"{name} {idx} is outside the grid")
        if blocked[idx]:
            raise NoPathError(f"{name} {idx} is inside an obstacle or its safety margin")
    cells = _astar(blocked, s_idx, g_idx)
    pts: list[Point] = [grid.center(c) for c in cells]
    pts[0], pts[-1] = start, goal  # exact endpoints
    return _simplify(grid, blocked, pts) if simplify else pts
