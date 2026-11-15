"""Collision checks: path points vs the voxel map and its inflated safety margin.

Three kinds of violations:
- `hard`: a point inside an occupied voxel, or an unknown one that the policy blocks
  (`unknown_is_obstacle` is a bool, an UnknownPolicy or an ElementPolicy)
- `margin`: a point closer to an obstacle than the safety margin
- `outside`: a point outside the voxel grid
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..graph.model import Point, SceneGraph
from .policy import PolicyLike, blocked_mask, raw_obstacles


@dataclass
class CollisionReport:
    samples: int = 0
    hard: list[Point] = field(default_factory=list)
    margin: list[Point] = field(default_factory=list)
    outside: list[Point] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.hard or self.margin or self.outside)

    @property
    def collision_free(self) -> bool:
        """No point inside an obstacle or outside the map (margin violations allowed)."""
        return not (self.hard or self.outside)

    def summary(self) -> str:
        return (
            f"{self.samples} samples: {len(self.hard)} hard, "
            f"{len(self.margin)} margin, {len(self.outside)} outside"
        )


def check_points(
    graph: SceneGraph,
    points: Iterable[Point],
    inflation_m: float = 0.25,
    unknown_is_obstacle: PolicyLike = True,
) -> CollisionReport:
    grid = graph.voxels
    raw = raw_obstacles(graph, unknown_is_obstacle)
    inflated = blocked_mask(graph, inflation_m, unknown_is_obstacle)
    report = CollisionReport()
    for p in points:
        report.samples += 1
        idx = grid.index(p)
        if not grid.in_bounds(idx):
            report.outside.append(p)
        elif raw[idx]:
            report.hard.append(p)
        elif inflated[idx]:
            report.margin.append(p)
    return report


def sample_path(waypoints: Sequence[Point], step: float) -> list[Point]:
    """Points along the polyline through `waypoints`, at most `step` meters apart."""
    pts: list[Point] = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        n = max(1, math.ceil(math.dist(a, b) / step))
        for i in range(1, n + 1):
            t = i / n
            pts.append(tuple(x + (y - x) * t for x, y in zip(a, b)))  # type: ignore[arg-type]
    return pts


def check_path(
    graph: SceneGraph,
    waypoints: Sequence[Point],
    inflation_m: float = 0.25,
    unknown_is_obstacle: PolicyLike = True,
    step: float | None = None,
) -> CollisionReport:
    """Check the straight segments between waypoints (sampled every half voxel by default)."""
    step = step or graph.voxels.resolution / 2
    return check_points(graph, sample_path(waypoints, step), inflation_m, unknown_is_obstacle)


def check_trajectory(
    graph: SceneGraph, trajectory: np.ndarray, inflation_m: float = 0.25
) -> CollisionReport:
    """Check flown positions, an (N, 3) array. Use for simulator or real flight logs."""
    return check_points(graph, [tuple(float(c) for c in row) for row in trajectory], inflation_m)
