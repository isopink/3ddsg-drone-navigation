"""Planned path vs flown trajectory: deviation metrics and CSV storage."""
from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..graph.model import Point


def distance_to_segment(p: Point, a: Point, b: Point) -> float:
    pa = np.subtract(p, a)
    ab = np.subtract(b, a)
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip(pa @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(pa - t * ab))


def distance_to_polyline(p: Point, polyline: Sequence[Point]) -> float:
    if len(polyline) == 1:
        return math.dist(p, polyline[0])
    return min(distance_to_segment(p, a, b) for a, b in zip(polyline, polyline[1:]))


def polyline_length(points: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


@dataclass(frozen=True)
class PlanComparison:
    samples: int
    max_deviation_m: float
    mean_deviation_m: float
    end_error_m: float  # horizontal distance between the last flown point and the goal
    planned_length_m: float
    flown_length_m: float
    duration_s: float

    def format(self) -> str:
        return (
            f"{self.samples} samples, {self.duration_s:.1f} s\n"
            f"max deviation from the plan : {self.max_deviation_m:.2f} m\n"
            f"mean deviation              : {self.mean_deviation_m:.2f} m\n"
            f"end error at the goal (xy)  : {self.end_error_m:.2f} m\n"
            f"path length planned / flown : {self.planned_length_m:.1f} m / {self.flown_length_m:.1f} m"
        )


def compare_to_plan(
    waypoints: Sequence[Point], trajectory: Sequence[tuple[float, float, float, float]]
) -> PlanComparison:
    """`trajectory` rows are (t, x, y, z) in graph coordinates.

    Only the part flown at flight height counts (take-off and landing are vertical and
    would only add noise): samples below 80% of the planned height are skipped.
    """
    if not waypoints or not trajectory:
        raise ValueError("need a plan and a trajectory")
    rows = np.asarray(trajectory, dtype=float)
    height = min(w[2] for w in waypoints)
    flying = rows[rows[:, 3] >= 0.8 * height]
    if len(flying) == 0:
        flying = rows
    pts = [tuple(r[1:4]) for r in flying]
    devs = [distance_to_polyline(p, waypoints) for p in pts]  # type: ignore[arg-type]
    goal = waypoints[-1]
    end = pts[-1]
    return PlanComparison(
        samples=len(flying),
        max_deviation_m=float(max(devs)),
        mean_deviation_m=float(np.mean(devs)),
        end_error_m=math.hypot(end[0] - goal[0], end[1] - goal[1]),
        planned_length_m=polyline_length(waypoints),
        flown_length_m=polyline_length(pts),  # type: ignore[arg-type]
        duration_s=float(rows[-1, 0] - rows[0, 0]),
    )


def save_trajectory(path: str | Path, trajectory: Sequence[tuple[float, float, float, float]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "z"])
        w.writerows(trajectory)
    return path


def load_trajectory(path: str | Path) -> list[tuple[float, float, float, float]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return [tuple(float(v) for v in r) for r in rows[1:]]  # type: ignore[misc]
