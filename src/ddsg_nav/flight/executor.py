"""Execute a planned path on any DroneBackend."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..graph.model import Point
from .backend import DroneBackend

MAX_VELOCITY = 0.3  # m/s, safety cap (Step 3 requirement)


def follow_path(
    backend: DroneBackend,
    waypoints: Sequence[Point],
    velocity: float = 0.2,
    max_velocity: float = MAX_VELOCITY,
    segment_pause: float = 0.0,
    on_waypoint: Callable[[int, Point], None] | None = None,
) -> None:
    """Take off at the first waypoint's height, fly through all waypoints, land.

    Always lands, also when something goes wrong. Speed is capped at `max_velocity`.
    `segment_pause` stops for that many seconds after each waypoint (use ~0.5 on a real drone).
    """
    if not waypoints:
        raise ValueError("no waypoints")
    velocity = min(velocity, max_velocity)
    try:
        backend.takeoff(waypoints[0][2])
        for i, wp in enumerate(waypoints):
            backend.go_to(wp[0], wp[1], wp[2], velocity)
            if on_waypoint is not None:
                on_waypoint(i, wp)
            if segment_pause > 0:
                time.sleep(segment_pause)
    finally:
        backend.land()
