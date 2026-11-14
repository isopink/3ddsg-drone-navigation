"""Execute a planned path on any DroneBackend."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..graph.model import Point
from .backend import DroneBackend
from .monitor import FlightMonitor

MAX_VELOCITY = 0.3  # m/s, safety cap (Step 3 requirement)


def follow_path(
    backend: DroneBackend,
    waypoints: Sequence[Point],
    velocity: float = 0.2,
    max_velocity: float = MAX_VELOCITY,
    segment_pause: float = 0.0,
    on_waypoint: Callable[[int, Point], None] | None = None,
    monitor: FlightMonitor | None = None,
) -> None:
    """Take off at the first waypoint's height, fly through all waypoints, land.

    Always lands, also when something goes wrong. Speed is capped at `max_velocity`.
    `segment_pause` stops for that many seconds after each waypoint (use ~0.5 on a real drone).
    With a `monitor`, the drone is checked after take-off and after every waypoint, and a
    FlightAbort is raised (after landing) when a check fails.
    """
    if not waypoints:
        raise ValueError("no waypoints")
    velocity = min(velocity, max_velocity)
    t0 = time.monotonic()
    try:
        backend.takeoff(waypoints[0][2])
        if monitor is not None:
            monitor.check(backend.status(), time.monotonic() - t0)
        for i, wp in enumerate(waypoints):
            backend.go_to(wp[0], wp[1], wp[2], velocity)
            if segment_pause > 0:
                time.sleep(segment_pause)
            if monitor is not None:
                monitor.check(backend.status(), time.monotonic() - t0)
            if on_waypoint is not None:
                on_waypoint(i, wp)
    finally:
        backend.land()
