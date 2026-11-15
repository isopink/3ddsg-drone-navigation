"""Execute a planned path on any DroneBackend."""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from ..graph.model import Point
from .backend import DroneBackend
from .monitor import FlightAbort, FlightMonitor
from .record import FlightRecorder

TESTED_VELOCITY = 0.3  # m/s, fastest speed checked so far; faster is allowed but untested


def follow_path(
    backend: DroneBackend,
    waypoints: Sequence[Point],
    velocity: float = 0.2,
    max_velocity: float | None = None,
    segment_pause: float = 0.0,
    on_waypoint: Callable[[int, Point], None] | None = None,
    monitor: FlightMonitor | None = None,
    recorder: FlightRecorder | None = None,
) -> None:
    """Take off at the first waypoint's height, fly through all waypoints, land.

    Always lands, also when something goes wrong. Speed is capped at `max_velocity` if given (no cap by default).
    `segment_pause` stops for that many seconds after each waypoint (use ~0.5 on a real drone).
    With a `monitor`, the drone is checked after take-off and after every waypoint, and a
    FlightAbort is raised (after landing) when a check fails.
    With a `recorder`, the drone status at those points and the flight events (take-off,
    waypoints, abort, landing) are recorded for the flight report and live view.
    """
    if not waypoints:
        raise ValueError("no waypoints")
    if velocity <= 0:
        raise ValueError("velocity must be above 0")
    if max_velocity is not None:
        velocity = min(velocity, max_velocity)
    t0 = time.monotonic()

    def event(kind: str, message: str = "", **data) -> None:
        if recorder is not None:
            recorder.event(kind, message, **data)

    def check() -> None:
        if monitor is None and recorder is None:
            return
        status = backend.status()
        if recorder is not None:
            recorder.status(status)
        if monitor is not None:
            monitor.check(status, time.monotonic() - t0)

    try:
        event("takeoff", f"to {waypoints[0][2]:.2f} m")
        backend.takeoff(waypoints[0][2])
        check()
        for i, wp in enumerate(waypoints):
            backend.go_to(wp[0], wp[1], wp[2], velocity)
            if segment_pause > 0:
                time.sleep(segment_pause)
            event("waypoint", f"{i + 1}/{len(waypoints)} reached", index=i)
            check()
            if on_waypoint is not None:
                on_waypoint(i, wp)
    except FlightAbort as e:
        event("abort", str(e))
        raise
    except BaseException as e:  # incl. KeyboardInterrupt: record it, land, re-raise
        event("error", f"{type(e).__name__}: {e}".rstrip(": "))
        raise
    finally:
        event("land", "landing")
        backend.land()
        event("landed", "on the ground")
