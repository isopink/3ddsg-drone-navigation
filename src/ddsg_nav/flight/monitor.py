"""In-flight monitor: checked after every waypoint (every segment ends in a short stop).

It does not replace the pre-flight checks in safety.py; it watches the flight itself.
Any violation raises FlightAbort, and `follow_path` lands the drone.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..graph.model import Box, Point
from .backend import Status
from .compare import distance_to_polyline


class FlightAbort(RuntimeError):
    """The flight was stopped by the monitor."""


@dataclass
class FlightMonitor:
    min_battery_v: float = 3.5  # in flight; the take-off limit in safety.py is higher
    max_deviation_m: float = 0.5  # from the planned path
    bounds: Box | None = None  # geofence in graph coordinates
    max_duration_s: float | None = None
    path: Sequence[Point] | None = None  # planned waypoints, for the deviation check
    bad_flags: tuple[str, ...] = ("isLocked", "isCrashed", "isTumbled")
    min_range_m: float | None = 0.2  # Multi-ranger: closest allowed obstacle; None turns it off

    def check(self, status: Status, elapsed_s: float = 0.0) -> None:
        if status.battery_v is not None and status.battery_v < self.min_battery_v:
            raise FlightAbort(f"battery {status.battery_v:.2f} V is below {self.min_battery_v} V")
        bad = [f for f in self.bad_flags if status.flags.get(f)]
        if bad:
            raise FlightAbort(f"drone state {bad}")
        if self.max_duration_s is not None and elapsed_s > self.max_duration_s:
            raise FlightAbort(f"flight took {elapsed_s:.0f} s, limit {self.max_duration_s:.0f} s")
        if self.min_range_m is not None:
            close = {d: r for d, r in status.ranges.items() if r < self.min_range_m}
            if close:
                d = min(close, key=close.get)
                raise FlightAbort(
                    f"obstacle {close[d]:.2f} m to the {d} (range limit {self.min_range_m} m)"
                )
        p = status.position
        if p is None:
            return
        if any(not math.isfinite(c) for c in p):
            raise FlightAbort(f"position estimate is not finite: {p}")
        if self.bounds is not None and not self.bounds.contains(p):
            raise FlightAbort(f"position {tuple(round(c, 2) for c in p)} left the allowed area")
        if self.path:
            dev = distance_to_polyline(p, self.path)
            if dev > self.max_deviation_m:
                raise FlightAbort(
                    f"{dev:.2f} m away from the planned path (limit {self.max_deviation_m} m)"
                )
