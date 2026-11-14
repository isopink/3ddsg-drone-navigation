"""Drone backend interface. The executor only talks to this, so the real
Crazyflie, MuJoCo and (later) Isaac Sim can be swapped without touching planners.

Coordinates are graph coordinates in meters (x, y, z up).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..graph.model import Point


@dataclass(frozen=True)
class Status:
    """What a backend can tell about the drone right now. Unknown values stay None / empty."""

    battery_v: float | None = None
    flags: dict[str, bool] = field(default_factory=dict)  # e.g. isLocked, isCrashed, isTumbled
    position: Point | None = None


class DroneBackend(ABC):
    @abstractmethod
    def takeoff(self, height: float) -> None:
        """Take off and hover at `height` above the floor."""

    @abstractmethod
    def go_to(self, x: float, y: float, z: float, velocity: float) -> None:
        """Fly to a point at `velocity` m/s. Blocks until arrived (or raises)."""

    @abstractmethod
    def position(self) -> Point:
        """Best available position estimate."""

    def status(self) -> Status:
        """Battery, drone state and position for the flight monitor. Optional to override."""
        return Status()

    @abstractmethod
    def land(self) -> None:
        """Land. Must be safe to call when not flying and safe to call twice."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (links, simulators). Lands first if still flying."""

    def __enter__(self) -> "DroneBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.land()
        finally:
            self.close()
