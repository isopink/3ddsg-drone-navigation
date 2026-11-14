"""Frame transform between the drone's frame and the graph's frame."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .model import Point


@dataclass(frozen=True)
class Transform:
    """Pose of the drone frame inside the graph frame (rotation about z, then translation):

        p_graph = R(yaw) @ p_drone + translation

    The drone frame starts at the take-off point, with x along the nose at take-off
    (the Flow deck only knows its heading relative to start). So `translation` is where
    the drone took off in graph coordinates and `yaw_deg` is the direction its nose
    pointed, measured counter-clockwise from the graph's x axis.
    """

    translation: Point = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0

    def to_graph(self, p: Point) -> Point:
        c, s = self._cos_sin()
        tx, ty, tz = self.translation
        return (c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + ty, p[2] + tz)

    def to_drone(self, p: Point) -> Point:
        c, s = self._cos_sin()
        tx, ty, tz = self.translation
        x, y = p[0] - tx, p[1] - ty
        return (c * x + s * y, -s * x + c * y, p[2] - tz)

    def _cos_sin(self) -> tuple[float, float]:
        a = math.radians(self.yaw_deg)
        return math.cos(a), math.sin(a)
