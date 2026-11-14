"""Scene graph -> MuJoCo scene (MJCF). Walls, doors and obstacles come from `graph.solids`."""
from __future__ import annotations

from ..graph.model import Point, SceneGraph
from .quadrotor import QuadrotorParams

DRONE_BODY = "drone"
DRONE_GEOM = "drone_col"
FLOOR_GEOM = "floor"


def build_mjcf(
    graph: SceneGraph,
    start: Point,
    params: QuadrotorParams | None = None,
    timestep: float = 0.002,
) -> str:
    """Return MJCF XML with the graph's solids and one free-flying drone at `start`."""
    p = params or QuadrotorParams()
    solids = []
    for i, box in enumerate(graph.solids):
        cx, cy, cz = box.center
        hx, hy, hz = (s / 2 for s in box.size)
        solids.append(
            f'    <geom name="solid_{i}" type="box" pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="0.62 0.62 0.68 0.55"/>'
        )
    ixx, iyy, izz = p.inertia
    sx, sy, sz = start
    return f"""<mujoco model="ddsg_scene">
  <option timestep="{timestep}" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="4.5 1.5 6" dir="0 0 -1"/>
    <geom name="{FLOOR_GEOM}" type="plane" size="20 20 0.1" rgba="0.92 0.92 0.92 1"/>
{chr(10).join(solids)}
    <body name="{DRONE_BODY}" pos="{sx:.4f} {sy:.4f} {sz:.4f}">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{p.mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{DRONE_GEOM}" type="sphere" size="{p.radius}" rgba="0.15 0.4 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""
