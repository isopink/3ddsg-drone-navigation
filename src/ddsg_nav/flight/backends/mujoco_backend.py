"""MuJoCo simulation backend: same interface as the real drone.

Headless. `trajectory` holds (t, x, y, z) samples; `collisions` counts simulation
steps in which the drone touched a wall/obstacle (the floor does not count).
"""
from __future__ import annotations


import numpy as np

from ...graph.model import Point, SceneGraph
from ...sim.quadrotor import GeometricController, QuadrotorParams
from ...sim.scene import DRONE_BODY, DRONE_GEOM, FLOOR_GEOM, build_mjcf
from ..backend import DroneBackend, Status

ARRIVE_POS_TOL = 0.06  # m
ARRIVE_VEL_TOL = 0.12  # m/s
LAND_Z = 0.04  # m, sphere radius is 0.05 so this presses on the floor


class SimTimeoutError(RuntimeError):
    """The simulated drone did not reach the target in time."""


class MuJoCoBackend(DroneBackend):
    def __init__(
        self,
        graph: SceneGraph,
        start: Point,
        params: QuadrotorParams | None = None,
        record_dt: float = 0.02,
    ) -> None:
        try:
            import mujoco
        except ImportError as e:  # pragma: no cover
            raise ImportError("MuJoCo backend needs: pip install -e '.[sim]'") from e
        self._mj = mujoco
        self._graph = graph
        self._landing = False
        self.params = params or QuadrotorParams()
        # start slightly above the floor so the sphere does not begin in contact
        floor_start = (start[0], start[1], max(start[2], self.params.radius + 0.01))
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(graph, floor_start, self.params))
        self.data = mujoco.MjData(self.model)
        self._body = self.model.body(DRONE_BODY).id
        self._geom = self.model.geom(DRONE_GEOM).id
        self._floor = self.model.geom(FLOOR_GEOM).id
        self._ctrl = GeometricController(self.params)
        self._dt = float(self.model.opt.timestep)
        self._record_every = max(1, round(record_dt / self._dt))
        self._steps = 0
        self._ref = np.array(floor_start, dtype=float)
        self._vref = np.zeros(3)
        self._flying = False
        self.trajectory: list[tuple[float, float, float, float]] = []
        self.collisions = 0
        mujoco.mj_forward(self.model, self.data)
        self._record()

    # -- state --------------------------------------------------------------
    @property
    def time(self) -> float:
        return float(self.data.time)

    def clock(self) -> float:
        return self.time

    @property
    def collided(self) -> bool:
        return self.collisions > 0

    def position(self) -> Point:
        x, y, z = self.data.xpos[self._body]
        return (float(x), float(y), float(z))

    def _record(self) -> None:
        x, y, z = self.data.xpos[self._body]
        self.trajectory.append((self.time, float(x), float(y), float(z)))

    def _check_collision(self) -> None:
        d = self.data
        for c in d.contact[: d.ncon]:
            pair = (c.geom1, c.geom2)
            if self._geom in pair:
                other = pair[1] if pair[0] == self._geom else pair[0]
                if other == self._floor:
                    continue
                # while landing, resting on a surface below (an upper floor) is not a collision
                below = c.pos[2] < d.xpos[self._body][2] - 0.5 * self.params.radius
                if self._landing and below:
                    continue
                self.collisions += 1
                return

    def _surface_below(self, x: float, y: float, z: float) -> float:
        """Top of the highest occupied voxel under (x, y) and below z: the floor to land on."""
        grid = self._graph.voxels
        i, j, k = grid.index((x, y, z))
        if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
            return 0.0
        column = grid.states[i, j, : max(0, min(k, grid.shape[2]))]
        hits = np.flatnonzero(column == 2)  # VoxelState.OCCUPIED
        if len(hits) == 0:
            return 0.0
        return float(grid.origin[2] + (hits[-1] + 1) * grid.resolution)

    # -- simulation ---------------------------------------------------------
    def _step(self) -> None:
        d = self.data
        rot = d.xmat[self._body].reshape(3, 3)
        if self._flying:
            thrust, torque = self._ctrl.compute(
                d.xpos[self._body].copy(),
                d.qvel[0:3].copy(),  # world-frame linear velocity (free joint)
                rot,
                d.qvel[3:6].copy(),  # body-frame angular velocity (free joint)
                self._ref,
                self._vref,
            )
            d.xfrc_applied[self._body, :3] = rot @ np.array([0.0, 0.0, thrust])
            d.xfrc_applied[self._body, 3:] = rot @ torque
        else:
            d.xfrc_applied[self._body, :] = 0.0
        self._mj.mj_step(self.model, d)
        self._steps += 1
        self._check_collision()
        if self._steps % self._record_every == 0:
            self._record()

    def _fly_to(self, target: np.ndarray, velocity: float) -> None:
        start = self.data.xpos[self._body].copy()
        self._ref = start.copy()
        total = float(np.linalg.norm(target - start))
        timeout = 2.0 * total / max(velocity, 1e-3) + 10.0
        t_end = self.time + timeout
        while True:
            to_go = target - self._ref
            dist = float(np.linalg.norm(to_go))
            step = velocity * self._dt
            if dist <= step:
                self._ref = target.copy()
                self._vref = np.zeros(3)
            else:
                self._vref = to_go / dist * velocity
                self._ref = self._ref + to_go / dist * step
            self._step()
            if dist <= step:
                pos_err = float(np.linalg.norm(target - self.data.xpos[self._body]))
                speed = float(np.linalg.norm(self.data.qvel[0:3]))
                if pos_err < ARRIVE_POS_TOL and speed < ARRIVE_VEL_TOL:
                    return
            if self.time > t_end:
                raise SimTimeoutError(
                    f"did not reach {tuple(np.round(target, 2))} in {timeout:.0f} s "
                    f"(at {tuple(np.round(self.position(), 2))})"
                )

    # -- DroneBackend -------------------------------------------------------
    def takeoff(self, height: float) -> None:
        x, y, _ = self.position()
        self._flying = True
        self._fly_to(np.array([x, y, height]), velocity=0.3)

    def go_to(self, x: float, y: float, z: float, velocity: float) -> None:
        if not self._flying:
            raise RuntimeError("not flying: call takeoff() first")
        self._fly_to(np.array([x, y, z], dtype=float), velocity)

    def status(self) -> Status:
        return Status(battery_v=4.1, position=self.position())  # no battery model

    def land(self) -> None:
        if not self._flying:
            return
        x, y, z = self.position()
        self._landing = True
        try:
            self._fly_to(np.array([x, y, self._surface_below(x, y, z) + LAND_Z]), velocity=0.2)
        finally:
            self._flying = False
            for _ in range(int(0.3 / self._dt)):  # let it settle with motors off
                self._step()

    def close(self) -> None:
        self.land()
