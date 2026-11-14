"""Real Crazyflie backend (cflib), with the pre-flight checks learned in Step 0.

NOT yet flight-tested: written from the Step 0 script and the cflib API, but the
`scripts/step0_hover_test.py` flow is the one validated on hardware so far.
Re-test with `--height 0.3` in a bright room over a textured floor before relying on it.

Coordinates: graph coordinates. `frame` says where the drone's estimator frame sits in
the graph (take-off point and the direction the nose pointed): see graph/transform.py.
"""
from __future__ import annotations

import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.position_hl_commander import PositionHlCommander

from ...graph.model import Point
from ...graph.transform import Transform
from .. import safety
from ..backend import DroneBackend

DEFAULT_URI = "radio://0/80/2M"


class CrazyflieBackend(DroneBackend):
    def __init__(
        self,
        uri: str = DEFAULT_URI,
        frame: Transform = Transform(),
        cache_dir: str = "./cache",
        min_battery: float = safety.MIN_BATTERY_V,
    ) -> None:
        self.uri = uri
        self.frame = frame
        self.min_battery = min_battery
        self._cache_dir = cache_dir
        self._scf: SyncCrazyflie | None = None
        self._pc: PositionHlCommander | None = None
        self._flying = False

    # -- connection ---------------------------------------------------------
    def connect(self) -> None:
        if self._scf is not None:
            return
        cflib.crtp.init_drivers()
        scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache=self._cache_dir))
        scf.open_link()
        self._scf = scf

    def close(self) -> None:
        try:
            self.land()
        finally:
            if self._scf is not None:
                self._scf.close_link()
                self._scf = None

    # -- helpers ------------------------------------------------------------
    def _read_log(self, variables: list[tuple[str, str]], samples: int = 1) -> list[dict]:
        assert self._scf is not None
        lg = LogConfig(name="read", period_in_ms=100)
        for name, typ in variables:
            lg.add_variable(name, typ)
        out: list[dict] = []
        with SyncLogger(self._scf, lg) as logger:
            for _, data, _ in logger:
                out.append(dict(data))
                if len(out) >= samples:
                    break
        return out

    def _flow_deck_attached(self, timeout: float = 5.0) -> bool:
        assert self._scf is not None
        attached = Event()
        self._scf.cf.param.add_update_callback(
            group="deck", name="bcFlow2", cb=lambda _, v: attached.set() if int(v) else None
        )
        return attached.wait(timeout=timeout)

    def preflight(self) -> None:
        """Raise SafetyError unless it is safe to take off."""
        assert self._scf is not None
        if not self._flow_deck_attached():
            raise safety.SafetyError("Flow deck v2 not detected (deck.bcFlow2): check seating")
        vbat = self._read_log([("pm.vbat", "float")])[0]["pm.vbat"]
        safety.check_battery(vbat, self.min_battery)
        info = self._read_log([("supervisor.info", "uint16_t")])[0]["supervisor.info"]
        safety.check_supervisor(safety.decode_supervisor(int(info)))
        heights = [
            int(s["range.zrange"]) for s in self._read_log([("range.zrange", "uint16_t")], 10)
        ]
        safety.check_ground_height(heights)

    # -- DroneBackend -------------------------------------------------------
    def takeoff(self, height: float) -> None:
        self.connect()
        assert self._scf is not None
        self.preflight()
        self._scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        self._pc = PositionHlCommander(self._scf, x=0.0, y=0.0, z=0.0, default_velocity=0.2)
        try:
            self._flying = True
            self._pc.take_off(height=self.frame.to_drone((0.0, 0.0, height))[2], velocity=0.2)
        except Exception:
            self.land()
            raise

    def go_to(self, x: float, y: float, z: float, velocity: float) -> None:
        if self._pc is None or not self._flying:
            raise RuntimeError("not flying: call takeoff() first")
        dx, dy, dz = self.frame.to_drone((x, y, z))
        self._pc.go_to(dx, dy, dz, velocity=velocity)

    def position(self) -> Point:
        self.connect()
        s = self._read_log(
            [("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float")]
        )[0]
        return self.frame.to_graph(
            (s["stateEstimate.x"], s["stateEstimate.y"], s["stateEstimate.z"])
        )

    def land(self) -> None:
        if self._pc is not None and self._flying:
            self._flying = False
            self._pc.land(velocity=0.2)
