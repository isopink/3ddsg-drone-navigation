"""Real Crazyflie backend (cflib), with the pre-flight checks learned in the hover test.

NOT yet flight-tested: written from the hover-test script and the cflib API, but the
`scripts/hover_test.py` flow is the one validated on hardware so far.
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
from ..backend import RANGE_DIRECTIONS, DroneBackend, Status

DEFAULT_URI = "radio://0/80/2M"
RANGE_INVALID_MM = 8000  # Multi-ranger reach is ~4 m; it reports 32767 when nothing is in view


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
        self._z_cmd = 0.0  # highest drone-frame z of the current move: the height to land from
        self._traj_log: LogConfig | None = None
        self._multiranger: bool | None = None  # None: not asked yet
        self._t0 = time.monotonic()  # time base of `trajectory` and `clock()`
        self.trajectory: list[tuple[float, float, float, float]] = []  # (t, x, y, z), graph frame

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
                self._multiranger = None

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

    def _reset_estimator(self) -> None:
        """Zero the Kalman estimate, as MotionCommander does at take-off (hover test).

        PositionHlCommander does not: its targets are absolute estimator coordinates, so an
        estimate that drifted on the ground would shift and rotate the whole route.
        """
        assert self._scf is not None
        self._scf.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self._scf.cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(2.0)

    def _check_armed(self) -> None:
        info = self._read_log([("supervisor.info", "uint16_t")])[0]["supervisor.info"]
        flags = safety.decode_supervisor(int(info))
        if not (flags["isArmed"] or flags["canFly"]):
            raise safety.SafetyError(f"drone did not arm {flags}: not taking off")

    def _start_trajectory_log(self) -> None:
        """Record the position estimate during the flight (async, 20 Hz)."""
        assert self._scf is not None
        self.trajectory = []
        lg = LogConfig(name="traj", period_in_ms=50)
        for v in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z"):
            lg.add_variable(v, "float")

        def on_data(_ts, data, _cfg):
            t = time.monotonic() - self._t0
            x, y, z = self.frame.to_graph(
                (data["stateEstimate.x"], data["stateEstimate.y"], data["stateEstimate.z"])
            )
            self.trajectory.append((t, x, y, z))

        lg.data_received_cb.add_callback(on_data)
        self._scf.cf.log.add_config(lg)
        lg.start()
        self._traj_log = lg

    def _stop_trajectory_log(self) -> None:
        # the Crazyflie keeps log blocks until they are deleted; leftovers make the next
        # connection print 'Error no LogEntry to handle id=N' (seen in the hover-test flights)
        lg, self._traj_log = self._traj_log, None
        if lg is not None:
            try:
                lg.stop()
                lg.delete()
            except Exception as e:  # never let cleanup break the landing
                print(f"log cleanup warning: {e}")

    def _multiranger_attached(self) -> bool:
        """Multi-ranger deck present (deck.bcMultiranger)? Asked once per connection."""
        if self._multiranger is None:
            assert self._scf is not None
            try:
                self._multiranger = int(self._scf.cf.param.get_value("deck.bcMultiranger")) == 1
            except Exception:  # parameter unknown or unreadable: fly as without the deck
                self._multiranger = False
        return self._multiranger

    def _read_ranges(self) -> dict[str, float]:
        """Multi-ranger distances in meters; directions with nothing in view are left out.

        A log block of its own: with the 18 bytes of the status block the 5 uint16 values
        would pass the 26-byte limit. SyncLogger deletes the block when the read is done.
        """
        s = self._read_log([(f"range.{d}", "uint16_t") for d in RANGE_DIRECTIONS])[0]
        mm = {d: int(s[f"range.{d}"]) for d in RANGE_DIRECTIONS}
        return {d: v / 1000.0 for d, v in mm.items() if v < RANGE_INVALID_MM}

    def status(self) -> Status:
        self.connect()
        s = self._read_log(
            [
                ("pm.vbat", "float"),
                ("supervisor.info", "uint16_t"),
                ("stateEstimate.x", "float"),
                ("stateEstimate.y", "float"),
                ("stateEstimate.z", "float"),
            ]
        )[0]
        flags = safety.decode_supervisor(int(s["supervisor.info"]))
        pos = self.frame.to_graph((s["stateEstimate.x"], s["stateEstimate.y"], s["stateEstimate.z"]))
        ranges = self._read_ranges() if self._multiranger_attached() else {}
        return Status(battery_v=float(s["pm.vbat"]), flags=flags, position=pos, ranges=ranges)

    # -- DroneBackend -------------------------------------------------------
    def clock(self) -> float:
        return time.monotonic() - self._t0

    def takeoff(self, height: float) -> None:
        self.connect()
        assert self._scf is not None
        self.preflight()
        self._reset_estimator()
        self._scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        self._check_armed()
        self._pc = PositionHlCommander(self._scf, x=0.0, y=0.0, z=0.0, default_velocity=0.2)
        self._start_trajectory_log()
        try:
            self._flying = True
            self._z_cmd = self.frame.to_drone((0.0, 0.0, height))[2]
            self._pc.take_off(height=self._z_cmd, velocity=0.2)
        except Exception:
            self.land()
            raise

    def go_to(self, x: float, y: float, z: float, velocity: float) -> None:
        if self._pc is None or not self._flying:
            raise RuntimeError("not flying: call takeoff() first")
        dx, dy, dz = self.frame.to_drone((x, y, z))
        self._z_cmd = max(self._pc.get_position()[2], dz)
        self._pc.go_to(dx, dy, dz, velocity=velocity)
        self._z_cmd = dz

    def position(self) -> Point:
        self.connect()
        s = self._read_log(
            [("stateEstimate.x", "float"), ("stateEstimate.y", "float"), ("stateEstimate.z", "float")]
        )[0]
        return self.frame.to_graph(
            (s["stateEstimate.x"], s["stateEstimate.y"], s["stateEstimate.z"])
        )

    def land(self) -> None:
        try:
            if self._pc is not None and self._flying:
                self._flying = False
                # Not PositionHlCommander.land(): it only learns its height when a move ends,
                # so after an interrupted take-off it lands "from 0 m" in 0 s and cuts the
                # motors in the air.
                assert self._scf is not None
                duration = max(self._pc.get_position()[2], self._z_cmd) / 0.2
                hl = self._scf.cf.high_level_commander
                hl.land(0.0, duration)
                time.sleep(duration)
                hl.stop()
        finally:
            self._stop_trajectory_log()
