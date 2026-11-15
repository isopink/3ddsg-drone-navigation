"""CrazyflieBackend against a fake Crazyflie (no radio, no drone).

The fake replaces the link, the parameters, the log blocks and the on-board commander.
The real cflib `PositionHlCommander` and `SyncLogger` run on top of it, so their own
bookkeeping is part of the test. `time.sleep` is replaced: nothing waits.
"""
import time

import pytest

pytest.importorskip("cflib")

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: E402

from ddsg_nav.flight import safety  # noqa: E402
from ddsg_nav.flight.backends import crazyflie as cfmod  # noqa: E402
from ddsg_nav.flight.executor import follow_path  # noqa: E402
from ddsg_nav.graph.transform import Transform  # noqa: E402


class Callbacks:
    def __init__(self):
        self.cbs = []

    def add_callback(self, cb):
        self.cbs.append(cb)

    def remove_callback(self, cb):
        self.cbs.remove(cb)


class FakeCf:
    """The drone: what it reports, and an ordered record of what it was told."""

    def __init__(self):
        self.events = []  # commands, in order
        self.blocks = []  # log blocks alive on the drone
        self.link_open = False
        self.flow_deck = True
        self.arms = True  # False: the arming request is ignored
        self.flags = {"canBeArmed"}
        self.vbat = 4.0
        self.zrange = [22, 23, 24, 23]  # mm, cycled
        self.estimate = (0.0, 0.0, 0.0)  # drone frame
        self.interrupt_during = None  # command name: Ctrl+C in the sleep that follows it
        self.fail_on = None  # command name: raises instead of being sent
        self._n = 0
        self.disconnected = Callbacks()
        self.param = self.log = self.supervisor = self.high_level_commander = self

    def cmd(self, name, *args):
        if self.fail_on == name:
            raise RuntimeError(f"{name} failed")
        self.events.append((name, *args))

    def names(self):
        return [e[0] for e in self.events]

    def sent(self, name):
        return [e[1:] for e in self.events if e[0] == name]

    def is_connected(self):
        return self.link_open

    # param
    def add_update_callback(self, group, name, cb):
        if self.flow_deck:
            cb(f"{group}.{name}", "1")

    def set_value(self, name, value):
        self.cmd("param", name, value)

    # log
    def add_config(self, lg):
        lg.cf = self

    def read(self, var):
        self._n += 1
        x, y, z = self.estimate
        return {
            "pm.vbat": self.vbat,
            "supervisor.info": sum(1 << safety.SUPERVISOR_BITS.index(f) for f in self.flags),
            "range.zrange": self.zrange[self._n % len(self.zrange)],
            "stateEstimate.x": x,
            "stateEstimate.y": y,
            "stateEstimate.z": z,
        }[var]

    # supervisor
    def send_arming_request(self, do_arm):
        self.cmd("arm", do_arm)
        if do_arm and self.arms:
            self.flags.add("isArmed")

    # high-level commander
    def takeoff(self, height, duration):
        self.cmd("takeoff", height, duration)

    def go_to(self, x, y, z, yaw, duration):
        self.cmd("go_to", x, y, z, yaw, duration)

    def land(self, height, duration):
        self.cmd("land", height, duration)

    def stop(self):
        self.cmd("stop")


class FakeLogConfig:
    def __init__(self, name, period_in_ms):
        self.name, self.vars, self.cf = name, [], None
        self.data_received_cb = Callbacks()

    def add_variable(self, name, typ):
        self.vars.append(name)

    def push(self):
        data = {v: self.cf.read(v) for v in self.vars}
        for cb in list(self.data_received_cb.cbs):
            cb(0, data, self)

    def start(self):
        self.cf.blocks.append(self.name)
        for _ in range(12):
            self.push()

    def stop(self):
        pass

    def delete(self):
        self.cf.blocks.remove(self.name)


class FakeLink(SyncCrazyflie):  # a real subclass: cflib helpers check isinstance
    def __init__(self, uri, cf=None):
        self.uri, self.cf = uri, cf

    def open_link(self):
        self.cf.cmd("open", self.uri)
        self.cf.link_open = True

    def close_link(self):
        self.cf.cmd("close")
        self.cf.link_open = False


@pytest.fixture
def cf(monkeypatch):
    cf = FakeCf()

    def sleep(seconds):
        if cf.interrupt_during and cf.events and cf.events[-1][0] == cf.interrupt_during:
            cf.interrupt_during = None
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(cfmod.cflib.crtp, "init_drivers", lambda: cf.cmd("init_drivers"))
    monkeypatch.setattr(cfmod, "Crazyflie", lambda rw_cache=None: cf)
    monkeypatch.setattr(cfmod, "SyncCrazyflie", FakeLink)
    monkeypatch.setattr(cfmod, "LogConfig", FakeLogConfig)
    return cf


def test_connect_once_and_close(cf):
    drone = cfmod.CrazyflieBackend("radio://0/80/2M")
    drone.connect()
    drone.connect()
    assert cf.events == [("init_drivers",), ("open", "radio://0/80/2M")]
    drone.close()
    drone.close()
    assert cf.names() == ["init_drivers", "open", "close"]
    assert not cf.link_open


@pytest.mark.parametrize(
    "fault",
    [
        lambda cf: setattr(cf, "flow_deck", False),
        lambda cf: setattr(cf, "vbat", 3.5),
        lambda cf: cf.flags.add("isLocked"),
        lambda cf: cf.flags.add("isCrashed"),
        lambda cf: cf.flags.add("isTumbled"),
        lambda cf: setattr(cf, "zrange", [201]),  # frozen
        lambda cf: setattr(cf, "zrange", [150, 160]),  # blocked
        lambda cf: setattr(cf, "arms", False),
    ],
    ids=["no-deck", "battery", "locked", "crashed", "tumbled", "frozen", "blocked", "not-armed"],
)
def test_no_take_off_when_a_check_fails(cf, fault, monkeypatch):
    fault(cf)
    drone = cfmod.CrazyflieBackend()
    if not cf.flow_deck:  # do not wait 5 s for the deck
        assert drone._flow_deck_attached.__defaults__ == (5.0,)
        monkeypatch.setattr(drone._flow_deck_attached.__func__, "__defaults__", (0.01,))
    with pytest.raises(safety.SafetyError):
        drone.takeoff(0.5)
    assert "takeoff" not in cf.names()
    if cf.arms:
        assert "arm" not in cf.names()  # failed before arming
    drone.close()
    assert cf.blocks == [] and not cf.link_open


def test_take_off_sequence(cf):
    drone = cfmod.CrazyflieBackend()
    drone.takeoff(0.5)
    assert cf.names() == ["init_drivers", "open", "param", "param", "arm", "takeoff"]
    # estimator reset, like MotionCommander in the hover test
    assert cf.sent("param") == [("kalman.resetEstimation", "1"), ("kalman.resetEstimation", "0")]
    assert cf.sent("arm") == [(True,)]
    assert cf.sent("takeoff") == [pytest.approx((0.5, 2.5))]  # 0.2 m/s
    assert cf.blocks == ["traj"]  # the pre-flight log blocks are gone
    drone.close()


def test_go_to_uses_the_graph_frame(cf):
    """Start offset (1, 2) and nose along graph +y (--yaw 90)."""
    drone = cfmod.CrazyflieBackend(frame=Transform((1.0, 2.0, 0.0), 90.0))
    with pytest.raises(RuntimeError):
        drone.go_to(1.0, 3.0, 0.5, 0.2)  # not flying
    drone.takeoff(0.5)
    drone.go_to(1.0, 2.0, 0.5, 0.2)  # the take-off point at height: nothing to do
    assert cf.sent("go_to") == []
    drone.go_to(1.0, 3.0, 0.5, 0.2)  # 1 m along graph +y = 1 m forward
    drone.go_to(0.0, 3.0, 0.5, 0.25)  # 1 m along graph -x = 1 m to the left
    a, b = cf.sent("go_to")
    assert a == pytest.approx((1.0, 0.0, 0.5, 0, 5.0))
    assert b == pytest.approx((1.0, 1.0, 0.5, 0, 4.0))
    drone.close()


def test_status_and_trajectory_are_in_the_graph_frame(cf):
    drone = cfmod.CrazyflieBackend(frame=Transform((1.0, 2.0, 0.0), 90.0))
    drone.takeoff(0.5)
    cf.estimate = (1.0, 0.0, 0.5)
    drone._traj_log.push()
    assert drone.trajectory[-1][1:] == pytest.approx((1.0, 3.0, 0.5))
    assert drone.position() == pytest.approx((1.0, 3.0, 0.5))
    s = drone.status()
    assert s.position == pytest.approx((1.0, 3.0, 0.5))
    assert s.battery_v == pytest.approx(4.0)
    assert s.flags["isArmed"] and not s.flags["isCrashed"]
    assert cf.blocks == ["traj"]  # reading the status leaves no log block behind
    drone.close()


def test_land_stops_the_motors_and_cleans_up(cf):
    drone = cfmod.CrazyflieBackend()
    drone.land()  # safe when not flying
    assert cf.events == []
    drone.takeoff(0.5)
    drone.land()
    drone.land()  # safe twice
    assert cf.sent("land") == [pytest.approx((0.0, 2.5))]
    assert cf.names()[-2:] == ["land", "stop"]
    assert cf.blocks == []
    with pytest.raises(RuntimeError):
        drone.go_to(0.0, 0.0, 0.5, 0.2)
    drone.close()
    assert cf.names()[-1] == "close" and len(cf.sent("land")) == 1


def test_close_lands_first(cf):
    drone = cfmod.CrazyflieBackend()
    drone.takeoff(0.5)
    drone.close()
    assert cf.names()[-3:] == ["land", "stop", "close"]
    assert cf.blocks == []


PATH = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5)]


def test_follow_path_flies_and_lands(cf):
    drone = cfmod.CrazyflieBackend()
    follow_path(drone, PATH, velocity=0.2)
    drone.close()
    assert cf.names()[4:] == ["arm", "takeoff", "go_to", "go_to", "land", "stop", "close"]
    assert cf.blocks == []


def test_lands_when_a_move_fails(cf):
    drone = cfmod.CrazyflieBackend()
    cf.fail_on = "go_to"
    with pytest.raises(RuntimeError, match="go_to failed"):
        follow_path(drone, PATH, velocity=0.2)
    assert cf.names()[-2:] == ["land", "stop"]
    assert cf.sent("land") == [pytest.approx((0.0, 2.5))]
    assert cf.blocks == []
    drone.close()


def test_lands_on_ctrl_c_during_a_move(cf):
    drone = cfmod.CrazyflieBackend()
    cf.interrupt_during = "go_to"
    with pytest.raises(KeyboardInterrupt):
        follow_path(drone, PATH, velocity=0.2)
    assert cf.names()[-3:] == ["go_to", "land", "stop"]
    assert cf.sent("land") == [pytest.approx((0.0, 2.5))]
    assert cf.blocks == []
    drone.close()


def test_ctrl_c_during_a_climb_lands_from_the_target_height(cf):
    drone = cfmod.CrazyflieBackend()
    drone.takeoff(0.5)
    cf.interrupt_during = "go_to"
    with pytest.raises(KeyboardInterrupt):
        drone.go_to(0.0, 0.0, 1.5, 0.2)  # may be anywhere between 0.5 and 1.5 m
    drone.land()
    assert cf.sent("land") == [pytest.approx((0.0, 7.5))]
    drone.close()


def test_ctrl_c_during_take_off_does_not_cut_the_motors_in_the_air(cf):
    """cflib still believes z = 0 here: its own land() would take 0 s and then stop."""
    drone = cfmod.CrazyflieBackend()
    cf.interrupt_during = "takeoff"
    with pytest.raises(KeyboardInterrupt):
        follow_path(drone, PATH, velocity=0.2)
    assert cf.names()[-3:] == ["takeoff", "land", "stop"]
    assert cf.sent("land") == [pytest.approx((0.0, 2.5))]
    assert cf.blocks == []
    drone.close()


def test_lands_when_take_off_fails(cf):
    drone = cfmod.CrazyflieBackend()
    cf.fail_on = "takeoff"
    with pytest.raises(RuntimeError, match="takeoff failed"):
        drone.takeoff(0.5)
    assert cf.names()[-2:] == ["land", "stop"]
    assert cf.blocks == []
    drone.close()


# ---- Multi-ranger deck -----------------------------------------------------

from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor  # noqa: E402

TYPE_BYTES = {"float": 4, "uint16_t": 2, "uint8_t": 1}
FAR = {"front": 32767, "back": 32767, "left": 32767, "right": 32767, "up": 32767}  # nothing in view


class SizedLogConfig(FakeLogConfig):
    """Refuses a block the real drone would refuse (26 bytes of log data)."""

    def add_variable(self, name, typ):
        super().add_variable(name, typ)
        self.size = getattr(self, "size", 0) + TYPE_BYTES[typ]
        assert self.size <= 26, f"log block {self.name} is {self.size} bytes: {self.vars}"


def _add_multiranger(cf, monkeypatch, mm, present="1"):
    """Give the fake drone the deck parameter and the 5 range variables (values in mm)."""
    cf.ranges_mm = dict(mm)
    cf.get_value = lambda name: {"deck.bcMultiranger": present}[name]
    base_read = cf.read

    def read(var):
        if var.startswith("range.") and var != "range.zrange":
            return cf.ranges_mm[var.split(".")[1]]
        return base_read(var)

    cf.read = read
    monkeypatch.setattr(cfmod, "LogConfig", SizedLogConfig)


def test_ranges_with_the_deck_are_in_meters(cf, monkeypatch):
    _add_multiranger(cf, monkeypatch, {"front": 1234, "back": 250, "left": 32767, "right": 3999, "up": 8000})
    drone = cfmod.CrazyflieBackend()
    drone.takeoff(0.5)
    s = drone.status()
    # left (32767) and up (8000): nothing in view, left out
    assert s.ranges == pytest.approx({"front": 1.234, "back": 0.25, "right": 3.999})
    assert s.battery_v == pytest.approx(4.0) and s.position == pytest.approx((0.0, 0.0, 0.0))
    assert cf.blocks == ["traj"]  # the range block is deleted after the read
    drone.close()
    assert cf.blocks == []


@pytest.mark.parametrize("present", [None, "0"])
def test_no_ranges_without_the_deck(cf, monkeypatch, present):
    if present is not None:  # parameter there but 0; None: the fake has no such parameter
        _add_multiranger(cf, monkeypatch, {}, present)  # reading a range would raise KeyError
    drone = cfmod.CrazyflieBackend()
    drone.takeoff(0.5)
    assert drone.status().ranges == {}
    FlightMonitor().check(drone.status())
    drone.close()


def test_the_deck_is_asked_once_per_connection(cf, monkeypatch):
    _add_multiranger(cf, monkeypatch, FAR)
    asked = []
    get = cf.get_value
    cf.get_value = lambda name: asked.append(name) or get(name)
    drone = cfmod.CrazyflieBackend()
    drone.status(), drone.status()
    assert asked == ["deck.bcMultiranger"]
    drone.close()
    drone.status()
    assert len(asked) == 2
    drone.close()


def test_monitor_lands_the_drone_near_an_obstacle(cf, monkeypatch):
    _add_multiranger(cf, monkeypatch, FAR)
    drone = cfmod.CrazyflieBackend()
    wps = [(0.0, 0.0, 0.5), (0.5, 0.0, 0.5), (1.0, 0.0, 0.5)]

    def on_waypoint(i, wp):
        if i == 0:
            cf.ranges_mm["front"] = 150  # a hand comes close

    with pytest.raises(FlightAbort, match="0.15 m to the front"):
        follow_path(drone, wps, monitor=FlightMonitor(), on_waypoint=on_waypoint)
    # waypoint 0 is the take-off point (no move); the check after the first 0.1 m step aborts
    assert [g[0] for g in cf.sent("go_to")] == pytest.approx([0.1])
    assert cf.names()[-2:] == ["land", "stop"] and cf.blocks == []
    drone.close()


def test_monitor_ignores_far_and_empty_readings(cf, monkeypatch):
    _add_multiranger(cf, monkeypatch, {**FAR, "left": 200, "up": 2100})  # 0.20 m: at the limit
    drone = cfmod.CrazyflieBackend()
    follow_path(drone, [(0.0, 0.0, 0.5), (0.5, 0.0, 0.5)], monitor=FlightMonitor())
    assert cf.sent("go_to")[-1][0] == 0.5 and cf.names()[-2:] == ["land", "stop"]
    drone.close()
