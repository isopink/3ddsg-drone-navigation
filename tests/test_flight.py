import math

import pytest

from ddsg_nav.flight import safety
from ddsg_nav.flight.backend import DroneBackend
from ddsg_nav.flight.executor import follow_path


class FakeBackend(DroneBackend):
    def __init__(self, fail_at: int | None = None):
        self.calls: list[tuple] = []
        self.fail_at = fail_at

    def takeoff(self, height):
        self.calls.append(("takeoff", height))

    def go_to(self, x, y, z, velocity):
        if self.fail_at is not None and sum(c[0] == "go_to" for c in self.calls) == self.fail_at:
            raise RuntimeError("boom")
        self.calls.append(("go_to", x, y, z, velocity))

    def position(self):
        return (0.0, 0.0, 0.0)

    def land(self):
        self.calls.append(("land",))

    def close(self):
        self.calls.append(("close",))


WAYPOINTS = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5)]


def test_follow_path_order():
    b = FakeBackend()
    follow_path(b, WAYPOINTS, velocity=0.2)
    names = [c[0] for c in b.calls]
    assert names == ["takeoff", "go_to", "go_to", "go_to", "land"]
    assert b.calls[0] == ("takeoff", 0.5)


def test_follow_path_has_no_default_speed_cap():
    b = FakeBackend()
    follow_path(b, WAYPOINTS, velocity=5.0)
    assert all(c[4] == 5.0 for c in b.calls if c[0] == "go_to")


def test_follow_path_caps_velocity_when_asked():
    b = FakeBackend()
    follow_path(b, WAYPOINTS, velocity=5.0, max_velocity=0.3)
    assert all(c[4] == 0.3 for c in b.calls if c[0] == "go_to")


def test_follow_path_lands_on_error():
    b = FakeBackend(fail_at=1)
    with pytest.raises(RuntimeError):
        follow_path(b, WAYPOINTS)
    assert b.calls[-1] == ("land",)


def test_split_segment():
    from ddsg_nav.flight.executor import split_segment

    a, b = (0.0, 0.0, 0.5), (1.0, 0.0, 0.5)
    pts = split_segment(a, b, 0.3)
    assert [p[0] for p in pts] == pytest.approx([0.25, 0.5, 0.75, 1.0]) and pts[-1] is b
    assert split_segment(a, b, 1.0) == [b] and split_segment(a, a, 0.1) == [a]
    assert len(split_segment(a, b, 0.1)) == 10  # no extra step from rounding


def test_follow_path_without_monitor_sends_the_waypoints_only():
    b = FakeBackend()
    follow_path(b, WAYPOINTS, velocity=0.2)
    assert [c[1:4] for c in b.calls if c[0] == "go_to"] == WAYPOINTS


def test_follow_path_with_monitor_flies_short_steps_through_every_waypoint():
    from ddsg_nav.flight.monitor import FlightMonitor

    b = FakeBackend()
    reached = []
    follow_path(b, WAYPOINTS, velocity=0.2, monitor=FlightMonitor(), on_waypoint=lambda i, wp: reached.append(i))
    moves = [c[1:4] for c in b.calls if c[0] == "go_to"]
    assert len(moves) == 1 + 10 + 10 and reached == [0, 1, 2]
    assert all(wp in moves for wp in WAYPOINTS) and moves[-1] == WAYPOINTS[-1]
    assert all(math.dist(p, q) <= 0.1 + 1e-9 for p, q in zip(moves, moves[1:]))
    assert b.calls[-1] == ("land",)
    with pytest.raises(ValueError):
        follow_path(FakeBackend(), WAYPOINTS, monitor=FlightMonitor(), check_step_m=0)


def test_follow_path_empty():
    with pytest.raises(ValueError):
        follow_path(FakeBackend(), [])


def test_context_manager_lands_and_closes():
    b = FakeBackend()
    with b:
        pass
    assert [c[0] for c in b.calls] == ["land", "close"]


# ---- safety ----------------------------------------------------------------

def test_decode_supervisor_values_seen_on_hardware():
    # 526: armed + autoArm + canFly + trajFinished  (healthy, from the real drone)
    ok = safety.decode_supervisor(526)
    assert ok["isArmed"] and ok["canFly"] and not ok["isLocked"]
    safety.check_supervisor(ok)
    # 580: autoArm + isLocked + trajFinished  (locked drone: would not take off)
    locked = safety.decode_supervisor(580)
    assert locked["isLocked"] and not locked["canFly"]
    with pytest.raises(safety.SafetyError):
        safety.check_supervisor(locked)
    # crashed state seen after the dark-room crash
    crashed = safety.decode_supervisor(0b10000100 | (1 << 9))
    assert crashed["isCrashed"]
    with pytest.raises(safety.SafetyError):
        safety.check_supervisor(crashed)


def test_check_battery():
    safety.check_battery(4.1)
    with pytest.raises(safety.SafetyError):
        safety.check_battery(3.6)


def test_check_ground_height():
    safety.check_ground_height([18, 17, 15, 19])  # healthy readings from hardware
    with pytest.raises(safety.SafetyError):
        safety.check_ground_height([201] * 10)  # sensor stuck at 201 mm before the crash
    with pytest.raises(safety.SafetyError):
        safety.check_ground_height([0, 0, 0, 0])  # frozen
    with pytest.raises(safety.SafetyError):
        safety.check_ground_height([])


def test_crazyflie_logs_the_position(monkeypatch):
    """No radio: a fake log block checks the variables, the block size and the callback."""
    pytest.importorskip("cflib")
    from ddsg_nav.flight.backends import crazyflie as cfmod
    from ddsg_nav.graph.transform import Transform

    sizes = {"float": 4, "uint16_t": 2}

    class FakeLog:
        def __init__(self, name, period_in_ms):
            self.vars, self.callbacks = [], []
            self.data_received_cb = type("C", (), {"add_callback": lambda _s, cb: self.callbacks.append(cb)})()

        def add_variable(self, name, typ):
            self.vars.append((name, typ))

        def start(self):
            pass

    class FakeScf:
        cf = type("CF", (), {"log": type("L", (), {"add_config": lambda _s, lg: None})()})()

    monkeypatch.setattr(cfmod, "LogConfig", FakeLog)
    drone = cfmod.CrazyflieBackend(frame=Transform((1.0, 2.0, 0.0), 0.0))
    drone._scf = FakeScf()
    drone._start_trajectory_log()
    lg = drone._traj_log
    names = [n for n, _ in lg.vars]
    assert {"stateEstimate.x", "stateEstimate.y", "stateEstimate.z"} <= set(names)
    assert sum(sizes[t] for _, t in lg.vars) <= 26  # Crazyflie log block payload limit
    lg.callbacks[0](0, {"stateEstimate.x": 0.5, "stateEstimate.y": 0.0, "stateEstimate.z": 0.4}, lg)
    assert drone.trajectory[0][1:] == pytest.approx((1.5, 2.0, 0.4))
    drone._scf = None  # nothing to close
