import pytest

from ddsg_nav.flight import safety
from ddsg_nav.flight.backend import DroneBackend
from ddsg_nav.flight.executor import MAX_VELOCITY, follow_path


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


def test_follow_path_caps_velocity():
    b = FakeBackend()
    follow_path(b, WAYPOINTS, velocity=5.0)
    assert all(c[4] == MAX_VELOCITY for c in b.calls if c[0] == "go_to")


def test_follow_path_lands_on_error():
    b = FakeBackend(fail_at=1)
    with pytest.raises(RuntimeError):
        follow_path(b, WAYPOINTS)
    assert b.calls[-1] == ("land",)


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
        safety.check_ground_height([0, 0, 0, 0])  # dead: Flow deck sensor did not start
    with pytest.raises(safety.SafetyError):
        safety.check_ground_height([65535] * 10)  # sensor error value
    with pytest.raises(safety.SafetyError):
        safety.check_ground_height([])
    # healthy sensor very close to the floor: steady or near-zero readings must not block take-off
    safety.check_ground_height([4] * 10)  # seen on hardware, falsely called "frozen" before
    safety.check_ground_height([1, 3, 1, 3, 3, 4, 4, 5, 4, 2])  # same drone, flew fine
    safety.check_ground_height([0, 1, 0, 2, 0])  # occasional 0 next to real readings
