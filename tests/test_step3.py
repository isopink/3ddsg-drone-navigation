"""Step 3 preparation: virtual rooms, flight monitor, plan-vs-flown comparison."""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ddsg_nav.flight.backend import DroneBackend, Status
from ddsg_nav.flight.compare import (
    compare_to_plan,
    distance_to_polyline,
    distance_to_segment,
    load_trajectory,
    save_trajectory,
)
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.graph.fake import build_row_graph
from ddsg_nav.graph.model import Box
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

ROOT = Path(__file__).resolve().parents[1]


# ---- virtual rooms ---------------------------------------------------------

def test_row_graph_layout():
    g = build_row_graph([2.0, 3.0, 1.5], depth=2.5, door_width=0.9)
    assert list(g.rooms) == ["A", "B", "C"] and len(g.doors) == 2
    assert g.rooms["B"].bounds.min[0] == 2.0 and g.rooms["B"].bounds.max[0] == 5.0
    assert g.rooms["C"].bounds.max[0] == 6.5
    assert validate_graph(g).ok


def test_row_graph_single_room_and_bad_input():
    assert len(build_row_graph([3.0]).doors) == 0
    with pytest.raises(ValueError):
        build_row_graph([])


@pytest.mark.parametrize(
    "sizes,depth,door", [((2, 2), 2, 1.0), ((1.5, 1.5, 1.5), 1.5, 0.8), ((3, 2, 2, 3), 2.5, 0.9)]
)
def test_row_graph_routes_are_collision_free(sizes, depth, door):
    g = build_row_graph(sizes, depth, door)
    last = chr(ord("A") + len(sizes) - 1)
    plan = plan_route(g, "A", last, height=0.5)
    assert plan.rooms[0] == "A" and plan.rooms[-1] == last
    assert check_path(g, plan.waypoints).ok


def test_row_graph_door_too_narrow_is_reported():
    g = build_row_graph([2, 2], 2, door_width=0.4)  # 0.4 m door with a 0.25 m margin on each side
    r = validate_graph(g)
    assert "door-narrower-than-margin" in {i.code for i in r.issues}


# ---- comparison ------------------------------------------------------------

def test_distance_helpers():
    assert distance_to_segment((1, 1, 0), (0, 0, 0), (2, 0, 0)) == 1.0
    assert distance_to_segment((-1, 0, 0), (0, 0, 0), (2, 0, 0)) == 1.0  # clamps to the end
    assert distance_to_segment((1, 1, 0), (0, 0, 0), (0, 0, 0)) == math.sqrt(2)  # degenerate segment
    assert distance_to_polyline((3, 0.5, 0), [(0, 0, 0), (2, 0, 0), (2, 3, 0)]) == 1.0


def _line(n=50, offset=0.0, z=0.5, length=4.0):
    t = np.linspace(0, 5, n)
    x = np.linspace(0, length, n)
    return [(float(ti), float(xi), float(offset), z) for ti, xi in zip(t, x)]


def test_compare_to_plan_numbers():
    plan = [(0, 0, 0.5), (4, 0, 0.5)]
    c = compare_to_plan(plan, _line(offset=0.1))
    assert c.max_deviation_m == pytest.approx(0.1) and c.mean_deviation_m == pytest.approx(0.1)
    assert c.end_error_m == pytest.approx(0.1)
    assert c.planned_length_m == pytest.approx(4.0) and c.flown_length_m == pytest.approx(4.0)
    assert c.duration_s == pytest.approx(5.0)
    assert "max deviation" in c.format()


def test_compare_ignores_takeoff_and_landing():
    plan = [(0, 0, 0.5), (4, 0, 0.5)]
    traj = [(0.0, 0, 0, 0.05), (0.5, 0, 0, 0.3)] + _line() + [(9.0, 4, 0, 0.05)]
    assert compare_to_plan(plan, traj).samples == 50  # only samples at flight height count


def test_compare_needs_data():
    with pytest.raises(ValueError):
        compare_to_plan([(0, 0, 0.5)], [])


def test_trajectory_csv_roundtrip(tmp_path):
    traj = _line(10)
    p = save_trajectory(tmp_path / "sub" / "t.csv", traj)
    assert np.allclose(np.array(load_trajectory(p)), np.array(traj))


# ---- monitor ---------------------------------------------------------------

PATH = [(0.0, 0.0, 0.5), (4.0, 0.0, 0.5)]
BOX = Box((-0.5, -1.0, 0.0), (4.5, 1.0, 1.5))


def test_monitor_accepts_a_healthy_flight():
    FlightMonitor(path=PATH, bounds=BOX).check(Status(4.0, {}, (2.0, 0.1, 0.5)), 10.0)
    FlightMonitor().check(Status(), 0.0)  # nothing known, nothing to complain about


@pytest.mark.parametrize(
    "status,kwargs,needle",
    [
        (Status(3.2, {}, (2, 0, 0.5)), {}, "battery"),
        (Status(4.0, {"isCrashed": True}, (2, 0, 0.5)), {}, "isCrashed"),
        (Status(4.0, {"isLocked": True}, (2, 0, 0.5)), {}, "isLocked"),
        (Status(4.0, {}, (2, 0.9, 0.5)), {"max_deviation_m": 0.5}, "planned path"),
        (Status(4.0, {}, (6.0, 0, 0.5)), {}, "allowed area"),
        (Status(4.0, {}, (2, 0, float("nan"))), {}, "not finite"),
    ],
)
def test_monitor_aborts(status, kwargs, needle):
    mon = FlightMonitor(path=PATH, bounds=BOX, **kwargs)
    with pytest.raises(FlightAbort, match=needle):
        mon.check(status, 1.0)


def test_monitor_duration_limit():
    mon = FlightMonitor(max_duration_s=30)
    mon.check(Status(), 29.0)
    with pytest.raises(FlightAbort, match="limit"):
        mon.check(Status(), 31.0)


class ScriptedBackend(DroneBackend):
    """Reports a scripted status each time it is asked."""

    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), []

    def takeoff(self, height):
        self.calls.append("takeoff")

    def go_to(self, x, y, z, velocity):
        self.calls.append(("go_to", x, y, z))

    def status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def position(self):
        return (0.0, 0.0, 0.0)

    def land(self):
        self.calls.append("land")

    def close(self):
        pass


def test_executor_lands_when_the_monitor_aborts():
    wps = [(0, 0, 0.5), (1, 0, 0.5), (2, 0, 0.5), (3, 0, 0.5)]
    ok = Status(4.0, {}, (0, 0, 0.5))
    bad = Status(4.0, {"isTumbled": True}, (1, 0, 0.5))
    b = ScriptedBackend([ok, ok, bad])  # after take-off, after waypoint 0, after waypoint 1 -> abort
    with pytest.raises(FlightAbort):
        follow_path(b, wps, monitor=FlightMonitor())
    assert b.calls[-1] == "land"
    assert sum(isinstance(c, tuple) for c in b.calls) == 2  # stopped before waypoints 2 and 3


def test_executor_without_monitor_is_unchanged():
    b = ScriptedBackend([Status()])
    follow_path(b, [(0, 0, 0.5), (1, 0, 0.5)])
    assert b.calls[0] == "takeoff" and b.calls[-1] == "land"


# ---- end to end in the simulator -------------------------------------------

pytest.importorskip("mujoco")

from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend  # noqa: E402


def _sim_flight(monitor_deviation=0.5):
    g = build_row_graph([2, 2, 2], 2.0, 1.0)
    plan = plan_route(g, "A", "C", height=0.5)
    start = plan.waypoints[0]
    mon = FlightMonitor(
        bounds=Box((-0.1, -0.1, 0.0), (6.1, 2.1, 1.5)), path=plan.waypoints, max_deviation_m=monitor_deviation
    )
    drone = MuJoCoBackend(g, (start[0], start[1], 0.0))
    return drone, plan, mon


def test_simulated_flight_stays_close_to_the_plan():
    drone, plan, mon = _sim_flight()
    with drone:
        follow_path(drone, plan.waypoints, velocity=0.3, monitor=mon)
    assert drone.collisions == 0
    c = compare_to_plan(plan.waypoints, drone.trajectory)
    assert c.max_deviation_m < 0.3 and c.end_error_m < 0.15, c.format()


def test_simulated_flight_aborts_and_lands_when_the_limit_is_tiny():
    drone, plan, mon = _sim_flight(monitor_deviation=0.005)  # tighter than the controller can hold
    with drone:
        with pytest.raises(FlightAbort):
            follow_path(drone, plan.waypoints, velocity=0.3, monitor=mon)
        assert drone.position()[2] < 0.1  # landed after the abort
    assert drone.collisions == 0


# ---- command line ----------------------------------------------------------

def test_fly_route_cli_sim():
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fly_route.py"), "--backend", "sim",
         "--rooms", "1.5", "1.5", "--depth", "1.5", "--door-width", "0.8", "go from A to B"],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env, timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "max deviation" in r.stdout and "wall contacts in the simulation: 0" in r.stdout
    bad = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fly_route.py"), "--backend", "sim", "go from A to Z"],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env, timeout=120,
    )
    assert bad.returncode == 1 and "cannot plan" in bad.stdout
