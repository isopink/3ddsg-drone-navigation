"""scripts/fly_route.py on a saved graph folder. Simulation and fakes only: no drone, no radio.

The graphs are generated here (graph/fake.py) and saved to a temporary folder.
"""
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mujoco")

from ddsg_nav.flight.backend import DroneBackend  # noqa: E402
from ddsg_nav.graph.fake import build_fake_graph, build_stairs_graph  # noqa: E402
from ddsg_nav.graph.io import save_graph  # noqa: E402
from ddsg_nav.graph.model import Box, VoxelState  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fly_route.py"
spec = importlib.util.spec_from_file_location("fly_route", SCRIPT)
fly_route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fly_route)


class FakeDrone(DroneBackend):
    """Stands in for CrazyflieBackend: records what it was asked to do."""

    made: list["FakeDrone"] = []

    def __init__(self, uri, frame):
        self.frame, self.calls, self.trajectory = frame, [], []
        FakeDrone.made.append(self)

    def takeoff(self, height):
        self.calls.append(("takeoff", height))

    def go_to(self, x, y, z, velocity):
        self.calls.append(("go_to", x, y, z))

    def position(self):
        return (0.0, 0.0, 0.0)

    def land(self):
        self.calls.append(("land",))

    def close(self):
        self.calls.append(("close",))


@pytest.fixture(autouse=True)
def no_real_drone(monkeypatch):
    """Whatever a test does, it cannot reach a radio or an LLM, or wait for the countdown."""
    from ddsg_nav.flight.backends import crazyflie

    FakeDrone.made = []
    monkeypatch.setattr(crazyflie, "CrazyflieBackend", FakeDrone)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(fly_route, "llm_from_env", lambda: None)  # rules only, no network


@pytest.fixture
def folder(tmp_path):
    """Rooms A - B - C in a row, 3 x 3 m each, saved in the exchange format."""
    return str(save_graph(build_fake_graph(), tmp_path / "graph"))


def run(capsys, *args):
    code = fly_route.main([str(a) for a in args])
    return code, capsys.readouterr().out


def test_graph_folder_flies_in_simulation(capsys, folder):
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "--velocity", 0.3, "from A to C")
    assert code == 0, out
    assert "3 rooms, 2 doors" in out and "route: A -> B -> C" in out
    assert "wall contacts in the simulation: 0" in out
    # where to put the drone: room, two nearest walls, the door, the nose
    assert "start: in room Room A (A) at x=1.50 y=1.50" in out
    assert "1.50 m from the door to B (toward +x)" in out
    assert "Nose 0 deg from the graph +x axis" in out


def test_virtual_rooms_as_before(capsys):
    code, out = run(capsys, "--backend", "sim", "--rooms", 1.5, 1.5, "--depth", 1.5,
                    "--door-width", 0.8, "go from A to B")
    assert code == 0, out
    assert "virtual space: 2 rooms [1.5, 1.5] x 1.5 m, doors 0.8 m wide" in out
    assert "wall contacts in the simulation: 0" in out
    code, out = run(capsys, "--backend", "sim", "go from A to Z")  # default rooms: A, B
    assert code == 1 and "cannot plan" in out


def test_start_off_center(capsys, folder):
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "--velocity", 0.3,
                    "--start", 0.6, 2.2, "from A to B")
    assert code == 0, out
    assert "at x=0.60 y=2.20: 0.60 m from the -x wall, 0.80 m from the +y wall" in out
    assert "(0.6, 2.2, 0.5)" in out  # the first waypoint
    assert "wall contacts in the simulation: 0" in out


@pytest.mark.parametrize(
    "xy, why",
    [((4.5, 1.5), "not inside the start room"), ((0.1, 0.1), "safety margin")],
    ids=["other-room", "too-close-to-a-wall"],
)
def test_bad_start_is_refused(capsys, folder, xy, why):
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "--start", *xy, "from A to B")
    assert code == 1 and "cannot plan" in out and why in out


def test_plan_from_keeps_the_rest_of_the_route():
    from ddsg_nav.planner.route import plan_route

    graph = build_fake_graph()
    plan = plan_route(graph, "A", "C")
    moved = fly_route.plan_from(graph, plan, (0.6, 2.2), 0.5, 0.25, "obstacle")
    assert moved.waypoints[0] == (0.6, 2.2, 0.5) == moved.keypoints[0]
    assert moved.keypoints[1:] == plan.keypoints[1:]
    door = plan.keypoints[1]
    assert moved.waypoints[moved.waypoints.index(door):] == plan.waypoints[plan.waypoints.index(door):]


def test_policy_for_unobserved_voxels(capsys, tmp_path):
    graph = build_fake_graph()
    graph.voxels.fill_box(Box((4.0, 0.0, 0.0), (5.0, 3.0, 2.5)), VoxelState.UNKNOWN)  # a band across B
    folder = str(save_graph(graph, tmp_path / "graph"))
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "from A to C")
    assert code == 1, out  # default: unknown is an obstacle
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "--velocity", 0.3,
                    "--policy", "free_in_rooms", "from A to C")
    assert code == 0, out
    assert "unobserved voxels: free_in_rooms" in out


def test_current_room(capsys, folder):
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "go to B")
    assert code == 1 and "cannot plan" in out
    code, out = run(capsys, "--backend", "sim", "--graph", folder, "--velocity", 0.3,
                    "--current-room", "A", "go to B")
    assert code == 0, out
    assert "goal: A -> B" in out


def test_bad_graph_arguments(capsys, tmp_path, folder):
    code, out = run(capsys, "--graph", tmp_path / "nothing", "from A to B")
    assert code == 1 and "cannot load the graph" in out
    with pytest.raises(SystemExit):
        fly_route.main(["--graph", folder, "--rooms", "2", "2", "from A to B"])


# ---- real drone: refusals, and the hand-over to the backend (a fake) -------

def test_real_drone_refuses_a_floor_change(capsys, tmp_path):
    folder = str(save_graph(build_stairs_graph(), tmp_path / "stairs"))
    code, out = run(capsys, "--backend", "crazyflie", "--graph", folder, "from A to C")
    assert code == 1, out
    assert "not flying the real drone: the route changes floors" in out
    assert FakeDrone.made == []


def test_real_drone_refuses_a_low_route(capsys, folder):
    code, out = run(capsys, "--backend", "crazyflie", "--graph", folder, "--height", 0.1, "from A to B")
    assert code == 1, out
    assert "not flying the real drone" in out and "outside the 0.2 to 3.0 m range" in out
    assert FakeDrone.made == []


def test_real_flight_problem():
    graph = build_fake_graph()

    def plan(*heights):
        return SimpleNamespace(rooms=["A", "B"], waypoints=[(1.5, 1.5, h) for h in heights])

    assert fly_route.real_flight_problem(graph, plan(0.2, 3.0)) is None
    assert "0.19 to 0.50" in fly_route.real_flight_problem(graph, plan(0.5, 0.19))
    assert "0.50 to 3.10" in fly_route.real_flight_problem(graph, plan(0.5, 3.1))
    stairs = build_stairs_graph()
    assert "changes floors" in fly_route.real_flight_problem(
        stairs, SimpleNamespace(rooms=["A", "S", "C"], waypoints=[(1.5, 1.5, 0.5)])
    )


def test_real_drone_is_not_flown_when_the_dry_run_fails(capsys, folder, monkeypatch):
    monkeypatch.setattr(fly_route, "fly", lambda *a, **k: "simulated abort")
    code, out = run(capsys, "--backend", "crazyflie", "--graph", folder, "from A to B")
    assert code == 1 and "the dry run failed: not flying the real drone" in out
    assert FakeDrone.made == []


def test_real_drone_gets_the_start_pose(capsys, folder):
    code, out = run(capsys, "--backend", "crazyflie", "--graph", folder, "--velocity", 0.3,
                    "--start", 0.6, 2.2, "--yaw", 90, "from A to B")
    assert code == 0, out
    assert "(dry run)" in out and "wall contacts in the simulation: 0" in out
    assert "REAL DRONE. Put it in room Room A (A) at x=0.60 y=2.20" in out
    assert "Nose 90 deg from the graph +x axis" in out
    (drone,) = FakeDrone.made
    assert drone.frame.translation == (0.6, 2.2, 0.0) and drone.frame.yaw_deg == 90
    assert drone.calls[0] == ("takeoff", 0.5)
    assert drone.calls[1] == ("go_to", 0.6, 2.2, 0.5)
    assert drone.calls[-2:] == [("land",), ("close",)]
