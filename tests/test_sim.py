import numpy as np
import pytest

pytest.importorskip("mujoco")

from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend  # noqa: E402
from ddsg_nav.flight.executor import follow_path  # noqa: E402
from ddsg_nav.graph.fake import build_fake_graph  # noqa: E402
from ddsg_nav.planner.route import plan_route  # noqa: E402
from ddsg_nav.sim.scene import build_mjcf  # noqa: E402


def test_mjcf_contains_all_solids():
    g = build_fake_graph(obstacle=True)
    xml = build_mjcf(g, (1.5, 1.5, 0.1))
    assert xml.count('name="solid_') == len(g.solids)
    assert 'name="drone"' in xml and 'name="floor"' in xml


def test_takeoff_hover_land():
    g = build_fake_graph()
    with MuJoCoBackend(g, (1.5, 1.5, 0.0)) as drone:
        drone.takeoff(0.5)
        x, y, z = drone.position()
        assert abs(z - 0.5) < 0.08 and abs(x - 1.5) < 0.08 and abs(y - 1.5) < 0.08
        drone.land()
        assert drone.position()[2] < 0.1
        drone.land()  # idempotent
    assert not drone.collided


def test_go_to_requires_takeoff():
    g = build_fake_graph()
    drone = MuJoCoBackend(g, (1.5, 1.5, 0.0))
    with pytest.raises(RuntimeError):
        drone.go_to(2.0, 1.5, 0.5, 0.2)


@pytest.mark.parametrize("obstacle", [False, True])
def test_fly_a_to_c_without_collision(obstacle):
    g = build_fake_graph(obstacle=obstacle)
    plan = plan_route(g, "A", "C", height=0.5)
    start = g.rooms["A"].center
    with MuJoCoBackend(g, (start[0], start[1], 0.0)) as drone:
        follow_path(drone, plan.waypoints, velocity=0.3)
        end = np.array(drone.position())
    assert drone.collisions == 0
    assert np.linalg.norm(end[:2] - np.array(plan.waypoints[-1][:2])) < 0.15
    assert end[2] < 0.1  # landed
    assert g.room_at((end[0], end[1], 0.5)).id == "C"


def test_drone_hits_wall_when_flown_into_it():
    """Sanity check of the collision detector: fly straight into the partition."""
    g = build_fake_graph()
    with MuJoCoBackend(g, (1.5, 0.5, 0.0)) as drone:
        drone.takeoff(0.5)
        try:
            drone.go_to(4.0, 0.5, 0.5, 0.3)  # partition wall at x=3 (y=0.5 is wall, not door)
        except Exception:
            pass  # may time out because the wall blocks it
    assert drone.collided


# ---- Multi-ranger: range sensor and obstacle stop --------------------------

from ddsg_nav.flight.backend import RANGE_DIRECTIONS  # noqa: E402
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor  # noqa: E402
from ddsg_nav.graph.model import Box  # noqa: E402

UNMAPPED = Box((2.0, 1.2, 0.0), (2.2, 1.8, 1.0))  # in room A, on the line y=1.5; not in the graph


def test_mjcf_extra_solids_are_separate_from_graph_solids():
    g = build_fake_graph()
    xml = build_mjcf(g, (1.5, 1.5, 0.1), extra_solids=[UNMAPPED])
    assert xml.count('name="solid_') == len(g.solids) and xml.count('name="extra_') == 1


def test_ranges_measure_the_room():
    g = build_fake_graph()  # room A: x 0..3, y 0..3, no ceiling solid, partition from x=2.9
    with MuJoCoBackend(g, (1.0, 0.6, 0.0)) as drone:
        drone.takeoff(0.5)
        x, y, _ = drone.position()
        r = drone.status().ranges
    assert set(r) == set(RANGE_DIRECTIONS) - {"up"}  # nothing above: left out
    assert r["back"] == pytest.approx(x, abs=0.03)
    assert r["front"] == pytest.approx(2.9 - x, abs=0.03)
    assert r["left"] == pytest.approx(3.0 - y, abs=0.03)
    assert r["right"] == pytest.approx(y, abs=0.03)


def test_ranges_see_an_unmapped_box_and_drop_far_directions():
    g = build_fake_graph()
    # on the door line the ray passes both doors and ends 8 m away: beyond the 4 m reach
    with MuJoCoBackend(g, (1.0, 1.5, 0.0)) as drone:
        drone.takeoff(0.5)
        assert "front" not in drone.status().ranges
    with MuJoCoBackend(g, (1.0, 1.5, 0.0), extra_solids=[UNMAPPED]) as drone:
        drone.takeoff(0.5)
        assert drone.status().ranges["front"] == pytest.approx(2.0 - drone.position()[0], abs=0.03)


def test_drone_stops_before_an_unmapped_box():
    g = build_fake_graph()
    path = [(0.5 + 0.05 * i, 1.5, 0.5) for i in range(41)]  # straight through the box, to x=2.5
    mon = FlightMonitor(path=path)
    with MuJoCoBackend(g, (0.5, 1.5, 0.0), extra_solids=[UNMAPPED]) as drone:
        with pytest.raises(FlightAbort, match="to the front"):
            follow_path(drone, path, velocity=0.3, monitor=mon)
        x, _, z = drone.position()
    assert drone.collisions == 0
    assert x < UNMAPPED.min[0] - drone.params.radius and z < 0.1  # stopped short and landed


MID_BOX = Box((1.4, 1.2, 0.0), (1.6, 1.8, 1.0))  # unmapped, in the middle of the segment below


@pytest.mark.parametrize("spacing", [2.0, 0.5, 0.07])
def test_stop_distance_does_not_depend_on_waypoint_spacing(spacing):
    """A 2 m line through an unmapped box, as one segment or as many: same stop, 0 contacts."""
    g = build_fake_graph()
    n = round(2.0 / spacing)
    path = [(0.5 + 2.0 * i / n, 1.5, 0.5) for i in range(n + 1)]
    mon = FlightMonitor(path=path)
    with MuJoCoBackend(g, (0.5, 1.5, 0.0), extra_solids=[MID_BOX]) as drone:
        with pytest.raises(FlightAbort, match="to the front"):
            follow_path(drone, path, velocity=0.3, monitor=mon)
        x, _, z = drone.position()
    assert drone.collisions == 0 and z < 0.1  # landed without touching the box
    gap = MID_BOX.min[0] - x
    assert 0.2 - 0.1 - 0.03 < gap < 0.2 + 0.03  # range limit 0.2 m, check step 0.1 m


def test_checks_at_waypoints_only_miss_the_box_on_a_long_segment():
    g = build_fake_graph()
    path = [(0.5, 1.5, 0.5), (2.5, 1.5, 0.5)]
    with MuJoCoBackend(g, (0.5, 1.5, 0.0), extra_solids=[MID_BOX]) as drone:
        try:
            follow_path(drone, path, velocity=0.3, monitor=FlightMonitor(path=path), check_step_m=None)
        except Exception:
            pass
    assert drone.collided


def test_without_the_range_check_the_drone_hits_the_box():
    g = build_fake_graph()
    path = [(0.5, 1.5, 0.5), (2.5, 1.5, 0.5)]
    with MuJoCoBackend(g, (0.5, 1.5, 0.0), extra_solids=[UNMAPPED]) as drone:
        try:
            follow_path(drone, path, velocity=0.3, monitor=FlightMonitor(min_range_m=None))
        except Exception:
            pass
    assert drone.collided
