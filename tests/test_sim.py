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
