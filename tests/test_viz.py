import numpy as np
import pytest

pytest.importorskip("open3d")

from ddsg_nav.graph.fake import build_fake_graph  # noqa: E402
from ddsg_nav.planner.route import plan_route  # noqa: E402
from ddsg_nav.viz import o3d_viz  # noqa: E402


def test_build_geometries_scene_only():
    g = build_fake_graph()
    geoms = o3d_viz.build_geometries(g)
    # occupied voxels + 3 room outlines + 2 door outlines
    assert len(geoms) == 1 + 3 + 2


def test_build_geometries_with_route_and_trajectory():
    g = build_fake_graph(obstacle=True)
    plan = plan_route(g, "A", "C")
    traj = np.array([[1.5, 1.5, 0.5], [3.0, 1.5, 0.5], [4.5, 2.0, 0.5], [7.5, 1.5, 0.5]])
    base = len(o3d_viz.build_geometries(g))
    with_route = o3d_viz.build_geometries(g, plan.waypoints)
    with_traj = o3d_viz.build_geometries(g, plan.waypoints, traj)
    assert len(with_route) > base
    assert len(with_traj) > len(with_route)


def test_cutaway_removes_south_wall():
    g = build_fake_graph()
    full = o3d_viz.build_geometries(g)[0]
    cut = o3d_viz.build_geometries(g, cutaway=True)[0]
    assert len(cut.get_voxels()) < len(full.get_voxels())


def test_scene_center():
    c = o3d_viz.scene_center(build_fake_graph())
    assert np.allclose(c, [4.5, 1.5, 1.25])
