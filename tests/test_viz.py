"""Open3D geometries, the display-free floor plan and scripts/check_routes.py.

Everything runs on the fake graphs; nothing here opens a window.
"""
import importlib.util
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from ddsg_nav.graph.fake import build_corridor_graph, build_fake_graph, build_stairs_graph
from ddsg_nav.graph.io import save_graph
from ddsg_nav.graph.model import VoxelState
from ddsg_nav.planner.route import plan_route
from ddsg_nav.viz import o3d_viz

ROOT = Path(__file__).resolve().parents[1]
needs_o3d = pytest.mark.skipif(importlib.util.find_spec("open3d") is None, reason="needs open3d")


def rgb(color) -> tuple[int, ...]:
    return tuple(int(round(c * 255)) for c in color)


def read_png(path) -> np.ndarray:
    """Decode a PNG written by o3d_viz.write_png (8-bit RGB, filter 0) and check its CRCs."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos : pos + 4])
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])[0] == zlib.crc32(tag + body)
        chunks[tag] = body
        pos += 12 + n
    w, h, depth, color_type = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, color_type) == (8, 2)
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + 3 * w)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(h, w, 3)


def has_color(img: np.ndarray, color) -> bool:
    return bool((img == np.array(rgb(color), np.uint8)).all(axis=2).any())


def pixel(img: np.ndarray, graph, p, k: int) -> tuple[int, ...]:
    """Color at graph point `p` in a floor plan drawn with `k` pixels per voxel."""
    grid = graph.voxels
    col = int((p[0] - grid.origin[0]) / grid.resolution * k)
    row = img.shape[0] - 1 - int((p[1] - grid.origin[1]) / grid.resolution * k)
    return tuple(int(v) for v in img[row, col])


# --- Open3D geometries ---------------------------------------------------------------


@needs_o3d
def test_build_geometries_scene_only():
    g = build_fake_graph()
    geoms = o3d_viz.build_geometries(g)
    # occupied voxels + 3 room outlines + 2 door outlines
    assert len(geoms) == 1 + 3 + 2


@needs_o3d
def test_build_geometries_with_route_and_trajectory():
    g = build_fake_graph(obstacle=True)
    plan = plan_route(g, "A", "C")
    traj = np.array([[1.5, 1.5, 0.5], [3.0, 1.5, 0.5], [4.5, 2.0, 0.5], [7.5, 1.5, 0.5]])
    base = len(o3d_viz.build_geometries(g))
    with_route = o3d_viz.build_geometries(g, plan.waypoints)
    with_traj = o3d_viz.build_geometries(g, plan.waypoints, traj)
    assert len(with_route) > base
    assert len(with_traj) > len(with_route)


@needs_o3d
def test_cutaway_removes_south_wall():
    g = build_fake_graph()
    full = o3d_viz.build_geometries(g)[0]
    cut = o3d_viz.build_geometries(g, cutaway=True)[0]
    assert len(cut.get_voxels()) < len(full.get_voxels())


def test_scene_center():
    c = o3d_viz.scene_center(build_fake_graph())
    assert np.allclose(c, [4.5, 1.5, 1.25])


@needs_o3d
def test_large_graph_draws_solids_instead_of_voxels():
    import open3d as o3d

    g = build_fake_graph(obstacle=True)
    walls = o3d_viz.build_geometries(g, max_voxels=1000)[0]
    assert isinstance(walls, o3d.geometry.TriangleMesh)
    assert len(walls.triangles) == 12 * len(g.solids)
    # same number of geometries, and the default still draws every voxel
    assert len(o3d_viz.build_geometries(g, max_voxels=1000)) == len(o3d_viz.build_geometries(g))
    assert isinstance(o3d_viz.build_geometries(g)[0], o3d.geometry.VoxelGrid)
    cut = o3d_viz.build_geometries(g, max_voxels=1000, cutaway=True)[0]
    assert len(cut.triangles) < len(walls.triangles)


@needs_o3d
def test_large_graph_without_solids_is_downsampled():
    g = build_fake_graph()
    g.solids = []
    n = int(np.count_nonzero(g.voxels.states == int(VoxelState.OCCUPIED)))
    full = o3d_viz.build_geometries(g, max_voxels=None)[0]
    coarse = o3d_viz.build_geometries(g, max_voxels=1000)[0]
    assert len(full.get_voxels()) == n
    assert 0 < len(coarse.get_voxels()) < n / 4
    assert coarse.voxel_size == pytest.approx(g.voxels.resolution * o3d_viz.downsample_factor(n, 1000))
    # the coarse voxels still span the whole building
    assert np.allclose(coarse.get_max_bound(), full.get_max_bound(), atol=coarse.voxel_size)


def test_downsample_factor():
    assert o3d_viz.downsample_factor(1000, 1000) == 1
    assert o3d_viz.downsample_factor(10**6, None) == 1
    assert o3d_viz.downsample_factor(8000, 1000) == 2
    assert o3d_viz.downsample_factor(8001, 1000) == 3


@needs_o3d
def test_room_labels_are_optional_geometry():
    g = build_fake_graph()
    base = o3d_viz.build_geometries(g)
    labelled = o3d_viz.build_geometries(g, labels=True)
    assert len(base) <= len(labelled) <= len(base) + len(g.rooms)
    label = o3d_viz._room_label(g.rooms["A"])
    if label is not None:  # this Open3D build has text meshes
        assert len(labelled) == len(base) + len(g.rooms)
        lo, hi = label.get_min_bound(), label.get_max_bound()
        assert hi[1] - lo[1] == pytest.approx(o3d_viz.LABEL_HEIGHT)
        assert g.rooms["A"].bounds.contains(tuple((lo + hi) / 2))


# --- floor plan: no display, no Open3D -------------------------------------------------


def test_floor_plan_png_of_the_fake_graph(tmp_path):
    g = build_fake_graph(obstacle=True)
    plan = plan_route(g, "A", "C")
    out = tmp_path / "plan.png"
    assert o3d_viz.save_floor_plan(g, str(out), plan.waypoints) == str(out)
    img = read_png(out)
    k = 16  # 92 voxels wide: the largest scale
    assert img.shape == (g.voxels.shape[1] * k, g.voxels.shape[0] * k, 3)
    for color in (o3d_viz.GRAY, o3d_viz.ROOM_BLUE, o3d_viz.DOOR_GREEN, o3d_viz.ROUTE_RED):
        assert has_color(img, color)
    assert pixel(img, g, plan.waypoints[0], k) == rgb(o3d_viz.START_GREEN)
    assert pixel(img, g, plan.waypoints[-1], k) == rgb(o3d_viz.GOAL_MAGENTA)
    assert pixel(img, g, (-0.06, 1.5), k) == rgb(o3d_viz.GRAY)  # west wall
    assert pixel(img, g, (0.8, 0.8), k) == (255, 255, 255)  # free floor in room A
    mid = tuple((a + b) / 2 for a, b in zip(plan.waypoints[0], plan.waypoints[1]))
    assert pixel(img, g, mid, k) == rgb(o3d_viz.ROUTE_RED)


def test_floor_plan_without_route_shows_obstacle_and_trajectory():
    g = build_fake_graph(obstacle=True)
    empty = build_fake_graph(obstacle=False)
    img = o3d_viz.render_floor_plan(g, labels=False)
    assert img.dtype == np.uint8
    assert not has_color(img, o3d_viz.ROUTE_RED)
    gray = lambda im: int((im == np.array(rgb(o3d_viz.GRAY), np.uint8)).all(axis=2).sum())  # noqa: E731
    assert gray(img) > gray(o3d_viz.render_floor_plan(empty, labels=False))
    # labels add blue pixels; a trajectory is drawn in its own color
    blue = lambda im: int((im == np.array(rgb(o3d_viz.ROOM_BLUE), np.uint8)).all(axis=2).sum())  # noqa: E731
    assert blue(o3d_viz.render_floor_plan(g)) > blue(img)
    traj = np.array([[1.0, 1.0, 0.5], [2.0, 1.0, 0.5]])
    assert has_color(o3d_viz.render_floor_plan(g, trajectory=traj), o3d_viz.TRAJ_BLUE)


def test_floor_plan_marks_unknown_voxels():
    g = build_fake_graph()
    i, j, k = g.voxels.index((1.0, 1.0, 0.5))
    g.voxels.states[i - 2 : i + 3, j - 2 : j + 3, k] = int(VoxelState.UNKNOWN)
    img = o3d_viz.render_floor_plan(g, labels=False)
    assert pixel(img, g, (1.0, 1.0), 16) == rgb(o3d_viz.UNKNOWN_YELLOW)
    # a slice above the unknown voxels does not show them
    assert not has_color(o3d_viz.render_floor_plan(g, height=1.5, labels=False), o3d_viz.UNKNOWN_YELLOW)


def test_floor_plan_size_is_bounded():
    g = build_fake_graph()
    small = o3d_viz.render_floor_plan(g, max_pixels=100)
    assert small.shape[:2] == (g.voxels.shape[1], g.voxels.shape[0])  # one pixel per voxel
    assert max(o3d_viz.render_floor_plan(g).shape[:2]) <= 1600


def test_floor_plan_handles_polygon_rooms_and_stairs(tmp_path):
    corridor = build_corridor_graph()
    plan = plan_route(corridor, "A", "C")
    assert has_color(o3d_viz.render_floor_plan(corridor, plan.waypoints), o3d_viz.ROUTE_RED)
    stairs = build_stairs_graph()
    ids = list(stairs.rooms)
    plan = plan_route(stairs, ids[0], ids[-1])
    o3d_viz.save_floor_plan(stairs, str(tmp_path / "stairs.png"), plan.waypoints)
    assert has_color(read_png(tmp_path / "stairs.png"), o3d_viz.GOAL_MAGENTA)


def test_text_mask_falls_back_for_unknown_characters():
    assert o3d_viz._text_mask("r1").tolist() == o3d_viz._text_mask("R1").tolist()
    assert o3d_viz._text_mask("호").tolist() == o3d_viz._text_mask("?").tolist()
    assert o3d_viz._text_mask("AB").shape == (7, 11)


# --- scripts/check_routes.py -----------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("check_routes", ROOT / "scripts" / "check_routes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def folder(tmp_path):
    return str(save_graph(build_fake_graph(obstacle=True), tmp_path / "floor"))


def test_script_default_mode_is_the_fake_collision_table(script, capsys):
    assert script.main([]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].split() == ["obstacle", "height", "pairs", "waypoints", "hard", "margin", "outside"]
    assert len(out.splitlines()) == 1 + 6 + 1
    assert "RESULT: zero collisions in all cases" in out


def test_script_previews_a_graph_folder_as_png(script, folder, tmp_path, capsys):
    out = tmp_path / "out" / "route.png"
    assert script.main(["--graph", folder, "from A to C", "--png", str(out)]) == 0
    text = capsys.readouterr().out
    assert "A -> C: rooms A -> B -> C" in text and f"saved {out}" in text
    assert "obstacle" not in text  # no fake-graph table in this mode
    assert has_color(read_png(out), o3d_viz.ROUTE_RED)


def test_script_default_png_goes_under_data(script, folder, tmp_path, monkeypatch):
    assert script.PREVIEWS == ROOT / "data" / "previews"
    assert script.png_path("", "floor_A_to_C", True) == ROOT / "data" / "previews" / "floor_A_to_C.png"
    monkeypatch.setattr(script, "PREVIEWS", tmp_path / "previews")  # do not write into the checkout
    assert script.main(["--graph", folder, "from A to C", "--png"]) == 0
    assert (tmp_path / "previews" / "floor_A_to_C.png").is_file()


def test_script_refuses_real_graph_pictures_inside_the_repo(script, folder, capsys):
    target = ROOT / "docs" / "media" / "must_not_exist.png"
    assert script.main(["--graph", folder, "from A to C", "--png", str(target)]) == 2
    assert "refused" in capsys.readouterr().out
    assert not target.exists()
    with pytest.raises(ValueError):
        script.png_path(str(ROOT / "preview.png"), "x", True)
    # under data/ is fine, and the fake graph may go anywhere
    assert script.png_path(str(ROOT / "data" / "x.png"), "x", True) == ROOT / "data" / "x.png"
    assert script.png_path(str(target), "x", False) == target


def test_script_all_pairs_table(script, folder, capsys):
    assert script.main(["--graph", folder, "--all-pairs"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["start", "goal", "waypoints", "length_m", "hard", "margin", "outside"]
    assert [line.split()[:2] for line in lines[1:4]] == [["A", "B"], ["A", "C"], ["B", "C"]]
    assert all(line.split()[4:] == ["0", "0", "0"] for line in lines[1:4])
    assert lines[4] == "3 reachable pairs, 0 unreachable, 0 with a collision or no path"


def test_script_all_pairs_skips_unreachable_rooms(script, tmp_path, capsys):
    g = build_fake_graph()
    del g.doors[next(d.id for d in g.doors.values() if "C" in (d.room_a, d.room_b))]
    folder = str(save_graph(g, tmp_path / "split"))
    assert script.main(["--graph", folder, "--all-pairs"]) == 0
    assert "1 reachable pairs, 2 unreachable, 0 with a collision or no path" in capsys.readouterr().out


def test_script_reports_bad_input(script, folder, tmp_path, capsys):
    assert script.main(["--graph", str(tmp_path / "missing"), "from A to C", "--png"]) == 1
    assert "cannot load graph" in capsys.readouterr().out
    assert script.main(["--graph", folder, "from A to Z", "--png", str(tmp_path / "x.png")]) == 1
    assert "cannot plan" in capsys.readouterr().out
    assert not (tmp_path / "x.png").exists()
