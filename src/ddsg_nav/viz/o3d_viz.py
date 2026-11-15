"""Open3D visualization of the scene graph, a planned route and a flown trajectory.

Colors: walls/obstacles gray, rooms blue outlines, doors green, planned route red
(start green, goal magenta), flown trajectory blue.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..graph.model import Box, Point, SceneGraph, VoxelState

GRAY = (0.55, 0.55, 0.6)
ROOM_BLUE = (0.2, 0.35, 0.8)
DOOR_GREEN = (0.1, 0.65, 0.25)
ROUTE_RED = (0.85, 0.1, 0.1)
START_GREEN = (0.1, 0.75, 0.2)
GOAL_MAGENTA = (0.8, 0.1, 0.7)
TRAJ_BLUE = (0.1, 0.35, 0.95)
UNKNOWN_YELLOW = (0.95, 0.8, 0.2)


def _o3d():
    try:
        import open3d as o3d
    except ImportError as e:  # pragma: no cover
        raise ImportError("visualization needs: pip install -e '.[viz]'") from e
    return o3d


def _cylinder(a: Point, b: Point, radius: float, color: tuple[float, float, float]):
    o3d = _o3d()
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    vec = b_ - a_
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return None
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12)
    d = vec / length
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    if s > 1e-9:
        angle = math.atan2(s, float(z @ d))
        mesh.rotate(mesh.get_rotation_matrix_from_axis_angle(axis / s * angle), center=(0, 0, 0))
    elif d[2] < 0:
        mesh.rotate(mesh.get_rotation_matrix_from_axis_angle([math.pi, 0, 0]), center=(0, 0, 0))
    mesh.translate((a_ + b_) / 2)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _sphere(p: Point, radius: float, color: tuple[float, float, float]):
    o3d = _o3d()
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=16)
    mesh.translate(p)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _polyline(points: Sequence[Point], radius: float, color: tuple[float, float, float]) -> list:
    return [m for a, b in zip(points, points[1:]) if (m := _cylinder(a, b, radius, color))]


def _box_outline(box: Box, color: tuple[float, float, float]):
    o3d = _o3d()
    aabb = o3d.geometry.AxisAlignedBoundingBox(box.min, box.max)
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    ls.paint_uniform_color(color)
    return ls


def _voxel_cloud(
    graph: SceneGraph, state: VoxelState, color: tuple[float, float, float], cutaway: bool = False
):
    o3d = _o3d()
    grid = graph.voxels
    idx = np.argwhere(grid.states == int(state))
    if len(idx) == 0:
        return None
    centers = np.asarray(grid.origin) + (idx + 0.5) * grid.resolution
    if cutaway and graph.rooms:  # drop everything south of the rooms: a cutaway view
        y_min = min(r.bounds.min[1] for r in graph.rooms.values())
        centers = centers[centers[:, 1] >= y_min]
        if len(centers) == 0:
            return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.paint_uniform_color(color)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=grid.resolution)


def build_geometries(
    graph: SceneGraph,
    waypoints: Sequence[Point] | None = None,
    trajectory: np.ndarray | None = None,
    show_unknown: bool = True,
    cutaway: bool = False,
) -> list:
    """Open3D geometries for the scene, an optional route and an optional flown trajectory.

    `cutaway=True` leaves out the south wall so the rooms are visible from the south.
    """
    geoms: list = []

    occupied = _voxel_cloud(graph, VoxelState.OCCUPIED, GRAY, cutaway)
    if occupied is not None:
        geoms.append(occupied)
    if show_unknown:
        unknown = _voxel_cloud(graph, VoxelState.UNKNOWN, UNKNOWN_YELLOW, cutaway)
        if unknown is not None:
            geoms.append(unknown)

    for room in graph.rooms.values():  # floor outline of each room
        if room.polygon:  # e.g. an L-shaped hallway: draw its real footprint, not its bounding box
            o3d = _o3d()
            pts = [(x, y, 0.01) for x, y in room.polygon]
            n = len(pts)
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector([(i, (i + 1) % n) for i in range(n)]),
            )
            ls.paint_uniform_color(ROOM_BLUE)
            geoms.append(ls)
            continue
        lo, hi = room.bounds.min, room.bounds.max
        geoms.append(_box_outline(Box((lo[0], lo[1], 0.0), (hi[0], hi[1], 0.01)), ROOM_BLUE))

    for door in graph.doors.values():  # door opening
        x, y, z = door.position
        half = door.width / 2
        geoms.append(
            _box_outline(Box((x - 0.12, y - half, 0.0), (x + 0.12, y + half, 2 * z)), DOOR_GREEN)
        )

    if waypoints:
        geoms += _polyline(waypoints, 0.03, ROUTE_RED)
        for p in waypoints[1:-1]:
            geoms.append(_sphere(p, 0.05, ROUTE_RED))
        geoms.append(_sphere(waypoints[0], 0.09, START_GREEN))
        geoms.append(_sphere(waypoints[-1], 0.09, GOAL_MAGENTA))

    if trajectory is not None and len(trajectory) > 1:
        # drawn 4 cm above the true height so it stays visible on top of the planned route
        pts = [(float(r[0]), float(r[1]), float(r[2]) + 0.04) for r in trajectory]
        geoms += _polyline(pts, 0.012, TRAJ_BLUE)

    return geoms


def scene_center(graph: SceneGraph) -> np.ndarray:
    lo = np.asarray(graph.voxels.origin)
    hi = lo + np.asarray(graph.voxels.shape) * graph.voxels.resolution
    return (lo + hi) / 2


VIEWS = {
    # (front vector from the look-at point toward the camera, up vector, zoom, field of view in degrees)
    # A narrow field of view is close to an orthographic projection: wall tops do not
    # fan outwards and look as thin as they really are.
    # zoom: the visible width is roughly 14 m * zoom at this field of view
    "top": ((0.0, -0.001, 1.0), (0.0, 1.0, 0.0), 0.34, 10.0),
    "oblique": ((0.0, -0.75, 1.0), (0.0, 0.0, 1.0), 0.36, 10.0),
}


def show(geoms: list, title: str = "ddsg_nav") -> None:
    """Open an interactive window (blocks until closed)."""
    _o3d().visualization.draw_geometries(geoms, window_name=title, width=1280, height=720)


def save_screenshot(
    geoms: list,
    path: str,
    center: Sequence[float],
    view: str = "top",
    size: tuple[int, int] = (1280, 720),
    zoom: float | None = None,
) -> bool:
    """Render to a PNG with a hidden window. Returns False if the window could not be created.
    `zoom` overrides the view's default (larger shows more; the default fits the 3-room row)."""
    o3d = _o3d()
    front, up, default_zoom, fov = VIEWS[view]
    zoom = default_zoom if zoom is None else zoom
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=size[0], height=size[1], visible=False):
        return False
    try:
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.background_color = np.array([1.0, 1.0, 1.0])
        opt.mesh_show_back_face = True
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=(fov - ctr.get_field_of_view()) / 5.0)  # one step = 5 deg
        ctr.set_lookat(list(center))
        ctr.set_front(list(front))
        ctr.set_up(list(up))
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(path, do_render=True)
    finally:
        vis.destroy_window()
    return True
