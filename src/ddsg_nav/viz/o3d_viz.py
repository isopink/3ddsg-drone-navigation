"""Open3D visualization of the scene graph, a planned route and a flown trajectory.

Colors: walls/obstacles gray, rooms blue outlines, doors green, planned route red
(start green, goal magenta), flown trajectory blue.

`save_floor_plan` draws the same things top-down into a PNG with numpy only. It needs
no display and no Open3D, so it works on machines where Open3D cannot open a window.

Large graphs: above `max_voxels` occupied voxels, `build_geometries` draws the graph's
solids as one mesh, or (without solids) a down-sampled voxel set, so a whole floor stays
responsive.
"""
from __future__ import annotations

import math
import struct
import zlib
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

MAX_VOXELS = 200_000  # more occupied voxels than this: draw solids or down-sample
LABEL_INSET = 0.3  # m, room label distance from the room's north-west corner
LABEL_HEIGHT = 0.3  # m, letter height of a room label in the Open3D scene


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


def downsample_factor(count: int, max_voxels: int | None) -> int:
    """Voxels per axis to merge into one so that about `max_voxels` are left to draw."""
    if not max_voxels or count <= max_voxels:
        return 1
    return math.ceil((count / max_voxels) ** (1 / 3))


def _voxel_cloud(
    graph: SceneGraph,
    state: VoxelState,
    color: tuple[float, float, float],
    cutaway: bool = False,
    factor: int = 1,
):
    """Voxels in `state`. With `factor` > 1, blocks of factor^3 voxels are drawn as one
    coarse voxel if any voxel in the block is in `state`."""
    o3d = _o3d()
    grid = graph.voxels
    idx = np.argwhere(grid.states == int(state))
    if len(idx) == 0:
        return None
    size = grid.resolution * factor
    if factor > 1:
        idx = np.unique(idx // factor, axis=0)
    centers = np.asarray(grid.origin) + (idx + 0.5) * size
    if cutaway and graph.rooms:  # drop everything south of the rooms: a cutaway view
        y_min = min(r.bounds.min[1] for r in graph.rooms.values())
        centers = centers[centers[:, 1] >= y_min]
        if len(centers) == 0:
            return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.paint_uniform_color(color)
    return o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)


def _solids_mesh(graph: SceneGraph, color: tuple[float, float, float], cutaway: bool = False):
    """All solids (walls, obstacles) as one triangle mesh: cheap to draw at any size."""
    o3d = _o3d()
    solids = graph.solids
    if cutaway and graph.rooms:
        y_min = min(r.bounds.min[1] for r in graph.rooms.values())
        solids = [b for b in solids if b.max[1] > y_min]
    mesh = o3d.geometry.TriangleMesh()
    for b in solids:
        w, d, h = (max(s, 1e-3) for s in b.size)
        box = o3d.geometry.TriangleMesh.create_box(w, d, h)
        box.translate(b.min)
        mesh += box
    if len(mesh.triangles) == 0:
        return None
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _room_label(room):
    """The room id as a flat text mesh on the floor, or None if Open3D cannot draw it
    (an Open3D build without text meshes, or an id its font does not cover)."""
    o3d = _o3d()
    try:
        mesh = o3d.t.geometry.TriangleMesh.create_text(room.id, depth=1.0).to_legacy()
    except Exception:
        return None
    if len(mesh.triangles) == 0:
        return None
    lo, hi = mesh.get_min_bound(), mesh.get_max_bound()
    scale = LABEL_HEIGHT / max(hi[1] - lo[1], 1e-9)
    mesh.translate(-np.array([lo[0], hi[1], lo[2]]))  # top-left corner to the origin
    mesh.scale(scale, center=(0, 0, 0))
    x, y = room.bounds.min[0] + LABEL_INSET, room.bounds.max[1] - LABEL_INSET
    mesh.translate((x, y, room.bounds.min[2] + 0.01))
    mesh.paint_uniform_color(ROOM_BLUE)
    mesh.compute_vertex_normals()
    return mesh


def build_geometries(
    graph: SceneGraph,
    waypoints: Sequence[Point] | None = None,
    trajectory: np.ndarray | None = None,
    show_unknown: bool = True,
    cutaway: bool = False,
    max_voxels: int | None = MAX_VOXELS,
    labels: bool = False,
) -> list:
    """Open3D geometries for the scene, an optional route and an optional flown trajectory.

    `cutaway=True` leaves out the south wall so the rooms are visible from the south.
    `labels=True` writes each room's id on its floor, in the north-west corner.
    With more than `max_voxels` occupied voxels the walls are drawn from the graph's
    solids, or from down-sampled voxels if it has none (`max_voxels=None`: always draw
    every voxel). Unknown voxels are down-sampled the same way.
    """
    geoms: list = []

    states = graph.voxels.states
    n_occupied = int(np.count_nonzero(states == int(VoxelState.OCCUPIED)))
    if max_voxels and n_occupied > max_voxels and graph.solids:
        occupied = _solids_mesh(graph, GRAY, cutaway)
    else:
        factor = downsample_factor(n_occupied, max_voxels)
        occupied = _voxel_cloud(graph, VoxelState.OCCUPIED, GRAY, cutaway, factor)
    if occupied is not None:
        geoms.append(occupied)
    if show_unknown:
        n_unknown = int(np.count_nonzero(states == int(VoxelState.UNKNOWN)))
        factor = downsample_factor(n_unknown, max_voxels)
        unknown = _voxel_cloud(graph, VoxelState.UNKNOWN, UNKNOWN_YELLOW, cutaway, factor)
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

    if labels:
        geoms += [m for room in graph.rooms.values() if (m := _room_label(room)) is not None]

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


# --- top-down floor plan: numpy only, no display, no Open3D -------------------------

# 5 x 7 pixel font: seven rows per character, two hex digits each, bit 4 = left pixel
_FONT = {
    "0": "0E11131519110E", "1": "040C040404040E", "2": "0E11010204081F", "3": "1F02040201110E",
    "4": "02060A121F0202", "5": "1F101E0101110E", "6": "0608101E11110E", "7": "1F010204080808",
    "8": "0E11110E11110E", "9": "0E11110F01020C", "A": "0E1111111F1111", "B": "1E11111E11111E",
    "C": "0E11101010110E", "D": "1C12111111121C", "E": "1F10101E10101F", "F": "1F10101E101010",
    "G": "0E11101711110F", "H": "1111111F111111", "I": "0E04040404040E", "J": "0702020202120C",
    "K": "11121418141211", "L": "1010101010101F", "M": "111B1515111111", "N": "11111915131111",
    "O": "0E11111111110E", "P": "1E11111E101010", "Q": "0E11111115120D", "R": "1E11111E141211",
    "S": "0F10100E01011E", "T": "1F040404040404", "U": "1111111111110E", "V": "11111111110A04",
    "W": "1111111515150A", "X": "11110A040A1111", "Y": "1111110A040404", "Z": "1F01020408101F",
    "-": "0000001F000000", "_": "0000000000001F", ".": "00000000000C0C", "?": "0E110102040004",
    " ": "00000000000000",
}


def _text_mask(text: str) -> np.ndarray:
    """Boolean (7, 6 * len - 1) bitmap of `text`. Lower case is drawn as upper case and
    characters outside the font (for example Hangul) as "?"."""
    mask = np.zeros((7, max(6 * len(text) - 1, 1)), dtype=bool)
    for i, ch in enumerate(text.upper()):
        rows = _FONT.get(ch, _FONT["?"])
        for r in range(7):
            bits = int(rows[2 * r : 2 * r + 2], 16)
            mask[r, 6 * i : 6 * i + 5] = [bool(bits >> (4 - c) & 1) for c in range(5)]
    return mask


class _Canvas:
    """An RGB image with drawing in graph coordinates (x right, y up)."""

    def __init__(self, img: np.ndarray, origin: tuple[float, float], px_per_m: float) -> None:
        self.img = img  # (h, w, 3) uint8, row 0 is the highest y
        self.origin = origin
        self.scale = px_per_m

    def px(self, x: float, y: float) -> tuple[float, float]:
        """Pixel (column, row) of a point."""
        return (
            (x - self.origin[0]) * self.scale,
            self.img.shape[0] - (y - self.origin[1]) * self.scale,
        )

    def _paint(self, c0: int, r0: int, mask: np.ndarray, color) -> None:
        """Color the pixels of `mask`, whose top-left corner is at (c0, r0); clipped."""
        h, w = self.img.shape[:2]
        ra, rb = max(r0, 0), min(r0 + mask.shape[0], h)
        ca, cb = max(c0, 0), min(c0 + mask.shape[1], w)
        if ra >= rb or ca >= cb:
            return
        part = mask[ra - r0 : rb - r0, ca - c0 : cb - c0]
        self.img[ra:rb, ca:cb][part] = np.round(np.asarray(color) * 255).astype(np.uint8)

    def disc(self, p: Sequence[float], radius_px: float, color) -> None:
        c, r = self.px(p[0], p[1])
        n = int(math.ceil(radius_px))
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
        self._paint(int(round(c)) - n, int(round(r)) - n, xx * xx + yy * yy <= radius_px**2, color)

    def square(self, p: Sequence[float], half_px: int, color) -> None:
        c, r = self.px(p[0], p[1])
        mask = np.ones((2 * half_px + 1, 2 * half_px + 1), dtype=bool)
        self._paint(int(round(c)) - half_px, int(round(r)) - half_px, mask, color)

    def line(self, a: Sequence[float], b: Sequence[float], width_px: float, color) -> None:
        (ca, ra), (cb, rb) = self.px(a[0], a[1]), self.px(b[0], b[1])
        n = max(1, int(math.ceil(math.hypot(cb - ca, rb - ra))))
        for t in np.linspace(0.0, 1.0, n + 1):
            p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            self.disc(p, width_px / 2, color)

    def polyline(self, pts: Sequence[Sequence[float]], width_px: float, color) -> None:
        for a, b in zip(pts, pts[1:]):
            self.line(a, b, width_px, color)

    def text(self, p: Sequence[float], text: str, color, size: int = 2) -> None:
        """`text` with its top-left corner at `p`; each font pixel is `size` image pixels."""
        mask = np.kron(_text_mask(text), np.ones((size, size), dtype=bool))
        c, r = self.px(p[0], p[1])
        self._paint(int(round(c)), int(round(r)), mask, color)


def floor_plan_image(graph: SceneGraph, z_min: float, z_max: float) -> np.ndarray:
    """Top-down RGB image (ny, nx, 3), uint8, of the voxels between `z_min` and `z_max`:
    a cell is gray if any voxel in the band is occupied, yellow if none is occupied but
    one is unknown, white otherwise. Row 0 is the highest y (north is up)."""
    grid = graph.voxels
    top = grid.shape[2] - 1
    k0 = min(max(int(math.floor((z_min - grid.origin[2]) / grid.resolution + 1e-9)), 0), top)
    k1 = min(max(int(math.floor((z_max - grid.origin[2]) / grid.resolution + 1e-9)), k0), top)
    band = grid.states[:, :, k0 : k1 + 1]
    occupied = (band == int(VoxelState.OCCUPIED)).any(axis=2)
    unknown = (band == int(VoxelState.UNKNOWN)).any(axis=2) & ~occupied
    img = np.full(occupied.shape + (3,), 255, dtype=np.uint8)
    img[unknown] = np.round(np.asarray(UNKNOWN_YELLOW) * 255)
    img[occupied] = np.round(np.asarray(GRAY) * 255)
    return img.transpose(1, 0, 2)[::-1]


def render_floor_plan(
    graph: SceneGraph,
    waypoints: Sequence[Point] | None = None,
    trajectory: np.ndarray | None = None,
    height: float = 0.5,
    labels: bool = True,
    max_pixels: int = 1600,
) -> np.ndarray:
    """Top-down floor plan as an (h, w, 3) uint8 image: walls, room outlines and labels,
    doors, the route with start and goal, and a flown trajectory.

    The walls are the voxels at the height of the route (its lowest to its highest
    waypoint), or `height` above the lowest room floor without a route: what the drone
    has to get past. Rooms are labelled with their id. One voxel is drawn as 1 to 16
    pixels so that the longer side stays near `max_pixels`; a whole floor is one image
    of that size, however many voxels it has.
    """
    grid = graph.voxels
    if waypoints:
        z_min, z_max = min(p[2] for p in waypoints), max(p[2] for p in waypoints)
    else:
        floors = [r.bounds.min[2] for r in graph.rooms.values()] or [grid.origin[2]]
        z_min = z_max = min(floors) + height
    k = int(min(max(max_pixels // max(grid.shape[0], grid.shape[1]), 1), 16))
    img = floor_plan_image(graph, z_min, z_max).repeat(k, axis=0).repeat(k, axis=1)
    cv = _Canvas(np.ascontiguousarray(img), (grid.origin[0], grid.origin[1]), k / grid.resolution)
    thin = max(1.5, k / 4)
    thick = max(4.0, k / 2)

    for room in graph.rooms.values():
        if room.polygon:
            xy = list(room.polygon)
        else:
            (x0, y0, _), (x1, y1, _) = room.bounds.min, room.bounds.max
            xy = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        cv.polyline(xy + xy[:1], thin, ROOM_BLUE)
    for door in graph.doors.values():
        cv.square(door.position, max(4, int(0.12 * cv.scale)), DOOR_GREEN)
    if trajectory is not None and len(trajectory) > 1:
        cv.polyline([(float(r[0]), float(r[1])) for r in trajectory], thin, TRAJ_BLUE)
    if waypoints:
        cv.polyline(waypoints, thick, ROUTE_RED)
        for p in waypoints[1:-1]:
            cv.disc(p, thick, ROUTE_RED)
        mark = max(9.0, 0.12 * cv.scale)
        cv.disc(waypoints[0], mark, START_GREEN)
        cv.disc(waypoints[-1], mark, GOAL_MAGENTA)
    if labels:  # last, so nothing covers them; in the north-west corner, clear of the route
        for room in graph.rooms.values():
            width_px = (room.bounds.max[0] - room.bounds.min[0]) * cv.scale
            size = int(min(max(width_px // (12 * len(room.id) + 8), 1), 3))
            corner = (room.bounds.min[0] + LABEL_INSET, room.bounds.max[1] - LABEL_INSET)
            cv.text(corner, room.id, ROOM_BLUE, size)
    return cv.img


def write_png(path: str, img: np.ndarray) -> None:
    """Write an (h, w, 3) uint8 image as a PNG (no image library needed)."""
    h, w, _ = img.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    rows = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * 3)], axis=1)  # filter 0
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)))
        f.write(chunk(b"IEND", b""))


def save_floor_plan(
    graph: SceneGraph,
    path: str,
    waypoints: Sequence[Point] | None = None,
    trajectory: np.ndarray | None = None,
    height: float = 0.5,
    labels: bool = True,
    max_pixels: int = 1600,
) -> str:
    """`render_floor_plan` written to `path` as a PNG. Returns `path`."""
    write_png(path, render_floor_plan(graph, waypoints, trajectory, height, labels, max_pixels))
    return path


