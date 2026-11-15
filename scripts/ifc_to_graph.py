"""Convert one storey of an IFC (BIM) model into the scene graph exchange format.

    cf-env\\Scripts\\python scripts\\ifc_to_graph.py model.ifc data\\ifc_4F --storey 4F [--res 0.1]
        [--exclude "^(PS|EPS|EV)\\d"]   # leave out shafts
        [--report]                      # element types, counts, what was skipped and why

Needs `pip install ifcopenshell` (plus shapely, which it installs).

What it does (see docs/graph_format.md for the output):
  rooms   one per IfcSpace on the storey: id = Name ("401"), name = LongName ("카페"),
          polygon = floor footprint, bounds = space volume. GlobalId is an alias.
          A space without 3D geometry or without floor area is skipped with a warning.
  doors   one per IfcDoor whose space boundaries touch two spaces. A door without any
          space boundary gets its rooms from geometry: the rooms whose footprint lies
          within 0.35 m of the door opening. Doors are treated
          as OPEN: the door leaf is not an obstacle, the wall opening is flyable.
          Spaces whose footprints touch (room separation lines, no wall between them)
          get an "open_*" connection at the middle of the shared edge.
  voxels  every other building element (walls, columns, slabs, stairs, windows,
          ceilings, MEP, fixtures, ...) is OCCUPIED (its surface, sampled at res/2).
          Space interiors and door openings that are not occupied are FREE.
          Everything else stays UNKNOWN.
          Stairs are OCCUPIED like any element; the space above the steps stays FREE.
          A room area with no floor under it (slab opening, stair well, atrium) is left
          UNKNOWN: not flyable.
  solids  boxes that cover the occupied voxels from 0.2 m above the floor (the
          simulator's walls; its ground plane stands in for the floor).
  frame   x, y as in the IFC world frame, in meters (other length units are converted);
          z = 0 at the storey elevation. Nothing is moved or turned: a building far from
          the origin keeps its offset, a rotated building stays rotated (its walls then
          cross the voxel grid at an angle).

Bad input raises ConversionError with what is wrong and what to do about it.
One command for convert, validate, plan and simulate: scripts\\ifc_pipeline.py
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element
    import ifcopenshell.util.placement
    import ifcopenshell.util.unit
    from shapely.geometry import LineString, MultiLineString, Polygon as ShPolygon, box as shapely_box
    from shapely.ops import unary_union
except ImportError:
    sys.exit('needs ifcopenshell: pip install -e ".[ifc]"')

from ddsg_nav.graph.io import save_graph
from ddsg_nav.graph.model import Box, Door, Room, SceneGraph, VoxelGrid, VoxelState, points_in_polygon

SKIP_AS_OBSTACLE = ("IfcSpace", "IfcOpeningElement", "IfcDoor", "IfcVirtualElement", "IfcAnnotation", "IfcGrid")
FLOOR_SKIP_M = 0.2  # solids start this far above the floor (floor finishes are not walls)
MAX_EXTENT_M = 2000.0  # a storey wider than this means wrong units
MIN_ROOM_M = 0.5  # a largest room smaller than this means wrong units
MIN_ROOM_AREA_M2 = 0.1  # a space with less floor area is skipped
DOOR_REACH_M = 0.35  # a door without space boundaries connects rooms this close to its opening


class ConversionError(ValueError):
    """The IFC cannot be converted. The message says what is wrong and what to do."""


# ---- geometry ----------------------------------------------------------------

def meshes(f, elements) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """World-coordinate triangle meshes (meters) keyed by element id."""
    if not elements:
        return {}
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, multiprocessing.cpu_count(), include=elements)
    out = {}
    if it.initialize():
        while True:
            sh = it.get()
            v = np.asarray(sh.geometry.verts, dtype=float).reshape(-1, 3)
            t = np.asarray(sh.geometry.faces, dtype=np.int64).reshape(-1, 3)
            if len(t):
                out[sh.id] = (v, t)
            if not it.next():
                break
    return out


_BARY: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _bary(n: int):
    if n not in _BARY:
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        _BARY[n] = (i[keep] / n, j[keep] / n)
    return _BARY[n]


def mark_surface(states: np.ndarray, origin: np.ndarray, res: float, v: np.ndarray, t: np.ndarray,
                 value: int) -> None:
    """Set every voxel a triangle passes through (sampled at res/2) to `value`.

    Samples sit a hair inside the solid (against the outward face normal, and away from
    the triangle's edges), so a face or edge that lies exactly on a voxel boundary does
    not mark the voxel outside it."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    inward = -1e-4 * normal / np.where(length > 0, length, 1.0)
    edge = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1),
                            np.linalg.norm(c - b, axis=1)]), axis=0)
    n = np.maximum(1, np.ceil(edge / (res / 2))).astype(int)
    shape = np.array(states.shape)
    for nv in np.unique(n):
        sel = np.nonzero(n == nv)[0]
        u, w = (x * (1 - 3e-4) + 1e-4 for x in _bary(int(nv)))  # just inside the triangle
        for chunk in np.array_split(sel, max(1, len(sel) * len(u) // 2_000_000 + 1)):
            A, AB, AC = a[chunk], b[chunk] - a[chunk], c[chunk] - a[chunk]
            p = (A + inward[chunk])[:, None] + AB[:, None] * u[None, :, None] + AC[:, None] * w[None, :, None]
            idx = np.floor((p.reshape(-1, 3) - origin) / res + 1e-9).astype(int)
            ok = np.all((idx >= 0) & (idx < shape), axis=1)
            idx = idx[ok]
            states[idx[:, 0], idx[:, 1], idx[:, 2]] = value


def fill_solid(states: np.ndarray, origin: np.ndarray, res: float, v: np.ndarray, t: np.ndarray,
               value: int) -> None:
    """Fill the inside of a closed mesh: cast a vertical ray through every voxel column
    and fill between pairs of crossings. Columns with an odd number of crossings (open
    mesh) are left alone."""
    nx_, ny_, nz_ = states.shape
    jitter = np.array([1.37e-5, 2.71e-5])  # keep rays off mesh edges that sit on voxel centers
    hits_col, hits_z = [], []
    for a, b, c in v[t]:
        area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(area2) < 1e-12:
            continue  # vertical face: parallel to the ray
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        i0 = max(0, int(np.ceil((lo[0] - origin[0]) / res - 0.5)))
        i1 = min(nx_ - 1, int(np.floor((hi[0] - origin[0]) / res - 0.5)))
        j0 = max(0, int(np.ceil((lo[1] - origin[1]) / res - 0.5)))
        j1 = min(ny_ - 1, int(np.floor((hi[1] - origin[1]) / res - 0.5)))
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px = origin[0] + (ii + 0.5) * res + jitter[0]
        py = origin[1] + (jj + 0.5) * res + jitter[1]
        w1 = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / area2
        w2 = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / area2
        inside = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        if not inside.any():
            continue
        z = a[2] + w1[inside] * (b[2] - a[2]) + w2[inside] * (c[2] - a[2])
        hits_col.append(ii[inside] * ny_ + jj[inside])
        hits_z.append(z)
    if not hits_col:
        return
    col = np.concatenate(hits_col)
    z = np.concatenate(hits_z)
    order = np.lexsort((z, col))
    col, z = col[order], z[order]
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    ends = np.r_[starts[1:], len(col)]
    for s, e in zip(starts, ends):
        if (e - s) % 2:
            continue
        i, j = divmod(int(col[s]), ny_)
        for z_in, z_out in zip(z[s:e:2], z[s + 1:e:2]):
            k0 = max(0, int(np.ceil((z_in - origin[2]) / res - 0.5)))
            k1 = min(nz_ - 1, int(np.floor((z_out - origin[2]) / res - 0.5)))
            if k1 >= k0:
                states[i, j, k0:k1 + 1] = value


def footprint(v: np.ndarray, t: np.ndarray):
    """2D outline of a mesh: union of its triangles projected onto the floor."""
    tris = []
    for tri in v[t][:, :, :2]:
        p = ShPolygon(tri)
        if p.area > 1e-6:
            tris.append(p)
    return unary_union(tris).buffer(0)


def solid_mask(grid: VoxelGrid, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Voxels whose center is inside the mesh's footprint and z range (good for
    extruded shapes such as openings)."""
    poly = footprint(v, t)
    if poly.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    if poly.geom_type == "MultiPolygon":
        polys = list(poly.geoms)
    else:
        polys = [poly]
    lo, hi = v.min(axis=0), v.max(axis=0)
    mask = np.zeros(grid.shape, dtype=bool)
    sx, sy, sz = grid.box_slices(Box(tuple(lo), tuple(hi)))
    if sx.stop <= sx.start or sy.stop <= sy.start or sz.stop <= sz.start:
        return mask
    xs = grid.origin[0] + (np.arange(sx.start, sx.stop) + 0.5) * grid.resolution
    ys = grid.origin[1] + (np.arange(sy.start, sy.stop) + 0.5) * grid.resolution
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inside = np.zeros(gx.shape, dtype=bool)
    for p in polys:
        inside |= points_in_polygon(gx, gy, tuple(p.exterior.coords)[:-1])
    mask[sx, sy, sz] = inside[:, :, None]
    return mask


def voxel_solids(grid: VoxelGrid, occupied: np.ndarray, k_min: int) -> list[Box]:
    """Boxes that cover exactly the occupied voxels from layer `k_min` up. Vertical runs
    with the same z range are merged into rectangles, so a straight wall is one box."""
    occ = occupied[:, :, k_min:]
    edges = np.diff(np.pad(occ, ((0, 0), (0, 0), (1, 1))).astype(np.int8), axis=2)
    i, j, k0 = np.nonzero(edges == 1)
    k1 = np.nonzero(edges == -1)[2]  # same (i, j) order as the starts
    o, res = np.array(grid.origin), grid.resolution
    boxes = []
    for z0, z1 in sorted({(int(p), int(q)) for p, q in zip(k0, k1)}):
        m = np.zeros(occ.shape[:2], dtype=bool)
        sel = (k0 == z0) & (k1 == z1)
        m[i[sel], j[sel]] = True
        for x0, y0 in np.argwhere(m):
            if not m[x0, y0]:
                continue  # already inside an earlier box
            y1 = y0 + 1
            while y1 < m.shape[1] and m[x0, y1]:
                y1 += 1
            x1 = x0 + 1
            while x1 < m.shape[0] and m[x1, y0:y1].all():
                x1 += 1
            m[x0:x1, y0:y1] = False
            lo = o + np.array([x0, y0, k_min + z0]) * res
            hi = o + np.array([x1, y1, k_min + z1]) * res
            boxes.append(Box(tuple(round(float(c), 6) for c in lo), tuple(round(float(c), 6) for c in hi)))
    return boxes


# ---- IFC -> graph --------------------------------------------------------------

def door_width(door, scale: float) -> float:
    """Clear width in meters: from the type name ("D900x2100") if present, else OverallWidth."""
    m = re.search(r"(\d{3,4})x(\d{3,4})", door.Name or "")
    if m:
        return int(m.group(1)) / 1000
    return float(door.OverallWidth or 0.9 / scale) * scale


def open_ifc(path: str):
    if not os.path.isfile(path):
        raise ConversionError(f"{path} not found. Check the path to the .ifc file.")
    try:
        return ifcopenshell.open(path)
    except Exception as e:
        raise ConversionError(f"{path} cannot be read as IFC ({e}). Export it again as an IFC2x3 or IFC4 "
                              f".ifc file.") from e


def storey_spaces(storey) -> list:
    """IfcSpace objects of a storey, whether aggregated under it or contained in it."""
    found = [o for r in storey.IsDecomposedBy or [] for o in r.RelatedObjects]
    found += [o for r in getattr(storey, "ContainsElements", None) or [] for o in r.RelatedElements]
    return [o for o in found if o.is_a("IfcSpace")]


def storey_label(storey) -> str:
    return storey.Name or storey.GlobalId


def storey_elevation(storey, scale: float) -> float:
    """Storey level in meters: the Elevation attribute, or the placement height without it."""
    if storey.Elevation is not None:
        return float(storey.Elevation) * scale
    return float(ifcopenshell.util.placement.get_storey_elevation(storey)) * scale


def list_storeys(path: str) -> list[tuple[str, float, int]]:
    """(name, elevation in meters, number of IfcSpace) for every storey, lowest first."""
    f = open_ifc(path)
    scale = ifcopenshell.util.unit.calculate_unit_scale(f)
    rows = [(storey_label(s), storey_elevation(s, scale), len(storey_spaces(s)))
            for s in f.by_type("IfcBuildingStorey")]
    return sorted(rows, key=lambda r: r[1])


@dataclass
class ConversionReport:
    """What the converter found, used and skipped (printed by `--report`)."""

    storey: str = ""
    elevation_m: float = 0.0
    unit_scale: float = 1.0
    elements: dict[str, int] = field(default_factory=dict)  # obstacle elements used, by IFC type
    skipped_elements: dict[tuple[str, str], int] = field(default_factory=dict)  # (type, reason): count
    spaces: int = 0
    rooms: int = 0
    skipped_spaces: list[tuple[str, str]] = field(default_factory=list)  # (room id, reason)
    doors: int = 0  # IfcDoor on the storey
    doors_by_boundary: int = 0  # connected through space boundaries
    doors_by_geometry: int = 0  # connected through the door position
    doors_without_boundary: int = 0
    skipped_doors: list[tuple[str, str]] = field(default_factory=list)  # (door, reason)
    open_connections: int = 0
    voids: list[tuple[str, float]] = field(default_factory=list)  # (room id, m2 without a floor)

    def format(self) -> str:
        lines = ["--- conversion report ---",
                 f"storey: {self.storey} (elevation {self.elevation_m:.2f} m, length unit {self.unit_scale:g} m)",
                 f"obstacle elements: {sum(self.elements.values())}"]
        lines += [f"  {t}: {n}" for t, n in sorted(self.elements.items())]
        lines.append(f"elements skipped: {sum(self.skipped_elements.values())}")
        lines += [f"  {t}: {n} ({why})" for (t, why), n in sorted(self.skipped_elements.items())]
        lines.append(f"spaces: {self.spaces}, rooms: {self.rooms}, skipped: {len(self.skipped_spaces)}")
        lines += [f"  {rid}: {why}" for rid, why in self.skipped_spaces]
        lines.append(f"doors: {self.doors}, connected by space boundary: {self.doors_by_boundary}, "
                     f"by geometry: {self.doors_by_geometry} (of {self.doors_without_boundary} without a "
                     f"boundary), skipped: {len(self.skipped_doors)}")
        lines += [f"  {door}: {why}" for door, why in self.skipped_doors]
        lines.append(f"open connections (no wall): {self.open_connections}")
        lines.append(f"rooms with a floor void: {len(self.voids)}")
        lines += [f"  {rid}: {area:.1f} m2 not flyable" for rid, area in self.voids]
        return "\n".join(lines)


def rooms_at_door(geo, polys: dict) -> list[str]:
    """Rooms a door leads to, from geometry: the door (or opening) footprint grown by
    DOOR_REACH_M against the room footprints. A room counts if its overlap is at least
    half the largest one (this drops a neighbour that only touches a corner); at most
    the two largest are returned, sorted by id."""
    shapes = [footprint(v, t) for v, t in geo]
    shape = unary_union([s for s in shapes if not s.is_empty])
    if shape.is_empty:  # degenerate mesh: use its bounding box
        v = np.vstack([v for v, _ in geo])
        shape = shapely_box(*v.min(axis=0)[:2], *v.max(axis=0)[:2])
    reach = shape.buffer(DOOR_REACH_M)
    overlap = {rid: reach.intersection(poly).area for rid, poly in polys.items()}
    best = max(overlap.values(), default=0.0)
    if best < 1e-3:
        return []
    near = sorted((rid for rid, a in overlap.items() if a >= best / 2), key=lambda r: -overlap[r])
    return sorted(near[:2])


def space_footprint(v: np.ndarray, t: np.ndarray):
    """(polygon, number of parts) of a space mesh; polygon is None without a usable area."""
    poly = footprint(v, t)
    parts = [p for p in getattr(poly, "geoms", [poly]) if p.geom_type == "Polygon" and not p.is_empty]
    if not parts:
        return None, 0
    poly = max(parts, key=lambda p: p.area)
    simple = poly.simplify(0.005)
    if simple.geom_type == "Polygon" and not simple.is_empty:
        poly = simple
    if poly.area < MIN_ROOM_AREA_M2:
        return None, len(parts)
    return poly, len(parts)


def convert(path: str, storey_name: str, res: float = 0.1, pad: float = 0.4,
            min_open_m: float = 0.7, exclude: str | None = None) -> tuple[SceneGraph, list[str]]:
    """Convert the storey called `storey_name` to a SceneGraph. Returns the graph and notes
    on what was skipped or assumed. Raises ConversionError on input that cannot be used."""
    graph, notes, _ = convert_with_report(path, storey_name, res, pad, min_open_m, exclude)
    return graph, notes


def convert_with_report(path: str, storey_name: str, res: float = 0.1, pad: float = 0.4,
                        min_open_m: float = 0.7, exclude: str | None = None,
                        ) -> tuple[SceneGraph, list[str], ConversionReport]:
    """`convert` plus a ConversionReport: element types, counts, what was skipped and why."""
    f = open_ifc(path)
    notes: list[str] = []
    scale = ifcopenshell.util.unit.calculate_unit_scale(f)  # file length unit -> meters
    if abs(scale - 1.0) > 1e-9:
        notes.append(f"length unit of the file is {scale:g} m; everything is converted to meters")
    storeys = {storey_label(s): s for s in f.by_type("IfcBuildingStorey")}
    available = ", ".join(f"{n} ({len(storey_spaces(s))} spaces)" for n, s in storeys.items()) or "none"
    if storey_name not in storeys:
        raise ConversionError(f"storey {storey_name!r} is not in the file. Use one of these names with "
                              f"--storey: {available}.")
    storey = storeys[storey_name]
    z0 = storey_elevation(storey, scale)
    report = ConversionReport(storey=storey_name, elevation_m=z0, unit_scale=scale)

    spaces = storey_spaces(storey)
    if not spaces:
        raise ConversionError(f"storey {storey_name!r} has no IfcSpace, and rooms come from IfcSpace. Pick a "
                              f"storey that has spaces ({available}), or export the model again with "
                              f"rooms/spaces included.")
    if exclude:
        dropped = [s.Name for s in spaces if re.search(exclude, s.Name or "")]
        spaces = [s for s in spaces if not re.search(exclude, s.Name or "")]
        notes.append(f"excluded spaces ({exclude}): {', '.join(dropped) or 'none'}")
        if not spaces:
            raise ConversionError(f"--exclude {exclude!r} removes every space of storey {storey_name!r}. "
                                  f"Use a narrower pattern.")
    report.spaces = len(spaces)
    # room ids: the space Name, made unique (GlobalId if a space has no name)
    ids: dict[int, str] = {}
    for sp in spaces:
        rid, n = sp.Name or sp.GlobalId, 2
        while rid in ids.values():
            rid, n = f"{sp.Name}_{n}", n + 1
        if rid != sp.Name:
            notes.append(f"space {sp.GlobalId} has {'a duplicate name' if sp.Name else 'no name'}; "
                         f"its room id is {rid}")
        ids[sp.id()] = rid

    space_mesh = meshes(f, [sp for sp in spaces if sp.Representation is not None])
    if not space_mesh:
        raise ConversionError(f"the {len(spaces)} spaces of storey {storey_name!r} have no 3D geometry. "
                              f"Export the model again with space geometry (body representation).")
    allv = np.vstack([v for v, _ in space_mesh.values()])
    extent = float((allv.max(axis=0) - allv.min(axis=0))[:2].max())
    largest_room = max(float((v.max(axis=0) - v.min(axis=0))[:2].max()) for v, _ in space_mesh.values())
    if extent > MAX_EXTENT_M or largest_room < MIN_ROOM_M:
        raise ConversionError(
            f"sizes are not plausible in meters: storey {storey_name!r} is {extent:g} m wide and its largest "
            f"room {largest_room:g} m. The file declares its length unit as {scale:g} m, which is probably "
            f"wrong (for example millimeters declared as meters). Fix the project units in the authoring "
            f"tool and export again.")
    # usable spaces: a mesh with a real floor area; the others are skipped with a warning
    shifted, polys = {}, {}
    for sp in spaces:
        rid = ids[sp.id()]
        if sp.id() not in space_mesh:
            why = "no 3D geometry"
        else:
            v, t = space_mesh[sp.id()]
            poly, parts = space_footprint(v, t)
            why = None if poly is not None else f"floor area below {MIN_ROOM_AREA_M2} m2"
        if why:
            notes.append(f"warning: space {rid} skipped: {why}")
            report.skipped_spaces.append((rid, why))
            continue
        if parts > 1:
            notes.append(f"space {rid}: footprint has {parts} parts; using the largest")
        shifted[sp.id()] = (v - [0, 0, z0], t)
        polys[rid] = poly
    if not shifted:
        raise ConversionError(f"none of the {len(spaces)} spaces of storey {storey_name!r} has a floor area. "
                              f"Export the model again with space geometry (body representation).")

    # grid extent: all usable spaces + padding
    allv = np.vstack([v for v, _ in shifted.values()])
    lo = np.floor((allv.min(axis=0) - pad) / res) * res
    hi = np.ceil((allv.max(axis=0) + pad) / res) * res
    lo[2] = min(lo[2], -2 * res)
    size = hi - lo
    grid = VoxelGrid.empty(tuple(lo), tuple(size), res, VoxelState.UNKNOWN)
    origin = np.array(grid.origin)

    # obstacles: every product with geometry except spaces, openings and doors
    elements, skipped_elements = Counter(), Counter()
    obstacles = []
    for e in f.by_type("IfcProduct"):
        if any(e.is_a(t) for t in SKIP_AS_OBSTACLE):
            continue
        if e.Representation is not None:
            obstacles.append(e)
        elif e.is_a("IfcElement"):
            skipped_elements[(e.is_a(), "no representation")] += 1
    occ = np.zeros(grid.shape, dtype=np.uint8)
    obstacle_mesh = meshes(f, obstacles)
    for e in obstacles:
        if e.id() not in obstacle_mesh:
            skipped_elements[(e.is_a(), "no 3D geometry")] += 1
            continue
        v, t = obstacle_mesh[e.id()]
        v = v - [0, 0, z0]
        if np.any(v.max(axis=0) < lo) or np.any(v.min(axis=0) > hi):
            skipped_elements[(e.is_a(), "outside the storey volume")] += 1
            continue
        elements[e.is_a()] += 1
        mark_surface(occ, origin, res, v, t, 1)
        fill_solid(occ, origin, res, v, t, 1)
    occupied = occ.astype(bool)
    report.elements, report.skipped_elements = dict(elements), dict(skipped_elements)

    # rooms
    graph = SceneGraph(voxels=grid)
    long_names = defaultdict(int)
    for sp in spaces:
        long_names[(sp.LongName or "").strip()] += 1
    free = np.zeros(grid.shape, dtype=bool)
    for sp in spaces:
        if sp.id() not in shifted:
            continue
        rid = ids[sp.id()]
        v, t = shifted[sp.id()]
        poly = polys[rid]
        ln = (sp.LongName or "").strip()
        unique = ln and long_names[ln] == 1
        room = Room(
            id=rid,
            name=ln if unique else rid,
            bounds=Box(tuple(float(c) for c in v.min(axis=0)), tuple(float(c) for c in v.max(axis=0))),
            aliases=tuple(a for a in (sp.GlobalId, f"{rid}호", f"{ln} {rid}" if ln and not unique else None) if a),
            polygon=tuple((round(x, 4), round(y, 4)) for x, y in list(poly.exterior.coords)[:-1]),
        )
        graph.add_room(room)
        free |= graph.room_mask(room)
    report.rooms = len(graph.rooms)

    # doors: the two rooms come from the space boundaries, or from the door position when
    # the door has no space boundary. Their openings become flyable.
    door_free = np.zeros(grid.shape, dtype=bool)
    candidates = []
    for d in f.by_type("IfcDoor"):
        bounded = [b.RelatingSpace for b in d.ProvidesBoundaries or [] if b.RelatingSpace is not None]
        rooms = sorted({ids[sp.id()] for sp in bounded if ids.get(sp.id()) in graph.rooms})
        if not rooms and ifcopenshell.util.element.get_container(d) != storey:
            continue  # a door of another storey
        candidates.append((d, bool(bounded), rooms, [r.RelatingOpeningElement for r in d.FillsVoids or []]))
    report.doors = len(candidates)
    door_mesh = meshes(f, [e for d, _, _, holes in candidates for e in (d, *holes)])
    for d, has_boundary, rooms, holes in candidates:
        label = f"door {(d.Name or 'unnamed').split(':')[-1]} ({d.GlobalId})"

        def skip(why: str) -> None:
            notes.append(f"{label} {why}")
            report.skipped_doors.append((label, why))

        geo = [door_mesh[h.id()] for h in holes if h.id() in door_mesh]
        if not geo and d.id() in door_mesh:
            geo = [door_mesh[d.id()]]
        geo = [(gv - [0, 0, z0], gt) for gv, gt in geo]
        how = ""
        if not has_boundary:
            report.doors_without_boundary += 1
            if not geo:
                skip("has no space boundary and no geometry; skipped")
                continue
            rooms, how = rooms_at_door(geo, polys), " (from geometry)"
        if len(rooms) == 1:
            skip(f"leads from {rooms[0]} to outside or an unmodelled space{how}; not a room connection")
            continue
        if len(rooms) > 2:
            skip(f"has space boundaries to {len(rooms)} rooms ({', '.join(rooms)}); expected 2, skipped")
            continue
        if not rooms:
            skip(f"has no room of this storey within {DOOR_REACH_M} m; skipped" if how else
                 "has no space boundary to a room of this storey; skipped")
            continue
        if not geo:
            skip("has no geometry; skipped")
            continue
        v = np.vstack([m[0] for m in geo])
        for gv, gt in geo:
            door_free |= solid_mask(grid, gv, gt)
        c = (v.min(axis=0) + v.max(axis=0)) / 2
        door_id = f"door_{rooms[0]}_{rooms[1]}"
        k = 2
        while door_id in graph.doors:
            door_id = f"door_{rooms[0]}_{rooms[1]}_{k}"
            k += 1
        graph.add_door(Door(door_id, rooms[0], rooms[1], (float(c[0]), float(c[1]), 1.0), door_width(d, scale)))
        if how:
            report.doors_by_geometry += 1
        else:
            report.doors_by_boundary += 1
    if report.doors_without_boundary:
        notes.append(f"{report.doors_without_boundary} of {report.doors} doors have no space boundary "
                     f"(IfcRelSpaceBoundary); {report.doors_by_geometry} were matched to two rooms from "
                     f"geometry. Check them, or export with space boundaries (Revit: 'Space boundaries: "
                     f"1st level')")

    # open connections: footprints that touch (no wall between the spaces). Pairs that
    # already have a door are skipped: their footprints often touch inside the doorway.
    names = sorted(polys)
    door_pairs = {frozenset((d.room_a, d.room_b)) for d in graph.doors.values()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset((a, b)) in door_pairs:
                continue
            shared = polys[a].boundary.intersection(polys[b].buffer(0.02))
            if shared.is_empty or shared.length < min_open_m:
                continue
            parts = list(shared.geoms) if isinstance(shared, MultiLineString) else [shared]
            seg = max((p for p in parts if isinstance(p, LineString)), key=lambda p: p.length, default=None)
            if seg is None:
                continue
            m = seg.interpolate(0.5, normalized=True)
            graph.add_door(Door(f"open_{a}_{b}", a, b, (float(m.x), float(m.y), 1.0), round(seg.length, 2)))
            notes.append(f"open connection {a} <-> {b}: shared edge {seg.length:.2f} m (no wall)")
            report.open_connections += 1

    # floor voids (atrium, stair well, slab opening): no floor under a free column -> not
    # flyable. The Flow deck measures height to the floor, so a sudden drop corrupts the estimate.
    k_floor = slice(max(0, grid.index((0, 0, -0.3))[2]), grid.index((0, 0, 0.05))[2] + 1)
    has_floor = occupied[:, :, k_floor].any(axis=2)
    void = free.any(axis=2) & ~has_floor
    for room in graph.rooms.values():
        m = graph.room_mask(room).any(axis=2) & void
        area = m.sum() * res * res
        if area >= 0.5:
            notes.append(f"room {room.id} ({room.name}): {area:.1f} m² without a floor below "
                         f"(void / stair well) -> left UNKNOWN, not flyable")
            report.voids.append((room.id, float(area)))
    free &= ~void[:, :, None]
    free |= door_free  # door openings: no floor check (the slab often stops under the wall)

    states = np.full(grid.shape, int(VoxelState.UNKNOWN), dtype=np.uint8)
    states[free] = int(VoxelState.FREE)
    states[occupied] = int(VoxelState.OCCUPIED)
    grid.states[...] = states
    graph.solids = voxel_solids(grid, occupied, grid.index((0.0, 0.0, FLOOR_SKIP_M))[2])
    notes.insert(0, f"storey {storey_name}: elevation {z0:.2f} m is z = 0; {len(obstacles)} obstacle elements")
    return graph, notes, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifc")
    ap.add_argument("folder")
    ap.add_argument("--storey", required=True, help="IfcBuildingStorey name, e.g. 4F")
    ap.add_argument("--res", type=float, default=0.1, help="voxel size in meters")
    ap.add_argument("--exclude", default=None,
                    help=r'regex on space names to leave out, e.g. "^(PS|EPS|EV)\d" for shafts')
    ap.add_argument("--report", action="store_true",
                    help="print a summary: element types, counts, what was skipped and why")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # room names may not fit the console code page
    try:
        graph, notes, report = convert_with_report(a.ifc, a.storey, a.res, exclude=a.exclude)
    except ConversionError as e:
        print(f"cannot convert: {e}")
        return 1
    save_graph(graph, a.folder)
    for n in notes:
        print("-", n)
    if a.report:
        print(report.format())
    print(f"saved {len(graph.rooms)} rooms, {len(graph.doors)} connections, voxels {graph.voxels.shape}, "
          f"{len(graph.solids)} solids to {a.folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
