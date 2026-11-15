"""Convert one storey of an IFC (BIM) model into the scene graph exchange format.

    cf-env\\Scripts\\python scripts\\ifc_to_graph.py model.ifc data\\ifc_4F --storey 4F [--res 0.1]
        [--exclude "^(PS|EPS|EV)\\d"]   # leave out shafts

Needs `pip install ifcopenshell` (plus shapely, which it installs).

What it does (see docs/graph_format.md for the output):
  rooms   one per IfcSpace on the storey: id = Name ("401"), name = LongName ("카페"),
          polygon = floor footprint, bounds = space volume. GlobalId is an alias.
  doors   one per IfcDoor whose space boundaries touch two spaces. Doors are treated
          as OPEN: the door leaf is not an obstacle, the wall opening is flyable.
          Spaces whose footprints touch (room separation lines, no wall between them)
          get an "open_*" connection at the middle of the shared edge.
  voxels  every other building element (walls, columns, slabs, stairs, windows,
          ceilings, MEP, fixtures, ...) is OCCUPIED (its surface, sampled at res/2).
          Space interiors and door openings that are not occupied are FREE.
          Everything else stays UNKNOWN.
  frame   x, y as in the IFC (meters); z = 0 at the storey elevation.
"""
from __future__ import annotations

import argparse
import multiprocessing
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
    from shapely.geometry import LineString, MultiLineString, Polygon as ShPolygon
    from shapely.ops import unary_union
except ImportError:
    sys.exit("needs ifcopenshell: cf-env\\Scripts\\python -m pip install ifcopenshell")

from ddsg_nav.graph.io import save_graph
from ddsg_nav.graph.model import Box, Door, Room, SceneGraph, VoxelGrid, VoxelState, points_in_polygon

SKIP_AS_OBSTACLE = ("IfcSpace", "IfcOpeningElement", "IfcDoor", "IfcVirtualElement", "IfcAnnotation", "IfcGrid")


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
    """Set every voxel a triangle passes through (sampled at res/2) to `value`."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    edge = np.max(np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1),
                            np.linalg.norm(c - b, axis=1)]), axis=0)
    n = np.maximum(1, np.ceil(edge / (res / 2))).astype(int)
    shape = np.array(states.shape)
    for nv in np.unique(n):
        sel = np.nonzero(n == nv)[0]
        u, w = _bary(int(nv))
        for chunk in np.array_split(sel, max(1, len(sel) * len(u) // 2_000_000 + 1)):
            A, AB, AC = a[chunk], b[chunk] - a[chunk], c[chunk] - a[chunk]
            p = A[:, None] + AB[:, None] * u[None, :, None] + AC[:, None] * w[None, :, None]
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


# ---- IFC -> graph --------------------------------------------------------------

def door_width(door, scale: float) -> float:
    """Clear width in meters: from the type name ("D900x2100") if present, else OverallWidth."""
    m = re.search(r"(\d{3,4})x(\d{3,4})", door.Name or "")
    if m:
        return int(m.group(1)) / 1000
    return float(door.OverallWidth or 0.9 / scale) * scale


def convert(path: str, storey_name: str, res: float, pad: float = 0.4,
            min_open_m: float = 0.7, exclude: str | None = None) -> tuple[SceneGraph, list[str]]:
    import ifcopenshell.util.unit as uu

    f = ifcopenshell.open(path)
    notes: list[str] = []
    scale = uu.calculate_unit_scale(f)  # file length unit -> meters
    storeys = {s.Name: s for s in f.by_type("IfcBuildingStorey")}
    if storey_name not in storeys:
        raise SystemExit(f"storey {storey_name!r} not found; available: {', '.join(storeys)}")
    storey = storeys[storey_name]
    z0 = float(storey.Elevation or 0.0) * scale

    spaces = [o for r in storey.IsDecomposedBy or [] for o in r.RelatedObjects if o.is_a("IfcSpace")]
    if exclude:
        dropped = [s.Name for s in spaces if re.search(exclude, s.Name or "")]
        spaces = [s for s in spaces if not re.search(exclude, s.Name or "")]
        notes.append(f"excluded spaces ({exclude}): {', '.join(dropped) or 'none'}")
    if not spaces:
        raise SystemExit(f"storey {storey_name} has no IfcSpace; rooms cannot be extracted")

    space_mesh = meshes(f, spaces)
    shifted = {k: (v - [0, 0, z0], t) for k, (v, t) in space_mesh.items()}

    # grid extent: all spaces + padding
    allv = np.vstack([v for v, _ in shifted.values()])
    lo = np.floor((allv.min(axis=0) - pad) / res) * res
    hi = np.ceil((allv.max(axis=0) + pad) / res) * res
    lo[2] = min(lo[2], -2 * res)
    size = hi - lo
    grid = VoxelGrid.empty(tuple(lo), tuple(size), res, VoxelState.UNKNOWN)
    origin = np.array(grid.origin)

    # obstacles: every product with geometry except spaces, openings and doors
    obstacles = [e for e in f.by_type("IfcProduct")
                 if e.Representation is not None and not any(e.is_a(t) for t in SKIP_AS_OBSTACLE)]
    occ = np.zeros(grid.shape, dtype=np.uint8)
    for v, t in meshes(f, obstacles).values():
        v = v - [0, 0, z0]
        if np.any(v.max(axis=0) < lo) or np.any(v.min(axis=0) > hi):
            continue
        mark_surface(occ, origin, res, v, t, 1)
        fill_solid(occ, origin, res, v, t, 1)
    occupied = occ.astype(bool)

    # rooms
    graph = SceneGraph(voxels=grid)
    long_names = defaultdict(int)
    for sp in spaces:
        long_names[(sp.LongName or "").strip()] += 1
    polys = {}
    free = np.zeros(grid.shape, dtype=bool)
    for sp in spaces:
        if sp.id() not in shifted:
            notes.append(f"space {sp.Name} has no geometry; skipped")
            continue
        v, t = shifted[sp.id()]
        poly = footprint(v, t)
        if poly.geom_type == "MultiPolygon":
            notes.append(f"space {sp.Name}: footprint has {len(poly.geoms)} parts; using the largest")
            poly = max(poly.geoms, key=lambda p: p.area)
        poly = poly.simplify(0.005)
        polys[sp.Name] = poly
        ln = (sp.LongName or "").strip()
        unique = ln and long_names[ln] == 1
        room = Room(
            id=sp.Name,
            name=ln if unique else sp.Name,
            bounds=Box(tuple(v.min(axis=0)), tuple(v.max(axis=0))),
            aliases=tuple(a for a in (sp.GlobalId, f"{sp.Name}호", f"{ln} {sp.Name}" if ln and not unique else None) if a),
            polygon=tuple((round(x, 4), round(y, 4)) for x, y in list(poly.exterior.coords)[:-1]),
        )
        graph.add_room(room)
        free |= graph.room_mask(room)

    # doors (via space boundaries); their openings become flyable
    door_free = np.zeros(grid.shape, dtype=bool)
    for d in f.by_type("IfcDoor"):
        rooms = sorted({b.RelatingSpace.Name for b in d.ProvidesBoundaries or []
                        if b.RelatingSpace.Name in graph.rooms})
        if len(rooms) != 2:
            what = "outside or an unmodelled space" if len(rooms) == 1 else "no space"
            notes.append(f"door {d.Name.split(':')[-1]} ({d.GlobalId}) connects {rooms or 'nothing'} "
                         f"to {what}; not a room connection")
            continue
        holes = [r.RelatingOpeningElement for r in d.FillsVoids or []]
        geo = meshes(f, holes or [d])
        if not geo:
            notes.append(f"door {d.GlobalId}: no geometry; skipped")
            continue
        v = np.vstack([m[0] for m in geo.values()]) - [0, 0, z0]
        for gv, gt in geo.values():
            door_free |= solid_mask(grid, gv - [0, 0, z0], gt)
        c = (v.min(axis=0) + v.max(axis=0)) / 2
        door_id = f"door_{rooms[0]}_{rooms[1]}"
        k = 2
        while door_id in graph.doors:
            door_id = f"door_{rooms[0]}_{rooms[1]}_{k}"
            k += 1
        graph.add_door(Door(door_id, rooms[0], rooms[1], (float(c[0]), float(c[1]), 1.0), door_width(d, scale)))

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
            graph.add_door(Door(f"open_{a}_{b}", a, b, (m.x, m.y, 1.0), round(seg.length, 2)))
            notes.append(f"open connection {a} <-> {b}: shared edge {seg.length:.2f} m (no wall)")

    # floor voids (atrium, stair well): no floor under a free column -> not flyable.
    # The Flow deck measures height to the floor, so a sudden drop corrupts the estimate.
    k_floor = slice(max(0, grid.index((0, 0, -0.3))[2]), grid.index((0, 0, 0.05))[2] + 1)
    has_floor = occupied[:, :, k_floor].any(axis=2)
    void = free.any(axis=2) & ~has_floor
    for room in graph.rooms.values():
        m = graph.room_mask(room).any(axis=2) & void
        area = m.sum() * res * res
        if area >= 0.5:
            notes.append(f"room {room.id} ({room.name}): {area:.1f} m² without a floor below "
                         f"(void / stair well) -> left UNKNOWN, not flyable")
    free &= ~void[:, :, None]
    free |= door_free  # door openings: no floor check (the slab often stops under the wall)

    states = np.full(grid.shape, int(VoxelState.UNKNOWN), dtype=np.uint8)
    states[free] = int(VoxelState.FREE)
    states[occupied] = int(VoxelState.OCCUPIED)
    grid.states[...] = states
    notes.insert(0, f"storey {storey_name}: elevation {z0:.2f} m is z = 0; {len(obstacles)} obstacle elements")
    return graph, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifc")
    ap.add_argument("folder")
    ap.add_argument("--storey", required=True, help="IfcBuildingStorey name, e.g. 4F")
    ap.add_argument("--res", type=float, default=0.1, help="voxel size in meters")
    ap.add_argument("--exclude", default=None,
                    help=r'regex on space names to leave out, e.g. "^(PS|EPS|EV)\d" for shafts')
    a = ap.parse_args()
    graph, notes = convert(a.ifc, a.storey, a.res, exclude=a.exclude)
    save_graph(graph, a.folder)
    for n in notes:
        print("-", n)
    print(f"saved {len(graph.rooms)} rooms, {len(graph.doors)} connections, voxels {graph.voxels.shape} "
          f"to {a.folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
