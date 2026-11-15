"""Build a small synthetic IFC with ifcopenshell (test helper; nothing binary is committed).

One storey "3F" far from the origin, 2.8 m walls on a slab, plus an empty storey "2F":

    y 5.2 +-------+-------+--------+
          |  201  |  202  |  204   |     201 Office, 202 Meeting, 204 Lab
          |       |       |        |
      2.2 +--D1---+--D2---+---D3---+     D1..D3: doors with space boundaries
          |  203 Corridor :  205   |     203 | 205: no wall (open connection)
      0.0 +---------------+---D4---+     D4: exterior door (one boundary only)
          0      3        6        9 x   a column stands at the center of 205

Options make the files a real export can produce: millimeter units, a wrong unit
declaration, no space boundaries, a room without a door, spaces without usable geometry,
a rotated building, another offset from the origin, a stair and a slab opening.
"""
from __future__ import annotations

from pathlib import Path

import math

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.guid

STOREY = "3F"
EMPTY_STOREY = "2F"
ORIGIN = (1250.0, 830.0)  # storey position in the world, meters
ELEVATION = 7.2
ROOMS = {  # name: (long name, x0, y0, x1, y1), storey coordinates in meters
    "201": ("Office", 0.0, 2.2, 2.9, 5.2),
    "202": ("Meeting", 3.1, 2.2, 6.0, 5.2),
    "203": ("Corridor", 0.0, 0.0, 6.0, 2.0),
    "204": ("Lab", 6.2, 2.2, 9.0, 5.2),
    "205": ("Lobby", 6.0, 0.0, 9.0, 2.0),
}
WALLS = {  # name: (x0, y0, x1, y1)
    "south": (-0.2, -0.2, 9.2, 0.0),
    "north": (-0.2, 5.2, 9.2, 5.4),
    "west": (-0.2, 0.0, 0.0, 5.2),
    "east": (9.0, 0.0, 9.2, 5.2),
    "middle": (0.0, 2.0, 9.0, 2.2),
    "partition_1": (2.9, 2.2, 3.1, 5.2),
    "partition_2": (6.0, 2.2, 6.2, 5.2),
}
DOORS = {  # name: (wall, x0, x1, rooms it bounds)
    "D1": ("middle", 1.0, 1.9, ("201", "203")),
    "D2": ("middle", 4.1, 5.0, ("202", "203")),
    "D3": ("middle", 7.1, 8.0, ("204", "205")),
    "D4": ("south", 7.0, 7.9, ("205",)),
}
COLUMN = (7.3, 0.8, 7.7, 1.2)
WALL_HEIGHT, SPACE_HEIGHT, DOOR_HEIGHT, SLAB_DEPTH = 2.8, 2.6, 2.1, 0.2
BAD_SPACES = ("206", "207")  # 206: no geometry at all, 207: a sliver with almost no floor area
SLIVER = (0.0, 5.3, 3.0, 5.3005)
STAIR = (3.2, 4.3, 4.1, 5.1)  # in a corner of 202: three steps going up in +x
STAIR_RISE = 0.2
SLAB_VOID = (4.8, 4.0, 5.8, 5.0)  # hole in the floor slab, in another corner of 202


def world(x: float, y: float, origin: tuple[float, float] = ORIGIN, rotation_deg: float = 0.0):
    """Storey coordinates -> IFC world coordinates (meters)."""
    c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    return (origin[0] + c * x - s * y, origin[1] + s * x + c * y)


def build_ifc(path: str | Path, unit: str = "m", declared_unit: str | None = None,
              space_boundaries: bool = True, lab_door: bool = True, bad_spaces: bool = False,
              origin: tuple[float, float] = ORIGIN, rotation_deg: float = 0.0,
              stair_and_void: bool = False) -> Path:
    """Write the building to `path`. Coordinates are in `unit` ("m" or "mm"); the file
    declares `declared_unit` (default: the same). `space_boundaries=False` leaves out
    every IfcRelSpaceBoundary; `lab_door=False` leaves room 204 without a door;
    `bad_spaces=True` adds the two BAD_SPACES; `origin` and `rotation_deg` place and turn
    the storey in the world (rotation about its origin, around z); `stair_and_void=True`
    adds STAIR (an IfcStair) and SLAB_VOID (an opening in the floor slab) to room 202."""
    k = {"m": 1.0, "mm": 1000.0}[unit]
    declared = {"m": "METERS", "mm": "MILLIMETERS"}[declared_unit or unit]
    run = ifcopenshell.api.run
    f = run("project.create_file", version="IFC4")
    project = run("root.create_entity", f, ifc_class="IfcProject", name="Synthetic test building")
    run("unit.assign_unit", f, length={"is_metric": True, "raw": declared})
    model = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=model)

    def placement(x: float, y: float, z: float, relative_to=None, turn_deg: float = 0.0):
        point = f.createIfcCartesianPoint((x * k, y * k, z * k))
        axis = ref = None
        if turn_deg:
            a = math.radians(turn_deg)
            axis, ref = f.createIfcDirection((0.0, 0.0, 1.0)), f.createIfcDirection((math.cos(a), math.sin(a), 0.0))
        return f.createIfcLocalPlacement(relative_to, f.createIfcAxis2Placement3D(point, axis, ref))

    def box(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float):
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        outline = f.createIfcPolyline([f.createIfcCartesianPoint((x * k, y * k)) for x, y in corners])
        profile = f.createIfcArbitraryClosedProfileDef("AREA", None, outline)
        base = f.createIfcAxis2Placement3D(f.createIfcCartesianPoint((0.0, 0.0, z0 * k)), None, None)
        solid = f.createIfcExtrudedAreaSolid(profile, base, f.createIfcDirection((0.0, 0.0, 1.0)), (z1 - z0) * k)
        shape = f.createIfcShapeRepresentation(body, "Body", "SweptSolid", [solid])
        return f.createIfcProductDefinitionShape(None, None, [shape])

    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    building = run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    run("aggregate.assign_object", f, products=[site], relating_object=project)
    run("aggregate.assign_object", f, products=[building], relating_object=site)
    storeys = {}
    for name, elevation in ((EMPTY_STOREY, ELEVATION - 3.6), (STOREY, ELEVATION)):
        s = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=name)
        s.Elevation = elevation * k
        s.ObjectPlacement = placement(origin[0], origin[1], elevation, turn_deg=rotation_deg)
        storeys[name] = s
    run("aggregate.assign_object", f, products=list(storeys.values()), relating_object=building)
    storey = storeys[STOREY]

    def element(ifc_class: str, name: str, shape):
        e = run("root.create_entity", f, ifc_class=ifc_class, name=name)
        e.ObjectPlacement = placement(0.0, 0.0, 0.0, storey.ObjectPlacement)
        e.Representation = shape
        return e

    spaces = {}
    for name, (long_name, x0, y0, x1, y1) in ROOMS.items():
        sp = element("IfcSpace", name, box(x0, y0, x1, y1, 0.0, SPACE_HEIGHT))
        sp.LongName = long_name
        spaces[name] = sp
    extra = []
    if bad_spaces:
        extra = [element("IfcSpace", BAD_SPACES[0], None),
                 element("IfcSpace", BAD_SPACES[1], box(*SLIVER, 0.0, SPACE_HEIGHT))]
    run("aggregate.assign_object", f, products=[*spaces.values(), *extra], relating_object=storey)

    slab = element("IfcSlab", "Floor", box(-0.2, -0.2, 9.2, 5.4, -SLAB_DEPTH, 0.0))
    built = [slab, element("IfcColumn", "Column", box(*COLUMN, 0.0, WALL_HEIGHT))]
    if stair_and_void:
        hole = element("IfcOpeningElement", "Slab opening", box(*SLAB_VOID, -SLAB_DEPTH - 0.1, 0.1))
        run("feature.add_feature", f, feature=hole, element=slab)
        x0, y0, x1, y1 = STAIR
        tread = (x1 - x0) / 3
        for i in range(3):
            built.append(element("IfcStair", f"Stair:step {i + 1}",
                                 box(x0 + i * tread, y0, x0 + (i + 1) * tread, y1, 0.0, (i + 1) * STAIR_RISE)))
    walls = {}
    for name, rect in WALLS.items():
        walls[name] = element("IfcWall", f"Basic Wall:{name}", box(*rect, 0.0, WALL_HEIGHT))
    built += walls.values()

    for name, (wall, x0, x1, rooms) in DOORS.items():
        if name == "D3" and not lab_door:
            continue
        _, y0, _, y1 = WALLS[wall]
        opening = element("IfcOpeningElement", f"{name} opening", box(x0, y0, x1, y1, 0.0, DOOR_HEIGHT))
        run("feature.add_feature", f, feature=opening, element=walls[wall])
        mid = (y0 + y1) / 2
        door = element("IfcDoor", f"Single-Flush:{round((x1 - x0) * 1000)}x{round(DOOR_HEIGHT * 1000)}:{name}",
                       box(x0, mid - 0.02, x1, mid + 0.02, 0.0, DOOR_HEIGHT))
        door.OverallWidth, door.OverallHeight = (x1 - x0) * k, DOOR_HEIGHT * k
        run("feature.add_filling", f, opening=opening, element=door)
        built.append(door)
        if space_boundaries:
            for room in rooms:
                f.createIfcRelSpaceBoundary(ifcopenshell.guid.new(), None, "1stLevel", None, spaces[room], door,
                                            None, "PHYSICAL", "INTERNAL" if len(rooms) == 2 else "EXTERNAL")
    run("spatial.assign_container", f, products=built, relating_structure=storey)

    path = Path(path)
    f.write(str(path))
    return path
