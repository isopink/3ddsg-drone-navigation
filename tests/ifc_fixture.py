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
declaration, no space boundaries, a room without a door.
"""
from __future__ import annotations

from pathlib import Path

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


def build_ifc(path: str | Path, unit: str = "m", declared_unit: str | None = None,
              space_boundaries: bool = True, lab_door: bool = True) -> Path:
    """Write the building to `path`. Coordinates are in `unit` ("m" or "mm"); the file
    declares `declared_unit` (default: the same). `space_boundaries=False` leaves out
    every IfcRelSpaceBoundary; `lab_door=False` leaves room 204 without a door."""
    k = {"m": 1.0, "mm": 1000.0}[unit]
    declared = {"m": "METERS", "mm": "MILLIMETERS"}[declared_unit or unit]
    run = ifcopenshell.api.run
    f = run("project.create_file", version="IFC4")
    project = run("root.create_entity", f, ifc_class="IfcProject", name="Synthetic test building")
    run("unit.assign_unit", f, length={"is_metric": True, "raw": declared})
    model = run("context.add_context", f, context_type="Model")
    body = run("context.add_context", f, context_type="Model", context_identifier="Body",
               target_view="MODEL_VIEW", parent=model)

    def placement(x: float, y: float, z: float, relative_to=None):
        point = f.createIfcCartesianPoint((x * k, y * k, z * k))
        return f.createIfcLocalPlacement(relative_to, f.createIfcAxis2Placement3D(point, None, None))

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
        s.ObjectPlacement = placement(ORIGIN[0], ORIGIN[1], elevation)
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
    run("aggregate.assign_object", f, products=list(spaces.values()), relating_object=storey)

    built = [element("IfcSlab", "Floor", box(-0.2, -0.2, 9.2, 5.4, -SLAB_DEPTH, 0.0)),
             element("IfcColumn", "Column", box(*COLUMN, 0.0, WALL_HEIGHT))]
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
