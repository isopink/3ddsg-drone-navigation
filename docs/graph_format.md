# Scene graph exchange format

Planning, validation, simulation and visualization all work on a `SceneGraph` (`src/ddsg_nav/graph/model.py`).
The simplest way to get real data in is to export one floor as a folder in this format
(to ask for one, send [3ddsg_request.md](3ddsg_request.md)).
No database connection is needed after that.

```
my_floor/
  graph.json    rooms, doors, solids, voxel grid metadata
  voxels.npz    array "states", shape = voxels.shape, uint8
```

## graph.json

```json
{
  "format": "ddsg-scene-graph",
  "version": 1,
  "voxels": {"origin": [-0.1, -0.1, 0.0], "resolution": 0.1, "shape": [92, 32, 25], "file": "voxels.npz"},
  "rooms": [
    {"id": "R401", "name": "401호", "aliases": ["Room 401", "2O2Fr$t4X7"],
     "bounds": {"min": [0, 0, 0], "max": [3, 3, 2.5]},
     "polygon": null}
  ],
  "doors": [
    {"id": "door_1", "room_a": "R401", "room_b": "R302", "position": [3.0, 1.5, 1.0], "width": 1.0}
  ],
  "solids": [{"min": [2.9, 0, 0], "max": [3.1, 1.0, 2.5]}]
}
```

| Field | Meaning |
|---|---|
| `voxels.origin` | Corner of voxel (0, 0, 0) in meters. Voxel (i, j, k) covers `origin + [i, i+1) * resolution`. |
| `voxels.resolution` | Voxel edge in meters. 0.1 is detailed; 0.2 is about 20x faster in the worst case. |
| `rooms[].bounds` | Bounding box of the room, in meters. Required. |
| `rooms[].polygon` | Exact floor outline `[[x, y], ...]` for non-rectangular rooms, or `null`. |
| `rooms[].aliases` | Every other name a person might use for the room: room number, IFC GlobalId, ... A sentence like "401호에서 302호로" is matched against `id`, `name` and `aliases`. |
| `doors[]` | A connection between two rooms. `position` is the center of the opening, `width` its clear width. |
| `solids` | Boxes of walls and obstacles. Only the simulator uses them; planning uses the voxels. |

Voxel states in `voxels.npz`:

| Value | State | Meaning |
|---|---|---|
| 0 | unknown | not observed |
| 1 | free | flyable |
| 2 | occupied | wall, obstacle |

All coordinates (voxels, rooms, doors, solids) must be in one frame, in meters, z up.
The drone's own frame is related to it by `graph/transform.py`:
`Transform(translation=<take-off point>, yaw_deg=<direction of the nose at take-off>)`.

## What an exporter has to decide

1. **One floor at a time.** Pick the grid origin, extent and resolution so that it covers the floor.
2. **Merge element voxels into one grid.** Occupied wall and obstacle voxels become 2. Room interiors that were never contradicted become 1. Anything never observed stays 0.
3. **Unobserved voxels:** keep them as 0 and let the planner policy decide
   (`obstacle`, `free`, `free_in_rooms`; see `planner/policy.py`). Do not guess in the exporter.
4. **Doors:** one entry per connection between two rooms. If a door's voxels were never observed, still list the door.

## Check an export

```
cf-env\Scripts\python scripts\graph_tools.py validate my_floor
cf-env\Scripts\python scripts\graph_tools.py validate my_floor --policy free_in_rooms --height 0.8
cf-env\Scripts\python scripts\graph_tools.py route my_floor "401호에서 302호로 가줘"
```

`validate` reports errors (no flyable space in a room, a door inside an obstacle, a door the drone cannot reach,
rooms outside the grid), warnings (isolated rooms, disconnected groups, mostly unobserved map, narrow doors) and
statistics (grid size, occupied/unknown ratio).
Example files: `scripts\graph_tools.py export-fake <folder> [--obstacle | --large]`.

## Loading from Neo4j instead

`Neo4jGraphSource` (`src/ddsg_nav/graph/neo4j_source.py`) builds the same `SceneGraph` from the database
and follows the four rules above. The schema is described by a `Neo4jMapping`; it is the only thing to edit.

```python
from ddsg_nav.graph.io import save_graph
from ddsg_nav.graph.neo4j_source import Neo4jGraphSource, Neo4jMapping

mapping = Neo4jMapping(room_label="IfcSpace", room_floor="storey", length_unit="mm")
source = Neo4jGraphSource("bolt://host:7687", "neo4j", password, floor="4", mapping=mapping)
graph = source.load()            # needs: pip install -e ".[neo4j]"
print("\n".join(source.notes))   # what was skipped (other floors, doors to unknown rooms)
save_graph(graph, "my_floor")    # then validate the folder as above
```

| Mapping field | Default | Meaning |
|---|---|---|
| `room_label`, `door_label`, `voxel_label` | `Space`, `Door`, `Voxel` | Node labels. |
| `element_label`, `voxel_rel` | `Element`, `HAS_VOXEL` | Voxels are read as `(element)-[voxel_rel]->(voxel)`. `element_label=None`: every voxel node. |
| `door_room_rel` | `CONNECTS` | `(door)-[door_room_rel]-(room)`, two per door. |
| `door_room_props` | `None` | Instead of the relationship: two door properties that hold the room ids. |
| `room_id`, `room_name`, `room_aliases` | `id`, `name`, `(number, global_id)` | Id is required. Name falls back to the id. Aliases are optional (string or list). |
| `room_floor` | `floor` | Compared with `floor` as text, ignoring case. `None`: no floor filter. |
| `room_min`, `room_max` | `bbox_min`, `bbox_max` | Required. One list property, or three names such as `("min_x", "min_y", "min_z")`. |
| `room_polygon` | `polygon` | Optional outline, `[x0, y0, x1, y1, ...]`. `None`: boxes only. |
| `door_id`, `door_position` | `id`, `position` | Required. Position is the center of the opening (list, three names, or a Neo4j point). |
| `door_width`, `default_door_width` | `width`, `0.9` | Optional property; the default (meters) is used when it is missing. |
| `voxel_position`, `voxel_anchor` | `position`, `center` | Required. Anchor is `center` or `corner` (minimum corner). |
| `voxel_size` | `size` | Property name, or a number for one constant edge length. |
| `voxel_state`, `voxel_state_values`, `voxel_default_state` | `state`, `occupied/free/unknown` or `2/1/0`, occupied | A voxel without a state is occupied. `voxel_state=None`: all occupied. |
| `length_unit` | `m` | `m`, `cm`, `mm`, `ft`, `in`, or meters per unit. Applied to every length. |
| `grid_resolution` | `None` | Meters. `None`: the voxel size of the data (0.1 if there are no voxels). |
| `grid_margin` | `0.3` | Meters added around the rooms in x and y, so outer walls fit in the grid. |
| `rooms_are_free` | `True` | Room interiors start free; `False` leaves them unknown unless a voxel says free. |
| `build_solids` | `True` | Boxes covering the occupied voxels, for the simulator. |
| `rooms_query`, `doors_query`, `voxels_query` | `None` | Cypher that replaces a default query when the schema does not fit the fields above (for example voxels stored as indices). Must return `properties(n) AS props` (doors also `rooms`: the two room ids); `$floor` is a parameter. |

How the grid is made: extent = bounding box of the floor's rooms plus `grid_margin`, origin aligned with
the source voxels. Every cell starts unknown, room interiors become free, then source voxels are written:
free first, occupied last. A cell touched by an occupied voxel is occupied, also at a coarser resolution.

Errors name the label or property and the mapping field to change, for example
`room 'R401' (label 'Space'): property 'bbox_min' is missing or empty; set Neo4jMapping.room_min ...`.
Doors that join a room on another floor, an unknown room, or not exactly two rooms are skipped and listed
in `source.notes`.

The schema is not known yet, so the defaults are placeholders. The questions that fill the mapping are in
[3ddsg_request.md](3ddsg_request.md).
