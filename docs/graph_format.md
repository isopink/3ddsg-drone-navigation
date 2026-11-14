# Scene graph exchange format

Planning, validation, simulation and visualization all work on a `SceneGraph` (`src/ddsg_nav/graph/model.py`).
The simplest way to get real data in is to export one floor as a folder in this format.
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
