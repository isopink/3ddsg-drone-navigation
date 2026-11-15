# Data request: the 3DDSG for drone navigation

To: whoever builds the 3D dynamic scene graph (3DDSG) of the building.
From: the drone navigation team (`3ddsg-drone-navigation`).

We plan drone routes from room to room on your graph. This page lists what we need,
the two ways to deliver it, and the questions whose answers let us load it.

## What we need

One floor is enough to start (the floor of the demo). For that floor:

| Item | Needed | Notes |
|---|---|---|
| Rooms | id, bounding box (min and max corner) | Also, if available: name, room number, IFC GlobalId, floor outline (polygon) for rooms that are not rectangular. |
| Doors | id, the two rooms it joins, center of the opening | Also, if available: clear width. Openings without a door leaf count as doors. |
| Voxels | position, size, state (occupied / free / unknown) | Walls, columns, furniture and any other obstacle. Per element is fine; we merge them into one grid. |
| Floor id | on every room | So we can select one floor. On doors and voxels too, if available. |
| One frame | all coordinates in the same frame, z up | Rooms, doors and voxels together. Tell us where the origin is and where x points. |
| Units | one length unit | Meters preferred. Millimeters are fine if we know. |

Not needed: materials, element types beyond "is it an obstacle", history, other floors.

## How to deliver it

Either option works. Option A is faster for us to start with.

**A. An export folder.** Two files, described in [graph_format.md](graph_format.md):
`graph.json` (rooms, doors, grid metadata) and `voxels.npz` (the voxel states). We check
it with `scripts/graph_tools.py validate <folder>` and send the report back.

**B. Neo4j access.** Read-only access (URI, user, password, database name) or a dump
(`neo4j-admin database dump`) with the Neo4j version. We load it with a mapping from
your schema to ours, which the answers below fill in.

Please do not send building data by public channels; the repository is public and we
keep data out of it.

## Questions

Short answers are enough. A small example (one room, one door, a few voxels, as Cypher
or a screenshot of the browser) answers most of them at once.

**Schema**

1. Which label do room nodes have? Which label do door nodes have?
2. How is a door linked to its two rooms: a relationship (which type?) or properties on
   the door (which names?)
3. Which label do voxel nodes have, and how are they linked to their building element
   (relationship type, element label)? Or are voxels stored as an array on the element?

**Rooms**

4. Which property is the unique room id? Which hold the name, the room number, the IFC
   GlobalId?
5. How is the room geometry stored: bounding box (which properties, one list or six
   numbers?), outline polygon, both?
6. Which property holds the floor, and what are its values (for example `4`, `"4F"`,
   an elevation)?

**Doors**

7. Which property holds the door position, and is it the center of the opening?
8. Is the clear width stored? Which property?

**Voxels**

9. Is the voxel position its center or its minimum corner? Coordinates or integer
   indices (if indices: grid origin and step)?
10. What is the voxel size? One size for the whole building, or per voxel (which
    property)?
11. How is the state stored (property name and values)? Are free and unobserved voxels
    stored, or only occupied ones?
12. How many voxels does one floor have, roughly?

**Frame and units**

13. Length unit of all coordinates?
14. Where is the origin, which way do x and y point, is z up? Is it the IFC project
    frame?
15. Floor level (z) and ceiling height of the demo floor?

**Dynamics**

16. Which parts of the graph change over time (doors open or closed, moved furniture,
    people), and how would we see a change (timestamp, version, state property)?

## What you get back

A validation report for the floor (rooms without flyable space, doors inside
obstacles, unreachable rooms, share of unobserved voxels) and example routes between
rooms, so problems in the data show up early.
