"""Neo4j graph source. NOT implemented: the database schema is not known yet.

What this adapter has to produce (the same SceneGraph as the fake graph and the file
export), from one floor of the building graph:

* rooms     IFC spaces of the floor -> `Room(id, name, bounds, aliases, polygon)`.
            Put every name people may use in `aliases` (room number, IFC GlobalId, ...).
* doors     connectivity relations between two spaces -> `Door(position, width)`.
* voxels    building-element voxels are stored per element in the database. Merge them
            into ONE regular grid: occupied wall/obstacle voxels -> OCCUPIED, room
            interiors -> FREE, anything never observed -> UNKNOWN. Pick the grid
            resolution and extent to cover the floor.
* solids    wall/obstacle boxes (only used by the simulator).
* frame     coordinates must be expressed in ONE frame; the drone-to-graph Transform
            (graph/transform.py) is applied at flight time.

Until the schema is known, the practical route is: export the floor in the format of
graph/io.py (docs/graph_format.md) and use `FileGraphSource`, then check it with
`scripts/graph_tools.py validate`.
"""
from __future__ import annotations

from .model import SceneGraph


class Neo4jGraphSource:
    def __init__(self, uri: str, user: str, password: str, floor: str) -> None:
        self.uri, self.user, self.floor = uri, user, floor
        self._password = password

    def load(self) -> SceneGraph:
        raise NotImplementedError(
            "Neo4j schema not known yet: export the floor with graph/io.py and use FileGraphSource"
        )
