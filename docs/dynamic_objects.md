# Dynamic objects and re-planning (design)

Design for issue #13. The prototype follows after #25, which provides the simulated
Multi-ranger sensor (five laser ranges: front, back, left, right, up; about 4 m).

## Problem

The route is planned once on the `SceneGraph` and then flown blindly. If reality differs
from the graph (a box in the hallway, a closed door), the only reaction today is the
abort planned in #25: land when a range drops below 0.2 m. This design adds the step
before that: notice the difference early, update a private copy of the map, and fly
around it.

## Which changes are handled

| Change | Version 1 | Reaction |
|---|---|---|
| New obstacle on the route (box, furniture, a person standing) | yes | local detour |
| Closed or blocked door | yes | route through other rooms, or land |
| Moving person | partly | treated as a static obstacle where it was seen |
| Obstacle that disappeared (map says occupied, space is free) | no | the map is only ever made stricter |
| Anything below the drone, or outside the five beams | no | not observable with this deck |

Simplest options taken:

- Observed obstacles are never removed during a flight and are thrown away when the
  flight ends. Nothing is written back to the graph or to Neo4j (that belongs to #12).
- No tracking or prediction of moving objects.

## How changes are detected

Input is the `Status` of the backend: position plus the five ranges from #25.

1. For each beam, compute the **expected range**: cast a ray from the current position
   through the voxel grid until the first `OCCUPIED` voxel (or 4 m).
2. A beam reports a **new obstacle** when the measured range is valid (below 4 m) and
   shorter than the expected range by more than a tolerance (start: 0.3 m, which covers
   position error and one or two voxels).
3. The hit point is `position + direction * range`. Its voxel is marked `OCCUPIED` in
   the overlay. Planning inflates it by the usual safety margin (0.25 m), the same as any
   mapped obstacle.

A shorter reading inside `UNKNOWN` space also counts as a new obstacle. A longer reading
than expected is ignored (see the table above).

Assumptions, to be confirmed in the prototype:

- The drone does not yaw, so the beams point along the graph axes. If it does yaw,
  `Status` needs a yaw value; that is a change to #25's file, not to this package.
- `Status` carries the ranges as an optional field. If it is missing (Crazyflie backend
  before #32), detection returns nothing and the flight behaves as today.

**Sampling.** The monitor runs only at waypoints, and simplified paths have long
segments. The dynamic flight loop therefore re-samples the path with the existing
`sample_path` at 0.5 m, so the map is checked at least every 0.5 m. With a 4 m beam and
0.2 m/s this leaves several looks at an obstacle before reaching it.

## When to re-plan

After every detection step that added something:

1. **Is the rest of the route still free?** Run `check_path` on the current position
   plus the remaining waypoints against the updated map. If it is fine (the obstacle is
   beside the route), keep flying. No re-plan.
2. **Local detour.** Otherwise hover and plan a voxel path from the current position to
   the next keypoint (door center or goal), then continue with the old route from there.
3. **Closed door.** If step 2 raises `NoPathError` and the next keypoint is a door, mark
   that door as blocked, re-run the global planner from the current room to the goal
   room without it, and plan the new legs from the current position.
4. **Give up.** Land with `FlightAbort` if no room route is left (`NoRoomRouteError`), if
   the goal itself is unreachable, or after 3 re-plans in one flight (simplest guard
   against oscillation).

After a re-plan, `FlightMonitor.path` is set to the new waypoints, so the deviation check
follows the new route. The 0.2 m abort from #25 stays active as the last line of
defence; re-planning should always trigger well before it.

## Package `src/ddsg_nav/dynamic/`

New files only. `graph/model.py`, `planner/` and `flight/` are not modified; the package
calls them.

**`view.py`: the wrapped graph**

- `DynamicGraph(base: SceneGraph)` keeps `base` untouched and owns a copy of the voxel
  states. It offers the attributes and methods the planners use (`voxels`, `rooms`,
  `doors`, `find_room`, `room_at`, `room_mask`, `room_graph`, `floor_z`,
  `door_floor_z`), so `plan_rooms`, `plan_voxel_path`, `blocked_mask` and `check_path`
  accept it unchanged.
- `add_obstacle(point: Point) -> bool`: mark the voxel occupied; `True` if it was new.
- `block_door(door_id: str) -> None`: `room_graph()` then omits that door's edge.
- `changes: list[Change]`: everything observed in this flight, for logs and tests.

**`detect.py`: sensor against map**

- `Change(kind: "obstacle" | "door", point: Point | None, door_id: str | None)`
- `expected_range(grid: VoxelGrid, position: Point, direction: Point, max_range=4.0) -> float`
- `detect(view: DynamicGraph, status: Status, tolerance_m=0.3) -> list[Change]`: applies
  the rules above, updates `view`, returns what was new.

**`replan.py`: decisions**

- `route_blocked(view, position: Point, remaining: Sequence[Point], inflation_m=0.25) -> bool`
- `replan(view, position: Point, plan: RoutePlan, next_keypoint: int, goal: str, height=0.5, inflation_m=0.25, unknown_is_obstacle=True) -> RoutePlan`:
  steps 2 and 3; raises `NoPathError` or `NoRoomRouteError` when nothing is left.

**`flight.py`: the loop**

- `fly_with_replanning(backend, graph, start, goal, velocity=0.2, monitor=None, max_replans=3, ...) -> FlightResult`
- `FlightResult(reached: bool, replans: int, changes: list[Change], waypoints_flown: list[Point])`

It has its own loop instead of calling `follow_path`, because `follow_path` always lands
when it returns and a re-plan must happen while hovering. It uses the same backend calls,
the same monitor checks and the same "always land" guarantee.

## Prototype and test (after #25)

A fake-graph flight in MuJoCo with a box that is in the simulator but not in the graph,
placed on the planned route. The test passes when the drone reaches the goal room, at
least one re-plan happened, and the flown trajectory keeps the safety margin from the
box. A second test closes the only door and expects a clean landing with `FlightAbort`.

## Known limits

- Five thin beams: an obstacle between beams, or lower than the flight height, is not
  seen until the 0.2 m abort, or not at all.
- Hit points inherit the position error; with poor positioning, obstacles land in the
  wrong voxel. The tolerance and the safety margin absorb about 0.3 m.
- A person who walks away leaves a phantom obstacle until the flight ends.
