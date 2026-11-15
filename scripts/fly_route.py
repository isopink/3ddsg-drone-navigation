"""Plan a route and fly it, in simulation or on the real drone.

The map is either virtual rooms (--rooms) or a saved graph folder (--graph, the exchange
format of docs/graph_format.md).

The virtual rooms and doors only exist in the map. Size them to the space you can really
fly in, put the drone at the center of the start room with its nose along +x (or pass
--yaw), and the planner keeps it inside the rooms and routes it through the doors.

Run from the repository root:
    # simulation, always do this first
    cf-env\\Scripts\\python scripts\\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
    # real drone: runs the simulation as a dry run first, then asks you to confirm
    cf-env\\Scripts\\python scripts\\fly_route.py --backend crazyflie --rooms 2 2 --depth 2 "A -> B"
    # a saved graph: the script prints where to put the drone; --start X Y if it stands elsewhere
    cf-env\\Scripts\\python scripts\\fly_route.py --backend sim --graph <folder> "from A to C"

Fly in a bright room over a textured floor (hover-test lesson), 30 cm to 1 m high.
Ctrl+C lands the drone.
"""
import argparse
import dataclasses
import math
import sys
import time

from ddsg_nav.flight.compare import compare_to_plan
from ddsg_nav.flight.executor import CHECK_STEP_M, follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.llm_goal import llm_from_env
from ddsg_nav.graph.fake import build_row_graph
from ddsg_nav.graph.io import GraphFormatError, load_graph
from ddsg_nav.graph.model import Box
from ddsg_nav.graph.transform import Transform
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.global_planner import NoRoomRouteError
from ddsg_nav.planner.local_planner import NoPathError, plan_voxel_path
from ddsg_nav.planner.policy import UnknownPolicy, blocked_mask
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph


# Real drone only: the Flow deck v2 measures height from 0.2 to 3.0 m above the floor.
MIN_REAL_HEIGHT = 0.2
MAX_REAL_HEIGHT = 3.0


def real_flight_problem(graph, plan) -> str | None:
    """Why the real drone must not fly this plan, or None."""
    floors = sorted({round(graph.floor_z(r), 3) for r in plan.rooms})
    if len(floors) > 1:
        return f"the route changes floors (floor levels {floors} m): stairs are not ready on the real drone"
    heights = [p[2] - floors[0] for p in plan.waypoints]
    lo, hi = min(heights), max(heights)
    if lo < MIN_REAL_HEIGHT - 1e-9 or hi > MAX_REAL_HEIGHT + 1e-9:
        return (f"the route flies {lo:.2f} to {hi:.2f} m above the floor, outside the "
                f"{MIN_REAL_HEIGHT} to {MAX_REAL_HEIGHT} m range of the Flow deck")
    return None


def plan_from(graph, plan, xy, height: float, margin: float, policy):
    """The same route, starting at `xy` in the start room instead of the planner's start point."""
    room = graph.rooms[plan.rooms[0]]
    start = (float(xy[0]), float(xy[1]), graph.floor_z(room) + height)
    if not room.contains(start):
        raise NoPathError(f"--start {start[0]:g} {start[1]:g} is not inside the start room {room.id}")
    target = plan.keypoints[1]  # the first door, or the goal
    if target not in plan.waypoints:
        raise NoPathError("cannot re-plan the first leg from --start")
    leg = plan_voxel_path(graph.voxels, start, target, blocked=blocked_mask(graph, margin, policy))
    rest = plan.waypoints[plan.waypoints.index(target) + 1:]
    return dataclasses.replace(plan, keypoints=[start] + plan.keypoints[1:], waypoints=leg + rest)


def _side(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "+x" if dx >= 0 else "-x"
    return "+y" if dy >= 0 else "-y"


def describe_start(graph, room_id: str, point, yaw: float) -> str:
    """Where to put the drone, for a person: room, the two nearest walls, the nearest door, nose."""
    room = graph.rooms[room_id]
    x, y = point[0], point[1]
    if room.polygon:
        corners = list(room.polygon)
    else:
        (x0, y0, _), (x1, y1, _) = room.bounds.min, room.bounds.max
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        ex, ey = bx - ax, by - ay
        t = max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / ((ex * ex + ey * ey) or 1.0)))
        qx, qy = ax + t * ex, ay + t * ey
        walls.append((math.hypot(qx - x, qy - y), _side(qx - x, qy - y)))
    walls.sort(key=lambda w: w[0])
    name = room.id if room.name == room.id else f"{room.name} ({room.id})"
    parts = [f"{d:.2f} m from the {side} wall" for d, side in walls[:2]]
    doors = [d for d in graph.doors.values() if room_id in (d.room_a, d.room_b)]
    if doors:
        door = min(doors, key=lambda d: math.hypot(d.position[0] - x, d.position[1] - y))
        dx, dy = door.position[0] - x, door.position[1] - y
        other = door.room_b if door.room_a == room_id else door.room_a
        parts.append(f"{math.hypot(dx, dy):.2f} m from the door to {other} (toward {_side(dx, dy)})")
    return (f"in room {name} at x={x:.2f} y={y:.2f}: {', '.join(parts)}. "
            f"Nose {yaw:.0f} deg from the graph +x axis (0 = +x, 90 = +y)")


def make_monitor(graph, waypoints, max_deviation: float) -> FlightMonitor:
    lo = [min(r.bounds.min[i] for r in graph.rooms.values()) for i in range(3)]
    hi = [max(r.bounds.max[i] for r in graph.rooms.values()) for i in range(3)]
    return FlightMonitor(bounds=Box(tuple(lo), tuple(hi)), path=waypoints, max_deviation_m=max_deviation)


def fly(backend, plan, graph, velocity, pause, max_deviation, check_step=CHECK_STEP_M):
    """Fly the plan. Returns why it was aborted, or None.

    `check_step`: longest move between two monitor checks in m; None checks at waypoints only.
    """
    monitor = make_monitor(graph, plan.waypoints, max_deviation)
    try:
        follow_path(backend, plan.waypoints, velocity=velocity, segment_pause=pause, monitor=monitor,
                    check_step_m=check_step)
    except FlightAbort as e:
        return str(e)
    except KeyboardInterrupt:
        return "interrupted by the user"
    return None


def report(label, backend, plan, aborted):
    print(f"\n--- {label} ---")
    if aborted:
        print(f"ABORTED by the flight monitor: {aborted} (the drone was landed)")
    if backend.trajectory:
        print(compare_to_plan(plan.waypoints, backend.trajectory).format())
    else:
        print("no trajectory recorded")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default="go from A to B")
    ap.add_argument("--backend", choices=["sim", "crazyflie"], default="sim")
    ap.add_argument("--graph", help="saved graph folder (graph.json + voxels.npz), instead of --rooms")
    ap.add_argument("--rooms", type=float, nargs="+", help="virtual room widths in m (x), default 2 2")
    ap.add_argument("--depth", type=float, default=2.0, help="virtual room depth in m (y)")
    ap.add_argument("--door-width", type=float, default=1.0, help="virtual door width in m")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"),
                    help="where the drone stands in the start room, graph coordinates "
                         "(default: the planner's start point, the room center if it is free)")
    ap.add_argument("--current-room", help="start room when the sentence names only the goal")
    ap.add_argument("--policy", choices=[p.value for p in UnknownPolicy], default=UnknownPolicy.OBSTACLE.value,
                    help="unobserved voxels: obstacle (default, safest), free_in_rooms or free")
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--velocity", type=float, default=0.2, help="m/s, no maximum (above 0.3 is untested)")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--max-deviation", type=float, default=0.5, help="abort if farther than this from the plan")
    ap.add_argument("--check-step", type=float, default=None,
                    help="real drone: stop and check every this many m along a segment "
                         "(use 0.1 with the Multi-ranger deck; default: check at waypoints only)")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="real drone: nose direction in the graph frame, degrees from +x toward +y")
    ap.add_argument("--uri", default="radio://0/80/2M")
    ap.add_argument("--skip-dry-run", action="store_true", help="real drone: do not simulate first")
    a = ap.parse_args(argv)

    if a.graph:
        if a.rooms:
            ap.error("--graph and --rooms cannot be used together")
        try:
            graph = load_graph(a.graph)
        except GraphFormatError as e:
            print(f"cannot load the graph: {e}")
            return 1
        space = f"graph {a.graph}: {len(graph.rooms)} rooms, {len(graph.doors)} doors"
    else:
        rooms = a.rooms or [2.0, 2.0]
        graph = build_row_graph(rooms, a.depth, a.door_width)
        space = f"virtual space: {len(rooms)} rooms {rooms} x {a.depth} m, doors {a.door_width} m wide"
    v = validate_graph(graph, a.height, a.margin, a.policy)
    if not v.ok:
        print(v.format())
        return 1
    try:
        goal = parse_goal(a.text, graph, current_room=a.current_room, llm=llm_from_env())
        print(f"goal: {goal.start} -> {goal.goal} (parsed by {goal.source})")
        plan = plan_route(graph, goal.start, goal.goal, a.height, a.margin, a.policy)
        if a.start:
            plan = plan_from(graph, plan, a.start, a.height, a.margin, a.policy)
    except (GoalError, NoPathError, NoRoomRouteError, KeyError) as e:
        print(f"cannot plan: {e}")
        return 1
    check = check_path(graph, plan.waypoints, a.margin, a.policy)
    print(space)
    if a.policy != UnknownPolicy.OBSTACLE.value:
        print(f"unobserved voxels: {a.policy} (the map can hide things that were never observed)")
    print(f"route: {' -> '.join(plan.rooms)}, {len(plan.waypoints)} waypoints, collision check: {check.summary()}")
    for p in plan.waypoints:
        print("   ", tuple(round(c, 2) for c in p))
    if not check.ok:
        print("the planned route violates the safety margin: not flying")
        return 1

    start = plan.waypoints[0]
    floor = graph.floor_z(plan.rooms[0])
    placement = describe_start(graph, plan.rooms[0], start, a.yaw)
    print(f"start: {placement}")

    if a.backend == "crazyflie":
        problem = real_flight_problem(graph, plan)
        if problem:
            print(f"not flying the real drone: {problem}")
            return 1

    if a.backend == "sim" or not a.skip_dry_run:
        from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend

        label = "simulation" + (" (dry run)" if a.backend != "sim" else "")
        sim = MuJoCoBackend(graph, (start[0], start[1], floor))
        aborted = fly(sim, plan, graph, a.velocity, 0.0, a.max_deviation)
        report(label, sim, plan, aborted)
        print(f"wall contacts in the simulation: {sim.collisions}")
        if a.backend == "sim":
            return 0 if (not aborted and sim.collisions == 0) else 1
        if aborted or sim.collisions:
            print("the dry run failed: not flying the real drone")
            return 1

    from ddsg_nav.flight.backends.crazyflie import CrazyflieBackend

    print(f"\nREAL DRONE. Put it {placement}, on a textured floor, in a bright room.")
    print("Starting in 8 seconds. Ctrl+C now to cancel, Ctrl+C later to land.")
    time.sleep(8)
    drone = CrazyflieBackend(a.uri, frame=Transform((start[0], start[1], floor), a.yaw))
    try:
        # The simulation checks every 0.1 m. On the real drone that means a stop and a status
        # read every 0.1 m, which is untested: off unless asked for (needed with the Multi-ranger).
        aborted = fly(drone, plan, graph, a.velocity, 0.5, a.max_deviation, check_step=a.check_step or None)
    finally:
        drone.close()
    report("real flight", drone, plan, aborted)
    return 0 if not aborted else 1


if __name__ == "__main__":
    sys.exit(main())
