"""Step 3: plan a route through virtual rooms and fly it, in simulation or on the real drone.

The virtual rooms and doors only exist in the map. Size them to the space you can really
fly in, put the drone at the center of the start room with its nose along +x (or pass
--yaw), and the planner keeps it inside the rooms and routes it through the doors.

Run from the repository root:
    # simulation, always do this first
    cf-env\\Scripts\\python scripts\\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
    # real drone: runs the simulation as a dry run first, then asks you to confirm
    cf-env\\Scripts\\python scripts\\fly_route.py --backend crazyflie --rooms 2 2 --depth 2 "A -> B"

Fly in a bright room over a textured floor (Step 0 lesson), 30 cm to 1 m high.
Ctrl+C lands the drone.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from ddsg_nav.flight.compare import compare_to_plan, save_trajectory
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.llm_goal import llm_from_env
from ddsg_nav.graph.fake import build_row_graph
from ddsg_nav.graph.model import Box
from ddsg_nav.graph.transform import Transform
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.local_planner import NoPathError
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

RUNS = Path("runs")


def make_monitor(graph, waypoints, max_deviation: float) -> FlightMonitor:
    lo = [min(r.bounds.min[i] for r in graph.rooms.values()) for i in range(3)]
    hi = [max(r.bounds.max[i] for r in graph.rooms.values()) for i in range(3)]
    return FlightMonitor(bounds=Box(tuple(lo), tuple(hi)), path=waypoints, max_deviation_m=max_deviation)


def fly(backend, plan, graph, velocity, pause, max_deviation):
    monitor = make_monitor(graph, plan.waypoints, max_deviation)
    aborted = None
    try:
        follow_path(backend, plan.waypoints, velocity=velocity, segment_pause=pause, monitor=monitor)
    except FlightAbort as e:
        aborted = str(e)
    return aborted


def report(label, backend, plan, aborted):
    print(f"\n--- {label} ---")
    if aborted:
        print(f"ABORTED by the flight monitor: {aborted} (the drone was landed)")
    if backend.trajectory:
        print(compare_to_plan(plan.waypoints, backend.trajectory).format())
    else:
        print("no trajectory recorded")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", default="go from A to B")
    ap.add_argument("--backend", choices=["sim", "crazyflie"], default="sim")
    ap.add_argument("--rooms", type=float, nargs="+", default=[2.0, 2.0], help="room widths in m (x)")
    ap.add_argument("--depth", type=float, default=2.0, help="room depth in m (y)")
    ap.add_argument("--door-width", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--velocity", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--max-deviation", type=float, default=0.5, help="abort if farther than this from the plan")
    ap.add_argument("--yaw", type=float, default=0.0, help="real drone: nose direction in degrees from +x")
    ap.add_argument("--uri", default="radio://0/80/2M")
    ap.add_argument("--skip-dry-run", action="store_true", help="real drone: do not simulate first")
    a = ap.parse_args()

    graph = build_row_graph(a.rooms, a.depth, a.door_width)
    v = validate_graph(graph, a.height, a.margin)
    if not v.ok:
        print(v.format())
        return 1
    try:
        goal = parse_goal(a.text, graph, llm=llm_from_env())
        print(f"goal: {goal.start} -> {goal.goal} (parsed by {goal.source})")
        plan = plan_route(graph, goal.start, goal.goal, a.height, a.margin)
    except (GoalError, NoPathError, KeyError) as e:
        print(f"cannot plan: {e}")
        return 1
    check = check_path(graph, plan.waypoints, a.margin)
    print(f"virtual space: {len(a.rooms)} rooms {a.rooms} x {a.depth} m, doors {a.door_width} m wide")
    print(f"route: {' -> '.join(plan.rooms)}, {len(plan.waypoints)} waypoints, collision check: {check.summary()}")
    for p in plan.waypoints:
        print("   ", tuple(round(c, 2) for c in p))
    if not check.ok:
        print("the planned route violates the safety margin: not flying")
        return 1

    start = plan.waypoints[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if a.backend == "sim" or not a.skip_dry_run:
        from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend

        sim = MuJoCoBackend(graph, (start[0], start[1], 0.0))
        aborted = fly(sim, plan, graph, a.velocity, 0.0, a.max_deviation)
        report("simulation" + (" (dry run)" if a.backend != "sim" else ""), sim, plan, aborted)
        print(f"wall contacts in the simulation: {sim.collisions}")
        if a.backend == "sim":
            path = save_trajectory(RUNS / f"{stamp}_sim.csv", sim.trajectory)
            print(f"trajectory saved to {path}")
            return 0 if (not aborted and sim.collisions == 0) else 1
        if aborted or sim.collisions:
            print("the dry run failed: not flying the real drone")
            return 1

    from ddsg_nav.flight.backends.crazyflie import CrazyflieBackend

    print(f"\nREAL DRONE. Put it at x={start[0]:.2f} y={start[1]:.2f} (center of room {plan.rooms[0]}), "
          f"nose pointing {a.yaw:.0f} deg from +x, on a textured floor, in a bright room.")
    print("Starting in 8 seconds. Ctrl+C now to cancel, Ctrl+C later to land.")
    time.sleep(8)
    drone = CrazyflieBackend(a.uri, frame=Transform((start[0], start[1], 0.0), a.yaw))
    try:
        aborted = fly(drone, plan, graph, a.velocity, 0.5, a.max_deviation)
    except KeyboardInterrupt:
        aborted = "interrupted by the user"
    finally:
        drone.close()
    report("real flight", drone, plan, aborted)
    if drone.trajectory:
        print(f"trajectory saved to {save_trajectory(RUNS / f'{stamp}_crazyflie.csv', drone.trajectory)}")
    return 0 if not aborted else 1


if __name__ == "__main__":
    sys.exit(main())
