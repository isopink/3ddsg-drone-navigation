"""Step 3: plan a route through virtual rooms and fly it, in simulation or on the real drone.

The virtual rooms and doors only exist in the map. Size them to the space you can really
fly in, put the drone at the center of the start room with its nose along +x (or pass
--yaw), and the planner keeps it inside the rooms and routes it through the doors.

Run from the repository root:
    # simulation, always do this first
    cf-env\\Scripts\\python scripts\\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
    # real drone: runs the simulation as a dry run first, then asks you to confirm
    cf-env\\Scripts\\python scripts\\fly_route.py --backend crazyflie --rooms 2 2 --depth 2 "A -> B"

    # watch it live in Rerun (the simulation then runs at wall-clock speed)
    cf-env\\Scripts\\python scripts\\fly_route.py --backend sim --live "go from A to C"

Every run is saved to runs/<time>/ with a flight report (report.html): plan vs simulation
vs real flight. Rebuild it with scripts/flight_report.py, replay it with scripts/replay_run.py.

Fly in a bright room over a textured floor (Step 0 lesson), 30 cm to 1 m high.
Ctrl+C lands the drone.
"""
import argparse
import contextlib
import sys
import time
from pathlib import Path

from ddsg_nav.flight.compare import compare_to_plan
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.flight.record import FlightRecorder, FlightRun, new_run_folder, plan_dict, save_run
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


def fly(backend, plan, graph, velocity, pause, max_deviation, label, kind, live=None):
    """Fly the plan. Returns (why it was aborted or None, the FlightRecorder)."""
    monitor = make_monitor(graph, plan.waypoints, max_deviation)
    listeners = [live.listener(label, battery=kind != "sim")] if live is not None else []
    recorder = FlightRecorder(label, kind, clock=backend.clock, listeners=listeners)
    aborted = None
    follow = live.follow(label, backend) if live is not None else contextlib.nullcontext()
    try:
        with follow:
            follow_path(backend, plan.waypoints, velocity=velocity, segment_pause=pause,
                        monitor=monitor, recorder=recorder)
    except FlightAbort as e:
        aborted = str(e)
    except KeyboardInterrupt:
        aborted = "interrupted by the user"
    return aborted, recorder


def save(folder, graph, plan, flights, args):
    meta = {
        "text": args.text, "backend": args.backend, "rooms": args.rooms, "depth": args.depth,
        "door_width": args.door_width, "height": args.height, "velocity": args.velocity,
        "margin": args.margin, "max_deviation": args.max_deviation, "yaw": args.yaw,
        "goal_source": getattr(args, "goal_source", None),
    }
    save_run(FlightRun(graph=graph, plan=plan_dict(plan), flights=flights, meta=meta), folder)
    print(f"\nrun saved to {folder}")
    try:
        from ddsg_nav.report import write_report

        print(f"flight report: {write_report(folder)}")
    except Exception as e:  # the flight data is saved; a report problem must not hide that
        print(f"could not write the flight report: {e}")


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
    ap.add_argument("--live", action="store_true", help="show the flight live in Rerun (pip install rerun-sdk)")
    ap.add_argument("--runs", default=str(RUNS), help="folder for run records (default: runs)")
    a = ap.parse_args()

    graph = build_row_graph(a.rooms, a.depth, a.door_width)
    v = validate_graph(graph, a.height, a.margin)
    if not v.ok:
        print(v.format())
        return 1
    try:
        goal = parse_goal(a.text, graph, llm=llm_from_env())
        print(f"goal: {goal.start} -> {goal.goal} (parsed by {goal.source})")
        a.goal_source = goal.source
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
    folder = new_run_folder(a.runs)
    flights = []
    live = None
    if a.live:
        from ddsg_nav.viz.live import LiveView

        live = LiveView(save=folder / "flight.rrd")
        live.log_scene(graph, plan)

    if a.backend == "sim" or not a.skip_dry_run:
        from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend

        label = "simulation" + (" (dry run)" if a.backend != "sim" else "")
        # with a real drone the dry run stays fast: the real flight is the one to watch
        sim = MuJoCoBackend(graph, (start[0], start[1], 0.0), realtime=a.live and a.backend == "sim")
        aborted, rec = fly(sim, plan, graph, a.velocity, 0.0, a.max_deviation, label, "sim", live)
        report(label, sim, plan, aborted)
        print(f"wall contacts in the simulation: {sim.collisions}")
        flights.append(rec.finish(sim.trajectory, plan.waypoints, aborted, sim.collisions))
        if a.backend == "sim":
            save(folder, graph, plan, flights, a)
            return 0 if (not aborted and sim.collisions == 0) else 1
        if aborted or sim.collisions:
            print("the dry run failed: not flying the real drone")
            save(folder, graph, plan, flights, a)
            return 1

    from ddsg_nav.flight.backends.crazyflie import CrazyflieBackend

    print(f"\nREAL DRONE. Put it at x={start[0]:.2f} y={start[1]:.2f} (center of room {plan.rooms[0]}), "
          f"nose pointing {a.yaw:.0f} deg from +x, on a textured floor, in a bright room.")
    print("Starting in 8 seconds. Ctrl+C now to cancel, Ctrl+C later to land.")
    time.sleep(8)
    drone = CrazyflieBackend(a.uri, frame=Transform((start[0], start[1], 0.0), a.yaw))
    try:
        aborted, rec = fly(drone, plan, graph, a.velocity, 0.5, a.max_deviation, "real flight", "crazyflie", live)
    finally:
        drone.close()
    report("real flight", drone, plan, aborted)
    flights.append(rec.finish(drone.trajectory, plan.waypoints, aborted))
    save(folder, graph, plan, flights, a)
    return 0 if not aborted else 1


if __name__ == "__main__":
    sys.exit(main())
