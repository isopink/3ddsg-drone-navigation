"""Check routes for collisions and draw them.

Run from the repository root.

Fake graph (as before):
    cf-env\\Scripts\\python scripts\\check_routes.py              # collision table for all room pairs
    cf-env\\Scripts\\python scripts\\check_routes.py --screenshots # also save PNGs to docs/media
    cf-env\\Scripts\\python scripts\\check_routes.py --sim         # overlay a MuJoCo flight (blue)
    cf-env\\Scripts\\python scripts\\check_routes.py --show "go from A to C" --obstacle
    cf-env\\Scripts\\python scripts\\check_routes.py --screenshots --sim --corridor  # room, L-shaped hallway, room

Route preview on a graph folder (the format of docs/graph_format.md):
    cf-env\\Scripts\\python scripts\\check_routes.py --graph data\\floor4 "from A to C"        # interactive window
    cf-env\\Scripts\\python scripts\\check_routes.py --graph data\\floor4 "from A to C" --png  # floor plan, no display needed
    cf-env\\Scripts\\python scripts\\check_routes.py --graph data\\floor4 "from A to C" --png out.png --view oblique
    cf-env\\Scripts\\python scripts\\check_routes.py --graph data\\floor4 --all-pairs          # collision table of that graph

`--png` without a file name writes to data\\previews\\, which git ignores. Pictures of a
real building must not be committed: with `--graph`, a path inside the repository but
outside data\\ is refused.
"""
import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.graph.fake import build_corridor_graph, build_fake_graph
from ddsg_nav.graph.io import GraphFormatError, load_graph
from ddsg_nav.planner.collision import check_path, check_trajectory
from ddsg_nav.planner.global_planner import NoRoomRouteError
from ddsg_nav.planner.local_planner import NoPathError
from ddsg_nav.planner.policy import UnknownPolicy
from ddsg_nav.planner.route import plan_route

REPO = Path(__file__).resolve().parents[1]
MEDIA = Path("docs/media")
PREVIEWS = REPO / "data" / "previews"
HEIGHTS = (0.3, 0.5, 1.0)
VIEWS = ("plan", "top", "oblique")


def collision_table() -> bool:
    """Plan every ordered room pair at several heights, with and without an obstacle."""
    all_ok = True
    print(f"{'obstacle':<9}{'height':<8}{'pairs':<7}{'waypoints':<11}{'hard':<6}{'margin':<8}{'outside'}")
    for obstacle in (False, True):
        graph = build_fake_graph(obstacle=obstacle)
        for h in HEIGHTS:
            n_pairs = n_wp = hard = margin = outside = 0
            for s, g in itertools.permutations(graph.rooms, 2):
                plan = plan_route(graph, s, g, height=h)
                r = check_path(graph, plan.waypoints)
                n_pairs += 1
                n_wp += len(plan.waypoints)
                hard += len(r.hard)
                margin += len(r.margin)
                outside += len(r.outside)
            all_ok &= (hard + margin + outside) == 0
            print(f"{str(obstacle):<9}{h:<8}{n_pairs:<7}{n_wp:<11}{hard:<6}{margin:<8}{outside}")
    print("RESULT:", "zero collisions in all cases" if all_ok else "COLLISIONS FOUND")
    return all_ok


def pair_table(graph, height: float, margin: float, policy: str) -> bool:
    """Plan and check every reachable room pair of `graph` (each pair once, in the
    graph's room order). Pairs that no chain of doors connects are counted, not planned."""
    rooms = nx.Graph(graph.room_graph())
    rooms.add_nodes_from(graph.rooms)
    width = max([len(r) for r in graph.rooms] + [5]) + 2
    print(f"{'start':<{width}}{'goal':<{width}}{'waypoints':<11}{'length_m':<10}{'hard':<6}{'margin':<8}{'outside'}")
    planned = bad = unreachable = 0
    for s, g in itertools.combinations(graph.rooms, 2):
        if not nx.has_path(rooms, s, g):
            unreachable += 1
            continue
        planned += 1
        try:
            plan = plan_route(graph, s, g, height, margin, policy)
        except (NoRoomRouteError, NoPathError) as e:
            bad += 1
            print(f"{s:<{width}}{g:<{width}}NO PATH: {e}")
            continue
        r = check_path(graph, plan.waypoints, margin, policy)
        bad += not r.ok
        length = sum(math.dist(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:]))
        print(
            f"{s:<{width}}{g:<{width}}{len(plan.waypoints):<11}{length:<10.2f}"
            f"{len(r.hard):<6}{len(r.margin):<8}{len(r.outside)}"
        )
    print(f"{planned} reachable pairs, {unreachable} unreachable, {bad} with a collision or no path")
    print("RESULT:", "zero collisions in all cases" if not bad else "COLLISIONS FOUND")
    return not bad


def fly_in_sim(graph, plan) -> np.ndarray:
    from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend
    from ddsg_nav.flight.executor import follow_path

    start = plan.waypoints[0]
    with MuJoCoBackend(graph, (start[0], start[1], 0.0)) as drone:
        follow_path(drone, plan.waypoints, velocity=0.3)
        traj = np.array(drone.trajectory)[:, 1:]
        report = check_trajectory(graph, traj)
        print(f"  flown trajectory: {report.summary()}, wall contacts {drone.collisions}")
    return traj[:: max(1, len(traj) // 250)]  # thin out for drawing


def png_path(arg: str, name: str, real_graph: bool) -> Path:
    """Where `--png` writes. No file name: data/previews/<name>.png. A picture of a
    real graph may not go into the repository outside data/."""
    if not arg:
        return PREVIEWS / f"{name}.png"
    path = Path(arg).resolve()
    if real_graph and path.is_relative_to(REPO) and not path.is_relative_to(REPO / "data"):
        raise ValueError(f"{path} is inside the repository; write under data\\ or outside the repository")
    return path


def preview(a) -> int:
    """Plan `a.text` on the chosen graph and show it: a window, or a PNG with --png."""
    from ddsg_nav.viz import o3d_viz

    try:
        graph = load_graph(a.graph) if a.graph else build_fake_graph(obstacle=a.obstacle)
    except GraphFormatError as e:
        print(f"cannot load graph: {e}")
        return 1
    ok = True
    if a.all_pairs:
        ok = pair_table(graph, a.height, a.margin, a.policy)
    if not a.text:
        return 0 if ok else 1

    try:
        goal = parse_goal(a.text, graph)
        plan = plan_route(graph, goal.start, goal.goal, a.height, a.margin, a.policy)
    except (GoalError, NoRoomRouteError, NoPathError, KeyError) as e:
        print(f"cannot plan: {e}")
        return 1
    report = check_path(graph, plan.waypoints, a.margin, a.policy)
    print(f"{goal.start} -> {goal.goal}: rooms {' -> '.join(plan.rooms)}, {len(plan.waypoints)} waypoints")
    print("collision check:", report.summary())
    ok &= report.ok
    traj = fly_in_sim(graph, plan) if a.sim else None

    if a.png is None:
        try:
            o3d_viz.show(o3d_viz.build_geometries(graph, plan.waypoints, traj, labels=True))
        except ImportError as e:
            print(f"no interactive window ({e}); use --png for a floor plan")
            return 1
        return 0 if ok else 1

    name = "_".join([Path(a.graph).resolve().name if a.graph else "fake", goal.start, "to", goal.goal])
    try:
        path = png_path(a.png, re.sub(r"[^\w.-]", "_", name), bool(a.graph))
    except ValueError as e:
        print(f"refused: {e}")
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    if a.view == "plan":
        o3d_viz.save_floor_plan(graph, str(path), plan.waypoints, traj, height=a.height)
        saved = True
    else:
        geoms = o3d_viz.build_geometries(
            graph, plan.waypoints, traj, cutaway=(a.view == "oblique"), labels=True
        )
        saved = o3d_viz.save_screenshot(geoms, str(path), o3d_viz.scene_center(graph), a.view, zoom=a.zoom)
    print(("saved " if saved else "FAILED to save ") + str(path))
    return 0 if ok and saved else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help='preview this route, e.g. "from A to C"')
    ap.add_argument("--graph", metavar="FOLDER", help="graph folder to use instead of the fake graph")
    ap.add_argument("--png", nargs="?", const="", metavar="FILE",
                    help="with a route text: write a PNG instead of opening a window (default: data/previews/)")
    ap.add_argument("--view", choices=VIEWS, default="plan",
                    help="with --png: plan = floor plan, no display needed (default); top, oblique = Open3D")
    ap.add_argument("--zoom", type=float, default=None, help="with --view top or oblique: Open3D zoom (larger shows more)")
    ap.add_argument("--all-pairs", action="store_true", help="collision table for every reachable room pair of the graph")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--policy", choices=[p.value for p in UnknownPolicy], default="obstacle",
                    help="how unobserved voxels are treated")
    ap.add_argument("--screenshots", action="store_true", help="save PNGs to docs/media")
    ap.add_argument("--sim", action="store_true", help="fly the route in MuJoCo and draw it")
    ap.add_argument("--show", metavar="TEXT", help="open an interactive window for this goal text")
    ap.add_argument("--obstacle", action="store_true", help="fake graph: add the box in room B")
    ap.add_argument("--corridor", action="store_true", help="with --screenshots: room A, L-shaped hallway, room C")
    ap.add_argument("--height", type=float, default=0.5)
    a = ap.parse_args(argv)

    if a.text or a.graph or a.all_pairs:
        return preview(a)

    ok = collision_table()

    if a.show or a.screenshots:
        from ddsg_nav.viz import o3d_viz

    if a.show:
        graph = build_fake_graph(obstacle=a.obstacle)
        goal = parse_goal(a.show, graph)
        plan = plan_route(graph, goal.start, goal.goal, height=a.height)
        traj = fly_in_sim(graph, plan) if a.sim else None
        o3d_viz.show(o3d_viz.build_geometries(graph, plan.waypoints, traj))

    if a.screenshots:
        MEDIA.mkdir(parents=True, exist_ok=True)
        if a.corridor:
            cases = [("A", "C", False, build_corridor_graph, "corridor")]
        else:
            cases = [("A", "C", o, lambda o=o: build_fake_graph(obstacle=o), "route") for o in (False, True)]
        for s, g, obstacle, build, prefix in cases:
            graph = build()
            plan = plan_route(graph, s, g, height=a.height)
            traj = fly_in_sim(graph, plan) if a.sim else None
            center = o3d_viz.scene_center(graph)
            tag = f"{prefix}_{s}_to_{g}{'_obstacle' if obstacle else ''}"
            for view in ("top", "oblique"):
                geoms = o3d_viz.build_geometries(
                    graph, plan.waypoints, traj, cutaway=(view == "oblique")
                )
                path = MEDIA / f"{tag}_{view}.png"
                zoom = {"top": 0.66, "oblique": 0.78}[view] if a.corridor else None  # taller layout
                saved = o3d_viz.save_screenshot(geoms, str(path), center, view, zoom=zoom)
                print(("saved " if saved else "FAILED to save ") + str(path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
