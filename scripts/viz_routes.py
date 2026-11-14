"""Step 2: check routes for collisions and render them with Open3D.

Run from the repository root:
    cf-env\\Scripts\\python scripts\\viz_routes.py              # collision table for all room pairs
    cf-env\\Scripts\\python scripts\\viz_routes.py --screenshots # also save PNGs to docs/media
    cf-env\\Scripts\\python scripts\\viz_routes.py --sim         # overlay a MuJoCo flight (blue)
    cf-env\\Scripts\\python scripts\\viz_routes.py --show "go from A to C" --obstacle
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.planner.collision import check_path, check_trajectory
from ddsg_nav.planner.route import plan_route

MEDIA = Path("docs/media")
HEIGHTS = (0.3, 0.5, 1.0)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshots", action="store_true", help="save PNGs to docs/media")
    ap.add_argument("--sim", action="store_true", help="fly the route in MuJoCo and draw it")
    ap.add_argument("--show", metavar="TEXT", help="open an interactive window for this goal text")
    ap.add_argument("--obstacle", action="store_true", help="with --show: add the box in room B")
    ap.add_argument("--height", type=float, default=0.5)
    a = ap.parse_args()

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
        cases = [("A", "C", False), ("A", "C", True)]
        for s, g, obstacle in cases:
            graph = build_fake_graph(obstacle=obstacle)
            plan = plan_route(graph, s, g, height=a.height)
            traj = fly_in_sim(graph, plan) if a.sim else None
            center = o3d_viz.scene_center(graph)
            tag = f"route_{s}_to_{g}{'_obstacle' if obstacle else ''}"
            for view in ("top", "oblique"):
                geoms = o3d_viz.build_geometries(
                    graph, plan.waypoints, traj, cutaway=(view == "oblique")
                )
                path = MEDIA / f"{tag}_{view}.png"
                saved = o3d_viz.save_screenshot(geoms, str(path), center, view)
                print(("saved " if saved else "FAILED to save ") + str(path))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
