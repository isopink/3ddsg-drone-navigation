"""How fast is planning on a floor-sized graph? (30 m x 20 m, 24 rooms by default)

Run from the repository root:
    cf-env\\Scripts\\python scripts\\benchmark_planning.py
    cf-env\\Scripts\\python scripts\\benchmark_planning.py --resolutions 0.1 0.2 0.25
"""
import argparse
import time

from ddsg_nav.graph.fake import build_large_graph
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.local_planner import plan_voxel_path
from ddsg_nav.planner.policy import blocked_mask
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--rooms-x", type=int, default=6)
    ap.add_argument("--rooms-y", type=int, default=4)
    a = ap.parse_args()

    print(f"{'res m':<7}{'voxels':<11}{'build':<8}{'inflate':<9}{'validate':<10}"
          f"{'route':<8}{'check':<8}{'full-A*':<9}{'rooms':<7}{'waypoints':<10}{'collisions'}")
    for res in a.resolutions:
        graph, t_build = timed(lambda: build_large_graph(a.rooms_x, a.rooms_y, resolution=res))
        _, t_inflate = timed(lambda: blocked_mask(graph, 0.25))
        _, t_val = timed(lambda: validate_graph(graph))
        start, goal = "R0_0", f"R{a.rooms_x - 1}_{a.rooms_y - 1}"
        plan, t_route = timed(lambda: plan_route(graph, start, goal, height=0.5))
        report, t_check = timed(lambda: check_path(graph, plan.waypoints))
        # worst case: one A* over the whole floor, ignoring the room structure
        blocked = blocked_mask(graph, 0.25)
        p0, p1 = plan.keypoints[0], plan.keypoints[-1]
        _, t_full = timed(lambda: plan_voxel_path(graph.voxels, p0, p1, blocked=blocked))
        print(f"{res:<7}{graph.voxels.states.size:<11}{t_build:<8.2f}{t_inflate:<9.2f}{t_val:<10.2f}"
              f"{t_route:<8.2f}{t_check:<8.2f}{t_full:<9.2f}{len(plan.rooms):<7}{len(plan.waypoints):<10}"
              f"{len(report.hard) + len(report.margin) + len(report.outside)}")
    print("times in seconds. route = plan_route corner to corner (room by room);")
    print("full-A* = a single A* over the whole floor between the same two points (worst case)")


if __name__ == "__main__":
    main()
