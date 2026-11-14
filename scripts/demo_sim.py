"""Plan a route on the fake graph and fly it in the MuJoCo simulation (headless).

Run from the repository root:
    cf-env\\Scripts\\python scripts\\demo_sim.py "go from A to C"
    cf-env\\Scripts\\python scripts\\demo_sim.py "A방에서 C방으로" --obstacle --height 0.8
"""
import argparse

import numpy as np

from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_fake_graph
from ddsg_nav.planner.route import plan_route


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="go from A to C")
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--velocity", type=float, default=0.3)
    ap.add_argument("--obstacle", action="store_true", help="add a box in room B")
    a = ap.parse_args()

    graph = build_fake_graph(obstacle=a.obstacle)
    goal = parse_goal(a.text, graph)
    plan = plan_route(graph, goal.start, goal.goal, height=a.height)
    print(f"goal      : {goal.start} -> {goal.goal}")
    print(f"rooms     : {' -> '.join(plan.rooms)}  (doors: {', '.join(plan.doors)})")
    print(f"waypoints : {len(plan.waypoints)}")
    for p in plan.waypoints:
        print("   ", tuple(round(c, 2) for c in p))

    start = graph.rooms[plan.rooms[0]].center
    with MuJoCoBackend(graph, (start[0], start[1], 0.0)) as drone:
        follow_path(drone, plan.waypoints, velocity=a.velocity)
        end = np.array(drone.position())
        target = np.array(plan.waypoints[-1])
        horiz_err = float(np.linalg.norm(end[:2] - target[:2]))
        path_len = float(
            np.sum(np.linalg.norm(np.diff(np.array(drone.trajectory)[:, 1:], axis=0), axis=1))
        )
        print(f"sim time  : {drone.time:.1f} s, flown distance {path_len:.1f} m")
        print(f"landed at : {tuple(round(float(c), 2) for c in end)}  (goal xy error {horiz_err:.2f} m)")
        print(f"collisions: {drone.collisions}")


if __name__ == "__main__":
    main()
