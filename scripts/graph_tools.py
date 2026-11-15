"""Tools for working with scene graph exports (see docs/graph_format.md).

Run from the repository root:
    cf-env\\Scripts\\python scripts\\graph_tools.py export-fake data\\fake [--obstacle | --large]
    cf-env\\Scripts\\python scripts\\graph_tools.py validate data\\fake [--height 0.5] [--policy free_in_rooms]
    cf-env\\Scripts\\python scripts\\graph_tools.py route data\\fake "401호에서 302호로 가줘"
"""
import argparse
import sys

from ddsg_nav.goal import GoalError, parse_goal
from ddsg_nav.llm_goal import llm_from_env
from ddsg_nav.graph.fake import build_fake_graph, build_large_graph
from ddsg_nav.graph.io import GraphFormatError, load_graph, save_graph
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.global_planner import NoRoomRouteError
from ddsg_nav.planner.local_planner import NoPathError
from ddsg_nav.planner.policy import UnknownPolicy
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import validate_graph

POLICIES = [p.value for p in UnknownPolicy]


def cmd_export_fake(a) -> int:
    graph = build_large_graph() if a.large else build_fake_graph(obstacle=a.obstacle)
    save_graph(graph, a.folder)
    print(f"saved {len(graph.rooms)} rooms, {len(graph.doors)} doors, voxels {graph.voxels.shape} to {a.folder}")
    return 0


def cmd_validate(a) -> int:
    graph = load_graph(a.folder)
    report = validate_graph(
        graph, a.height, a.margin, a.policy, check_connectivity=not a.skip_connectivity
    )
    print(report.format())
    return 0 if report.ok else 1


def cmd_route(a) -> int:
    graph = load_graph(a.folder)
    try:
        goal = parse_goal(a.text, graph, current_room=a.current_room, llm=llm_from_env())
        print(f"goal: {goal.start} -> {goal.goal} (parsed by {goal.source})")
        plan = plan_route(graph, goal.start, goal.goal, a.height, a.margin, a.policy)
    except (GoalError, NoRoomRouteError, NoPathError, KeyError) as e:
        print(f"cannot plan: {e}")
        return 1
    report = check_path(graph, plan.waypoints, a.margin, a.policy)
    print(f"{goal.start} -> {goal.goal}: rooms {' -> '.join(plan.rooms)}")
    print(f"{len(plan.waypoints)} waypoints:")
    for p in plan.waypoints:
        print("   ", tuple(round(c, 2) for c in p))
    print("collision check:", report.summary())
    return 0 if report.ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export-fake", help="write the fake graph in the exchange format")
    p.add_argument("folder")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--obstacle", action="store_true")
    g.add_argument("--large", action="store_true", help="30 x 20 m, 24 rooms")
    p.set_defaults(fn=cmd_export_fake)

    for name, fn, help_ in (("validate", cmd_validate, "check a graph export"),
                            ("route", cmd_route, "plan a route from a sentence")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("folder")
        if name == "route":
            p.add_argument("text")
            p.add_argument("--current-room", default=None)
        p.add_argument("--height", type=float, default=0.5)
        p.add_argument("--margin", type=float, default=0.25)
        p.add_argument("--policy", choices=POLICIES, default="obstacle",
                       help="how unobserved voxels are treated")
        if name == "validate":
            p.add_argument("--skip-connectivity", action="store_true")
        p.set_defaults(fn=fn)

    a = ap.parse_args()
    try:
        return a.fn(a)
    except GraphFormatError as e:
        print(f"invalid graph export: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
