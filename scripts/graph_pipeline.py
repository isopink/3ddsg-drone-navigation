"""One command to check a scene graph end to end: load, validate, plan, simulate.

    python scripts\\graph_pipeline.py data\\4F                       # a folder in the exchange format
    python scripts\\graph_pipeline.py --neo4j bolt://host:7687 --floor 4F
    python scripts\\graph_pipeline.py --fake                         # the built-in 3-room graph
        [--from ROOM --to ROOM]    one route; default: a few of the longest room pairs
        [--routes 3] [--height 0.5] [--margin 0.25] [--velocity 0.3]
        [--policy obstacle|free|free_in_rooms]    how unobserved voxels are treated
        [--save FOLDER]            write the loaded graph in the exchange format (docs\\graph_format.md)

Steps: 1 load the graph, 2 validate it, 3 plan the routes, 4 fly them in MuJoCo. The
report lists rooms, doors, grid size, the share of unknown voxels, rooms that cannot be
reached (with the reason), routes, wall contacts and every failure with its reason.
Exit code 0 only if all steps pass.

Neo4j: the user and password are read from the environment variables DDSG_NEO4J_USER and
DDSG_NEO4J_PASSWORD, never from the command line.

Needs `pip install -e ".[sim]"` (and `.[neo4j]` for --neo4j).
"""
from __future__ import annotations

import argparse
import os
import sys

from ddsg_nav import pipeline
from ddsg_nav.graph.source import FakeGraphSource, FileGraphSource
from ddsg_nav.planner.policy import UnknownPolicy

POLICIES = [p.value for p in UnknownPolicy]
ENV_USER, ENV_PASSWORD = "DDSG_NEO4J_USER", "DDSG_NEO4J_PASSWORD"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="graph folder in the exchange format")
    ap.add_argument("--neo4j", metavar="URI", help="load one floor from Neo4j instead (needs --floor)")
    ap.add_argument("--floor", metavar="NAME", help="floor to load from Neo4j")
    ap.add_argument("--fake", action="store_true", help="use the built-in 3-room graph")
    ap.add_argument("--from", dest="start", metavar="ROOM", help="start room (id, name or alias)")
    ap.add_argument("--to", dest="goal", metavar="ROOM", help="goal room")
    ap.add_argument("--routes", type=int, default=3, help="how many routes to pick without --from/--to")
    ap.add_argument("--height", type=float, default=0.5, help="flight height above the floor in meters")
    ap.add_argument("--margin", type=float, default=0.25, help="distance kept from walls in meters")
    ap.add_argument("--velocity", type=float, default=0.3, help="simulated flight speed in m/s")
    ap.add_argument("--policy", choices=POLICIES, default="obstacle",
                    help="how unobserved voxels are treated (see planner/policy.py)")
    ap.add_argument("--save", metavar="FOLDER", help="write the loaded graph in the exchange format")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # room names may not fit the console code page

    if sum([a.folder is not None, a.neo4j is not None, a.fake]) != 1:
        ap.error("give exactly one graph: a folder, --neo4j URI --floor NAME, or --fake")
    if (a.neo4j is None) != (a.floor is None):
        ap.error("--neo4j and --floor go together")

    if a.neo4j is not None:
        user, password = os.environ.get(ENV_USER), os.environ.get(ENV_PASSWORD)
        if not user or not password:
            print(f"FAILED load: set {ENV_USER} and {ENV_PASSWORD} in the environment "
                  f"(they are not taken from the command line)")
            return 1
        from ddsg_nav.graph.neo4j_source import Neo4jGraphSource
        source = Neo4jGraphSource(uri=a.neo4j, user=user, password=password, floor=a.floor)
        name = f"Neo4j {a.neo4j}, floor {a.floor}"
    elif a.fake:
        source, name = FakeGraphSource(), "built-in fake graph"
    else:
        source, name = FileGraphSource(a.folder), a.folder

    print(f"Graph pipeline: {name}")
    result = pipeline.run_source(source, a.save, a.start, a.goal, a.routes, a.height, a.margin,
                                 a.velocity, a.policy, log=lambda line: print(line, flush=True))
    print(pipeline.format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
