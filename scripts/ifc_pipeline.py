"""One command from an IFC file to simulated flights: convert, validate, plan, simulate.

    python scripts\\ifc_pipeline.py model.ifc                 # list the storeys and stop
    python scripts\\ifc_pipeline.py model.ifc --storey 4F     # run all four steps
        [--out data\\4F]            graph folder (default data\\<storey>, never committed)
        [--from ROOM --to ROOM]    one route; default: a few of the longest room pairs
        [--routes 3] [--exclude "^(PS|EPS|EV)\\d"] [--res 0.1] [--height 0.5] [--margin 0.25] [--velocity 0.3]

Steps: 1 convert the storey (scripts\\ifc_to_graph.py) and save the graph, 2 validate it,
3 plan the routes, 4 fly them in MuJoCo. The report lists rooms, doors, routes, wall
contacts and every failure with its reason. Exit code 0 only if all steps pass.

Needs `pip install -e ".[ifc,sim]"`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from ddsg_nav import pipeline
from ddsg_nav.graph.io import save_graph
from ddsg_nav.pipeline import PipelineResult, RouteResult, fly_route, pick_pairs  # noqa: F401 (shared steps)

from ifc_to_graph import ConversionError, convert, list_storeys

HINTS = pipeline.Hints(
    scope="storey",
    no_pairs="Check the doors and space boundaries of the IFC (see the notes above)",
    doorless="They cannot be reached. If one should be, its door has no space boundary to it in the "
             "IFC: fix the door or the export; shafts can be left out with --exclude",
)


def run(ifc: str, storey: str, out: str | Path | None = None, start: str | None = None,
        goal: str | None = None, routes: int = 3, res: float = 0.1, height: float = 0.5,
        margin: float = 0.25, velocity: float = 0.3, exclude: str | None = None,
        log=lambda line: None) -> PipelineResult:
    """Run the four steps. Stops at the first step that fails; `log` gets progress lines.

    Step 1 is here; steps 2 to 4 are shared with graph_pipeline.py (ddsg_nav/pipeline.py).
    """
    result = PipelineResult()
    try:
        graph, result.notes = convert(ifc, storey, res, exclude=exclude)
    except ConversionError as e:
        result.failures.append(f"convert: {e}")
        return result
    result.graph = graph
    result.out = save_graph(graph, out or Path("data") / re.sub(r"[^\w.-]+", "_", storey))
    n_open = sum(d.startswith("open_") for d in graph.doors)
    log(f"[1/4] convert   {len(graph.rooms)} rooms, {len(graph.doors) - n_open} doors, {n_open} open "
        f"connections, {len(graph.solids)} wall boxes, voxels {graph.voxels.shape} -> {result.out}")

    return pipeline.check_graph(graph, result, start, goal, routes, height, margin, velocity,
                                log=log, hints=HINTS)


def format_report(result: PipelineResult) -> str:
    return pipeline.format_report(result, HINTS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ifc")
    ap.add_argument("--storey", help="IfcBuildingStorey name; without it the storeys are listed")
    ap.add_argument("--out", help="graph folder (default data\\<storey>)")
    ap.add_argument("--from", dest="start", metavar="ROOM", help="start room (id, name or alias)")
    ap.add_argument("--to", dest="goal", metavar="ROOM", help="goal room")
    ap.add_argument("--routes", type=int, default=3, help="how many routes to pick without --from/--to")
    ap.add_argument("--exclude", help=r'regex on space names to leave out, e.g. "^(PS|EPS|EV)\d" for shafts')
    ap.add_argument("--res", type=float, default=0.1, help="voxel size in meters")
    ap.add_argument("--height", type=float, default=0.5, help="flight height above the floor in meters")
    ap.add_argument("--margin", type=float, default=0.25, help="distance kept from walls in meters")
    ap.add_argument("--velocity", type=float, default=0.3, help="simulated flight speed in m/s")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # room names may not fit the console code page

    if not a.storey:
        try:
            storeys = list_storeys(a.ifc)
        except ConversionError as e:
            print(f"FAILED convert: {e}")
            return 1
        if not storeys:
            print(f"FAILED convert: {a.ifc} has no IfcBuildingStorey. Export the model again with its "
                  f"levels (building storeys).")
            return 1
        print(f"storeys in {a.ifc}:")
        for name, elevation, spaces in storeys:
            print(f"  {name:<20} elevation {elevation:8.2f} m   {spaces} spaces")
        print("run again with --storey NAME (a storey that has spaces)")
        return 0

    print(f"IFC pipeline: {a.ifc}, storey {a.storey}")
    result = run(a.ifc, a.storey, a.out, a.start, a.goal, a.routes, a.res, a.height, a.margin,
                 a.velocity, a.exclude, log=lambda line: print(line, flush=True))
    print(format_report(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
