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
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from ddsg_nav.flight.executor import follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.graph.io import save_graph
from ddsg_nav.graph.model import SceneGraph
from ddsg_nav.planner.collision import check_path
from ddsg_nav.planner.global_planner import NoRoomRouteError
from ddsg_nav.planner.local_planner import NoPathError
from ddsg_nav.planner.route import plan_route
from ddsg_nav.validate import ValidationReport, validate_graph

from ifc_to_graph import ConversionError, convert, list_storeys


@dataclass
class RouteResult:
    start: str
    goal: str
    rooms: list[str] = field(default_factory=list)
    length_m: float = 0.0
    flown: bool = False
    wall_contacts: int = 0
    failure: str | None = None  # "step: reason"


@dataclass
class PipelineResult:
    graph: SceneGraph | None = None
    out: Path | None = None
    notes: list[str] = field(default_factory=list)  # from the converter
    validation: ValidationReport | None = None
    doorless: list[str] = field(default_factory=list)  # rooms nothing connects to
    routes: list[RouteResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)  # "step: reason"

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def wall_contacts(self) -> int:
        return sum(r.wall_contacts for r in self.routes)


def pick_pairs(graph: SceneGraph, n: int) -> list[tuple[str, str]]:
    """Up to `n` room pairs to fly: the longest routes first, each room used once."""
    g = graph.room_graph()
    lengths = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
    pairs = sorted((-d, a, b) for a in lengths for b, d in lengths[a].items() if a < b)
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    for _, a, b in pairs:
        if len(chosen) < n and a not in used and b not in used:
            chosen.append((a, b))
            used |= {a, b}
    return chosen


def fly_route(graph: SceneGraph, start: str, goal: str, height: float, margin: float,
              velocity: float) -> RouteResult:
    """Plan one route, check it against the voxels and fly it in the simulator."""
    r = RouteResult(start, goal)
    try:
        plan = plan_route(graph, start, goal, height, margin)
    except KeyError as e:
        r.failure = (f"plan: {e.args[0]}. Use a room id or name of this storey: "
                     f"{', '.join(sorted(graph.rooms))}")
        return r
    except NoRoomRouteError as e:
        r.failure = f"plan: {e}: no chain of doors connects them. Pick rooms that are connected"
        return r
    except NoPathError as e:
        r.failure = f"plan: {e}. Try a lower --margin or another --height"
        return r
    r.start, r.goal, r.rooms = plan.rooms[0], plan.rooms[-1], plan.rooms
    r.length_m = sum(math.dist(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:]))
    check = check_path(graph, plan.waypoints, margin)
    if not check.ok:
        r.failure = f"plan: the route breaks the {margin} m safety margin ({check.summary()})"
        return r
    try:
        from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend, SimTimeoutError
        sim = MuJoCoBackend(graph, (plan.waypoints[0][0], plan.waypoints[0][1], graph.floor_z(r.start)))
    except ImportError:
        r.failure = 'simulate: MuJoCo is not installed. Run: pip install -e ".[sim]"'
        return r
    try:
        follow_path(sim, plan.waypoints, velocity=velocity,
                    monitor=FlightMonitor(path=plan.waypoints, max_deviation_m=0.5))
    except (FlightAbort, SimTimeoutError) as e:
        r.failure = f"simulate: flight aborted: {e}"
    r.flown = r.failure is None
    r.wall_contacts = sim.collisions
    if sim.collisions and r.failure is None:
        r.failure = f"simulate: the drone touched a wall ({sim.collisions} contact steps)"
    return r


def run(ifc: str, storey: str, out: str | Path | None = None, start: str | None = None,
        goal: str | None = None, routes: int = 3, res: float = 0.1, height: float = 0.5,
        margin: float = 0.25, velocity: float = 0.3, exclude: str | None = None,
        log=lambda line: None) -> PipelineResult:
    """Run the four steps. Stops at the first step that fails; `log` gets progress lines."""
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

    report = result.validation = validate_graph(graph, height, margin)
    result.failures += [f"validate: {i.code}: {i.message}" for i in report.errors]
    degree = graph.room_graph().degree
    result.doorless = sorted(r for r in graph.rooms if degree[r] == 0)
    log(f"[2/4] validate  {len(report.errors)} errors, {len(report.warnings)} warnings")
    if not report.ok:
        return result

    if (start is None) != (goal is None):
        result.failures.append("plan: give both --from and --to, or neither")
        return result
    pairs = [(start, goal)] if start is not None else pick_pairs(graph, routes)
    if not pairs:
        result.failures.append("plan: no two rooms are connected, so there is nothing to fly. Check the "
                               "doors and space boundaries of the IFC (see the notes above)")
        return result
    log(f"[3/4] plan      {len(pairs)} routes at {height} m, {margin} m from walls")
    for i, (a, b) in enumerate(pairs, 1):
        r = fly_route(graph, a, b, height, margin, velocity)
        result.routes.append(r)
        if r.failure:
            result.failures.append(f"route {r.start} -> {r.goal}: {r.failure}")
        log(f"[4/4] simulate  route {i}/{len(pairs)} {r.start} -> {r.goal}: "
            + (r.failure or f"{r.length_m:.1f} m, {r.wall_contacts} wall contacts"))
    return result


def format_report(result: PipelineResult) -> str:
    lines = ["", "--- report ---"]
    lines += [f"note: {n}" for n in result.notes]
    if result.graph is not None:
        g = result.graph
        n_open = sum(d.startswith("open_") for d in g.doors)
        lines.append(f"rooms: {len(g.rooms)}")
        lines.append(f"doors: {len(g.doors) - n_open} (+ {n_open} open connections without a wall)")
    if result.doorless:
        lines.append(f"rooms without a door: {', '.join(result.doorless)}. They cannot be reached. If one "
                     f"should be, its door has no space boundary to it in the IFC: fix the door or the "
                     f"export; shafts can be left out with --exclude")
    if result.validation is not None:
        lines += [f"warning: {i.code}: {i.message}" for i in result.validation.warnings
                  if i.code != "isolated-room"]
    planned = [r for r in result.routes if r.rooms]
    lines.append(f"routes planned: {len(planned)} of {len(result.routes)}")
    for r in planned:
        lines.append(f"  {' > '.join(r.rooms)}: {r.length_m:.1f} m, "
                     + (f"{r.wall_contacts} wall contacts" if r.flown else "not flown to the end"))
    lines.append(f"routes flown: {sum(r.flown for r in result.routes)}")
    lines.append(f"wall contacts: {result.wall_contacts}")
    lines += [f"FAILED {f}" for f in result.failures]
    lines.append(f"RESULT: {'PASS' if result.ok else 'FAIL'} ({len(result.failures)} failures)")
    return "\n".join(lines)


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
