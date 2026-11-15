"""Check a scene graph end to end: validate it, pick room pairs, plan routes, fly them in
MuJoCo, and report. Works on any `GraphSource` (fake graph, exchange folder, Neo4j).

`scripts/graph_pipeline.py` runs it on a graph; `scripts/ifc_pipeline.py` converts an IFC
storey first and then calls `check_graph`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from .flight.executor import follow_path
from .flight.monitor import FlightAbort, FlightMonitor
from .graph.io import save_graph
from .graph.model import SceneGraph, VoxelState
from .graph.source import GraphSource
from .planner.collision import check_path
from .planner.global_planner import NoRoomRouteError
from .planner.local_planner import NoPathError
from .planner.policy import PolicyLike, UnknownPolicy
from .planner.route import plan_route
from .validate import ValidationReport, validate_graph


@dataclass(frozen=True)
class Hints:
    """The words that differ between the commands: what the graph is and how to fix it."""

    scope: str = "graph"  # "Use a room id or name of this <scope>"
    no_pairs: str = "Check the doors of the graph"
    # text after "rooms without a door: A, B. "; None lists every unreachable room with its
    # reason instead, plus the grid size and the share of unknown voxels
    doorless: str | None = None


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
    out: Path | None = None  # folder the graph was saved to
    notes: list[str] = field(default_factory=list)  # from the converter or the loader
    validation: ValidationReport | None = None
    doorless: list[str] = field(default_factory=list)  # rooms nothing connects to
    unreachable: dict[str, str] = field(default_factory=dict)  # room -> why it cannot be reached
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


def unknown_share(graph: SceneGraph) -> float:
    """Share of the voxels that were never observed, 0..1."""
    states = graph.voxels.states
    return float((states == int(VoxelState.UNKNOWN)).sum()) / max(states.size, 1)


def unreachable_rooms(graph: SceneGraph, validation: ValidationReport | None = None) -> dict[str, str]:
    """Rooms outside the largest group of connected rooms, each with the reason.

    A door the validation found blocked or out of reach does not connect its rooms.
    """
    errors = [i for i in validation.errors] if validation is not None else []

    def named(kind: str, ident: str) -> list[str]:
        return [i.code for i in errors if re.search(rf"\b{kind} {re.escape(ident)}(?!\w)", i.message)]

    dead_doors = {d for d in graph.doors if named("door", d)}
    g = nx.Graph()
    g.add_nodes_from(graph.rooms)
    g.add_edges_from((d.room_a, d.room_b) for d in graph.doors.values() if d.id not in dead_doors)
    groups = sorted(nx.connected_components(g), key=lambda c: (-len(c), min(c)))
    if len(groups) < 2:
        return {}
    main = groups[0]
    out: dict[str, str] = {}
    for group in groups[1:]:
        for rid in sorted(group):
            doors = sorted(d.id for d in graph.doors.values() if rid in (d.room_a, d.room_b))
            dead = [d for d in doors if d in dead_doors]
            if "room-not-flyable" in named("room", rid):
                why = "no flyable space at the flight height"
            elif not doors:
                why = "no door"
            elif len(dead) == len(doors):
                why = f"blocked or out of reach: door {', '.join(dead)}"
            else:
                others = ", ".join(sorted(group - {rid}))
                why = f"connected only to {others}, not to the main group of {len(main)} rooms"
                if dead:
                    why += f" (blocked: door {', '.join(dead)})"
            out[rid] = why
    return out


def fly_route(graph: SceneGraph, start: str, goal: str, height: float, margin: float,
              velocity: float, policy: PolicyLike = UnknownPolicy.OBSTACLE,
              hints: Hints = Hints()) -> RouteResult:
    """Plan one route, check it against the voxels and fly it in the simulator."""
    r = RouteResult(start, goal)
    try:
        plan = plan_route(graph, start, goal, height, margin, policy)
    except KeyError as e:
        r.failure = (f"plan: {e.args[0]}. Use a room id or name of this {hints.scope}: "
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
    check = check_path(graph, plan.waypoints, margin, policy)
    if not check.ok:
        r.failure = f"plan: the route breaks the {margin} m safety margin ({check.summary()})"
        return r
    try:
        from .flight.backends.mujoco_backend import MuJoCoBackend, SimTimeoutError
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


def check_graph(graph: SceneGraph, result: PipelineResult | None = None, start: str | None = None,
                goal: str | None = None, routes: int = 3, height: float = 0.5, margin: float = 0.25,
                velocity: float = 0.3, policy: PolicyLike = UnknownPolicy.OBSTACLE,
                log=lambda line: None, hints: Hints = Hints()) -> PipelineResult:
    """Steps 2 to 4 on a loaded graph: validate, plan, simulate.

    Stops at the first step that fails; `log` gets progress lines. Step 1 (getting the
    graph) belongs to the caller, which may pass its own `result` to be filled in.
    """
    result = result if result is not None else PipelineResult()
    result.graph = graph
    report = result.validation = validate_graph(graph, height, margin, policy)
    result.failures += [f"validate: {i.code}: {i.message}" for i in report.errors]
    degree = graph.room_graph().degree
    result.doorless = sorted(r for r in graph.rooms if degree[r] == 0)
    result.unreachable = unreachable_rooms(graph, report)
    log(f"[2/4] validate  {len(report.errors)} errors, {len(report.warnings)} warnings")
    if not report.ok:
        return result

    if (start is None) != (goal is None):
        result.failures.append("plan: give both --from and --to, or neither")
        return result
    pairs = [(start, goal)] if start is not None else pick_pairs(graph, routes)
    if not pairs:
        result.failures.append("plan: no two rooms are connected, so there is nothing to fly. "
                               f"{hints.no_pairs}")
        return result
    log(f"[3/4] plan      {len(pairs)} routes at {height} m, {margin} m from walls")
    for i, (a, b) in enumerate(pairs, 1):
        r = fly_route(graph, a, b, height, margin, velocity, policy, hints)
        result.routes.append(r)
        if r.failure:
            result.failures.append(f"route {r.start} -> {r.goal}: {r.failure}")
        log(f"[4/4] simulate  route {i}/{len(pairs)} {r.start} -> {r.goal}: "
            + (r.failure or f"{r.length_m:.1f} m, {r.wall_contacts} wall contacts"))
    return result


def run_source(source: GraphSource, save: str | Path | None = None, start: str | None = None,
               goal: str | None = None, routes: int = 3, height: float = 0.5, margin: float = 0.25,
               velocity: float = 0.3, policy: PolicyLike = UnknownPolicy.OBSTACLE,
               log=lambda line: None, hints: Hints = Hints()) -> PipelineResult:
    """All four steps on any graph source: load (and save to `save`), validate, plan, simulate."""
    result = PipelineResult()
    try:
        graph = source.load()
    except Exception as e:  # a missing folder, a bad file, a database error: all "cannot load"
        result.failures.append(f"load: {e}")
        return result
    result.graph = graph
    grid = graph.voxels
    line = (f"[1/4] load      {len(graph.rooms)} rooms, {len(graph.doors)} doors, {len(graph.solids)} "
            f"wall boxes, voxels {grid.shape} of {grid.resolution} m, {unknown_share(graph):.1%} unknown")
    if save is not None:
        try:
            result.out = save_graph(graph, save)
        except OSError as e:
            log(line)
            result.failures.append(f"save: cannot write {save}: {e}")
            return result
        line += f" -> {result.out}"
    log(line)
    return check_graph(graph, result, start, goal, routes, height, margin, velocity, policy, log, hints)


def format_report(result: PipelineResult, hints: Hints = Hints()) -> str:
    lines = ["", "--- report ---"]
    lines += [f"note: {n}" for n in result.notes]
    if result.graph is not None:
        g = result.graph
        n_open = sum(d.startswith("open_") for d in g.doors)
        lines.append(f"rooms: {len(g.rooms)}")
        lines.append(f"doors: {len(g.doors) - n_open} (+ {n_open} open connections without a wall)")
        if hints.doorless is None:
            nx_, ny, nz = g.voxels.shape
            res = g.voxels.resolution
            lines.append(f"grid: {nx_} x {ny} x {nz} voxels of {res} m "
                         f"({nx_ * res:.1f} x {ny * res:.1f} x {nz * res:.1f} m)")
            lines.append(f"unknown voxels: {unknown_share(g):.1%}")
    if hints.doorless is not None:
        if result.doorless:
            lines.append(f"rooms without a door: {', '.join(result.doorless)}. {hints.doorless}")
    elif result.validation is not None:
        lines.append(f"rooms that cannot be reached: {len(result.unreachable)}")
        lines += [f"  {rid}: {why}" for rid, why in result.unreachable.items()]
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
