"""Flight records: what happened during a flight, saved as a run folder.

A `FlightRecorder` is passed to `follow_path`. It collects drone status samples (battery,
state flags, position) and events (take-off, each waypoint, abort, landing) on the same
clock as the backend's `trajectory`, and can forward them to live listeners (the Rerun
view in viz/live.py).

A run folder holds one route and every flight of it (a simulation dry run and the real
flight, for example), so the report can put plan, simulation and real flight side by side:

    runs/20261115_101500/
        run.json          meta, plan, and per flight: result, comparison, status, events
        graph/            the scene graph flown on (graph/io.py format)
        flight_1.csv      trajectory (t, x, y, z) of each flight, graph frame
        report.html       written by scripts/flight_report.py

runs/ is git-ignored: it can hold lab graphs.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..graph.io import load_graph, save_graph
from ..graph.model import Point, SceneGraph
from .backend import Status
from .compare import compare_to_plan, load_trajectory, save_trajectory

FORMAT = "ddsg-flight-run"
VERSION = 1

Trajectory = list[tuple[float, float, float, float]]


@dataclass
class FlightLog:
    """One flight of the planned route."""

    label: str  # "simulation", "simulation (dry run)", "real flight"
    backend: str  # "sim" or "crazyflie"
    trajectory: Trajectory = field(default_factory=list)  # (t, x, y, z), graph frame
    status: list[dict[str, Any]] = field(default_factory=list)  # t, battery_v, flags, position
    events: list[dict[str, Any]] = field(default_factory=list)  # t, kind, message, (index)
    aborted: str | None = None  # why the flight stopped early, None if it completed
    collisions: int | None = None  # simulation only: steps in contact with a wall
    comparison: dict[str, float] | None = None  # compare.PlanComparison fields

    @property
    def ok(self) -> bool:
        return self.aborted is None and not self.collisions


Listener = Callable[[str, dict[str, Any]], None]  # (kind "status" | "event", record)


class FlightRecorder:
    """Collects status samples and events for one flight.

    `clock` must use the same time base as the backend's trajectory, so status, events
    and positions line up (`backend.clock`).
    """

    def __init__(
        self,
        label: str,
        backend: str,
        clock: Callable[[], float] = time.monotonic,
        listeners: Sequence[Listener] = (),
    ) -> None:
        self.log = FlightLog(label=label, backend=backend)
        self.clock = clock
        self.listeners = list(listeners)

    def _emit(self, kind: str, record: dict[str, Any]) -> None:
        for listener in self.listeners:
            try:
                listener(kind, record)
            except Exception as e:  # a broken viewer must never stop a flight
                print(f"flight listener warning: {e}")

    def status(self, status: Status) -> None:
        record = {
            "t": self.clock(),
            "battery_v": status.battery_v,
            "flags": {k: bool(v) for k, v in status.flags.items()},
            "position": list(status.position) if status.position is not None else None,
        }
        self.log.status.append(record)
        self._emit("status", record)

    def event(self, kind: str, message: str = "", **data: Any) -> None:
        record = {"t": self.clock(), "kind": kind, "message": message, **data}
        self.log.events.append(record)
        self._emit("event", record)

    def finish(
        self,
        trajectory: Sequence[tuple[float, float, float, float]],
        waypoints: Sequence[Point],
        aborted: str | None = None,
        collisions: int | None = None,
    ) -> FlightLog:
        log = self.log
        log.trajectory = [tuple(float(v) for v in row) for row in trajectory]  # type: ignore[misc]
        log.aborted = aborted
        log.collisions = collisions
        if log.trajectory and waypoints:
            log.comparison = asdict(compare_to_plan(waypoints, log.trajectory))
        return log


@dataclass
class FlightRun:
    """A planned route and its flights, as saved in a run folder."""

    graph: SceneGraph
    plan: dict[str, Any]  # rooms, doors, keypoints, waypoints
    flights: list[FlightLog]
    meta: dict[str, Any] = field(default_factory=dict)  # text, created, height, margin, ...
    folder: Path | None = None

    @property
    def waypoints(self) -> list[Point]:
        return [tuple(p) for p in self.plan["waypoints"]]  # type: ignore[misc]


def plan_dict(plan: Any) -> dict[str, Any]:
    """A planner.route.RoutePlan as plain JSON data."""
    return {
        "rooms": list(plan.rooms),
        "doors": list(plan.doors),
        "keypoints": [list(p) for p in plan.keypoints],
        "waypoints": [list(p) for p in plan.waypoints],
    }


def new_run_folder(root: str | Path = "runs", stamp: str | None = None) -> Path:
    """runs/<YYYYmmdd_HHMMSS>, with a suffix if that already exists."""
    root = Path(root)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    folder, n = root / stamp, 2
    while folder.exists():
        folder, n = root / f"{stamp}_{n}", n + 1
    return folder


def save_run(run: FlightRun, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_graph(run.graph, folder / "graph")
    flights = []
    for i, f in enumerate(run.flights, start=1):
        csv_name = f"flight_{i}.csv"
        save_trajectory(folder / csv_name, f.trajectory)
        flights.append(
            {
                "label": f.label,
                "backend": f.backend,
                "aborted": f.aborted,
                "collisions": f.collisions,
                "comparison": f.comparison,
                "trajectory_file": csv_name,
                "status": f.status,
                "events": f.events,
            }
        )
    meta = {"created": datetime.now().isoformat(timespec="seconds"), **run.meta}
    doc = {"format": FORMAT, "version": VERSION, "meta": meta, "plan": run.plan, "flights": flights}
    (folder / "run.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    run.folder = folder
    return folder


def load_run(folder: str | Path) -> FlightRun:
    folder = Path(folder)
    path = folder / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (is this a run folder?)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    if doc.get("version") != VERSION:
        raise ValueError(f"{path}: version {doc.get('version')} (supported: {VERSION})")
    flights = []
    for f in doc["flights"]:
        traj_path = folder / f["trajectory_file"]
        flights.append(
            FlightLog(
                label=f["label"],
                backend=f["backend"],
                trajectory=load_trajectory(traj_path) if traj_path.is_file() else [],
                status=f.get("status", []),
                events=f.get("events", []),
                aborted=f.get("aborted"),
                collisions=f.get("collisions"),
                comparison=f.get("comparison"),
            )
        )
    return FlightRun(
        graph=load_graph(folder / "graph"),
        plan=doc["plan"],
        flights=flights,
        meta=doc.get("meta", {}),
        folder=folder,
    )
