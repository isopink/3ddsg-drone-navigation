"""Live flight view in Rerun (https://rerun.io): map, plan and drone position while it flies.

    view = LiveView()                      # opens the Rerun viewer
    view.log_scene(graph, plan)
    recorder = FlightRecorder(..., listeners=[view.listener("real flight")])
    with view.follow("real flight", drone):
        follow_path(drone, plan.waypoints, recorder=recorder)

The same calls replay a saved run (`replay_run`). Needs `pip install rerun-sdk`.

Entities: world/map (occupied, not observed), world/rooms, world/doors, world/plan,
world/flights/<flight>/{trajectory,drone}, plots/{deviation,height,battery}/<flight>,
events. Timeline "flight_time" is the flight clock in seconds.
"""
from __future__ import annotations

import re
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..flight.record import FlightRun, Listener, bad_states
from ..graph.model import Point, SceneGraph, VoxelState
from ..report import deviations

TIMELINE = "flight_time"
MAX_UNKNOWN_POINTS = 60000

# reference palette (dataviz skill), dark-surface steps: Rerun's viewer is dark
PLAN_RGB = (0x39, 0x87, 0xE5)
FLIGHT_RGB = [(0xD9, 0x59, 0x26), (0x19, 0x9E, 0x70)]
OCCUPIED_RGB = (0xC3, 0xC2, 0xB7)
UNKNOWN_RGBA = (0x89, 0x87, 0x81, 60)
ROOM_RGB = (0x89, 0x87, 0x81)
CRITICAL_RGB = (0xD0, 0x3B, 0x3B)


def _rr():
    try:
        import rerun as rr
    except ImportError as e:  # pragma: no cover
        raise ImportError("the live view needs: pip install rerun-sdk") from e
    return rr


def _viewer_path() -> str | None:
    """The `rerun` viewer installed next to this Python (works without activating the venv)."""
    import shutil

    folder = Path(sys.executable).parent
    for name in ("rerun.exe", "rerun"):
        if (folder / name).is_file():
            return str(folder / name)
    return shutil.which("rerun")


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or "flight"


class LiveView:
    """Logs a scene and flights to Rerun. `spawn` opens the viewer; `save` writes an .rrd file."""

    def __init__(self, app_id: str = "ddsg_flight", spawn: bool = True, save: str | Path | None = None) -> None:
        rr = self.rr = _rr()
        rr.init(app_id, spawn=False)
        sinks = []
        self.viewer = False
        if spawn:
            try:
                rr.spawn(connect=False, hide_welcome_screen=True, executable_path=_viewer_path())
                sinks.append(rr.GrpcSink())
                self.viewer = True
            except Exception as e:  # no viewer: keep flying, the .rrd file still records it
                print(f"could not start the Rerun viewer ({e}); open the .rrd file later with: rerun <file>")
        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(rr.FileSink(str(save)))
        if sinks:
            rr.set_sinks(*sinks)
        self._waypoints = np.zeros((0, 3))
        self._flights: list[str] = []
        self._bad: dict[str, list[str]] = {}  # last bad drone state per flight
        self._send_blueprint()

    def _send_blueprint(self) -> None:
        import rerun.blueprint as rrb

        self.rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Map", origin="world"),
                    rrb.Vertical(
                        rrb.TimeSeriesView(name="Distance from plan (m)", origin="plots/deviation"),
                        rrb.TimeSeriesView(name="Height (m)", origin="plots/height"),
                        rrb.TimeSeriesView(name="Battery (V)", origin="plots/battery"),
                        rrb.TextLogView(name="Events", origin="events"),
                    ),
                    column_shares=[3, 2],
                ),
                rrb.TimePanel(state="expanded"),
            )
        )

    def color(self, label: str) -> tuple[int, int, int]:
        if label not in self._flights:
            self._flights.append(label)
        return FLIGHT_RGB[min(self._flights.index(label), len(FLIGHT_RGB) - 1)]

    # -- static scene ----------------------------------------------------------
    def log_scene(self, graph: SceneGraph, plan: Any | None = None, show_unknown: bool = True) -> None:
        rr = self.rr
        g = graph.voxels
        origin, res = np.asarray(g.origin), g.resolution
        occ = np.argwhere(g.states == int(VoxelState.OCCUPIED))
        rr.log(
            "world/map/occupied",
            rr.Boxes3D(
                centers=origin + (occ + 0.5) * res,
                half_sizes=np.full((len(occ), 3), res / 2),
                colors=OCCUPIED_RGB,
                fill_mode="solid",
            ),
            static=True,
        )
        if show_unknown:
            unk = np.argwhere(g.states == int(VoxelState.UNKNOWN))
            if len(unk) > MAX_UNKNOWN_POINTS:
                unk = unk[:: int(np.ceil(len(unk) / MAX_UNKNOWN_POINTS))]
            rr.log(
                "world/map/not_observed",
                rr.Points3D(origin + (unk + 0.5) * res, radii=res * 0.15, colors=UNKNOWN_RGBA),
                static=True,
            )
        rooms = list(graph.rooms.values())
        if rooms:
            rr.log(
                "world/rooms",
                rr.Boxes3D(
                    mins=[r.bounds.min for r in rooms],
                    sizes=[r.bounds.size for r in rooms],
                    labels=[r.id if r.name == r.id else f"{r.name} ({r.id})" for r in rooms],
                    colors=ROOM_RGB,
                    fill_mode="majorwireframe",
                ),
                static=True,
            )
        doors = list(graph.doors.values())
        if doors:
            rr.log(
                "world/doors",
                rr.Points3D(
                    [d.position for d in doors],
                    radii=0.05,
                    colors=OCCUPIED_RGB,
                    labels=[f"{d.id} {d.room_a}-{d.room_b}" for d in doors],
                ),
                static=True,
            )
        if plan is not None:
            self.log_plan(plan.waypoints if hasattr(plan, "waypoints") else plan["waypoints"])

    def log_plan(self, waypoints: Sequence[Point]) -> None:
        rr = self.rr
        self._waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 3)
        rr.log("world/plan", rr.LineStrips3D([self._waypoints], colors=PLAN_RGB, radii=0.012), static=True)
        rr.log("world/plan/waypoints", rr.Points3D(self._waypoints, colors=PLAN_RGB, radii=0.025), static=True)
        rr.log("plots/height/planned", rr.SeriesLines(colors=PLAN_RGB, names="planned"), static=True)

    def _series_style(self, label: str) -> None:
        rr, s, c = self.rr, slug(label), self.color(label)
        for plot in ("deviation", "height", "battery"):
            rr.log(f"plots/{plot}/{s}", rr.SeriesLines(colors=c, names=label, widths=2), static=True)

    # -- flight samples ----------------------------------------------------------
    def log_samples(self, label: str, rows: np.ndarray, full_track: np.ndarray | None = None) -> None:
        """Log trajectory rows (t, x, y, z). `full_track` is everything flown so far."""
        rr, s, c = self.rr, slug(label), self.color(label)
        if len(rows) == 0:
            return
        dev = deviations(rows[:, 1:4], self._waypoints) if len(self._waypoints) else None
        for i, row in enumerate(rows):
            rr.set_time(TIMELINE, duration=float(row[0]))
            rr.log(f"world/flights/{s}/drone", rr.Points3D([row[1:4]], colors=c, radii=0.05))
            rr.log(f"plots/height/{s}", rr.Scalars(float(row[3])))
            if len(self._waypoints):
                rr.log("plots/height/planned", rr.Scalars(float(self._waypoints[:, 2].min())))
            if dev is not None:
                rr.log(f"plots/deviation/{s}", rr.Scalars(float(dev[i])))
        track = full_track if full_track is not None else rows
        rr.set_time(TIMELINE, duration=float(rows[-1, 0]))
        rr.log(f"world/flights/{s}/trajectory", rr.LineStrips3D([track[:, 1:4]], colors=c, radii=0.01))

    def log_telemetry(self, label: str, rows: Sequence[tuple[float, float, int]]) -> None:
        """Log telemetry rows (t, battery V, supervisor.info): the battery plot, and an event
        whenever the drone enters or leaves a bad state (locked, crashed, tumbled)."""
        rr, s = self.rr, slug(label)
        for t, vbat, info in rows:
            rr.set_time(TIMELINE, duration=float(t))
            rr.log(f"plots/battery/{s}", rr.Scalars(float(vbat)))
            bad = bad_states(info)
            if bad != self._bad.get(label, []):
                text = f"[{label}] drone state {bad}" if bad else f"[{label}] drone state ok"
                rr.log("events", rr.TextLog(text, level="WARN" if bad else "INFO"))
                self._bad[label] = bad

    def listener(self, label: str, battery: bool = True) -> Listener:
        """A FlightRecorder listener that logs status samples and events."""
        rr, s = self.rr, slug(label)
        self.color(label)
        self._series_style(label)
        last_position: list[Any] = [None]

        def on_record(kind: str, rec: dict[str, Any]) -> None:
            rr.set_time(TIMELINE, duration=float(rec["t"]))
            if kind == "status":
                if rec.get("position") is not None:
                    last_position[0] = rec["position"]
                if battery and rec.get("battery_v") is not None:
                    rr.log(f"plots/battery/{s}", rr.Scalars(float(rec["battery_v"])))
                bad = [k for k, v in rec.get("flags", {}).items() if v and k in ("isLocked", "isCrashed", "isTumbled")]
                if bad:
                    rr.log("events", rr.TextLog(f"[{label}] drone state {bad}", level="WARN"))
            else:
                level = "ERROR" if rec["kind"] in ("abort", "error") else "INFO"
                rr.log("events", rr.TextLog(f"[{label}] {rec['kind']}: {rec['message']}", level=level))
                where = rec.get("position") or last_position[0]
                if level == "ERROR" and where is not None:
                    rr.log(f"world/flights/{s}/abort", rr.Points3D([where], colors=CRITICAL_RGB, radii=0.08))

        return on_record

    @contextmanager
    def follow(self, label: str, backend: Any, period_s: float = 0.1) -> Iterator[None]:
        """Stream `backend.trajectory` and `backend.telemetry` to the viewer while the block
        runs (background thread)."""
        stop = threading.Event()
        self._series_style(label)

        def run() -> None:
            sent = sent_tel = 0
            while True:
                done = stop.wait(period_s)
                traj = list(getattr(backend, "trajectory", []))
                if len(traj) > sent:
                    rows = np.asarray(traj, dtype=float).reshape(-1, 4)
                    self.log_samples(label, rows[sent:], full_track=rows)
                    sent = len(rows)
                tel = list(getattr(backend, "telemetry", []))
                if len(tel) > sent_tel:
                    self.log_telemetry(label, tel[sent_tel:])
                    sent_tel = len(tel)
                if done:
                    return

        thread = threading.Thread(target=run, name=f"rerun-{slug(label)}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=5)


def replay_run(run: FlightRun, view: LiveView, chunk_s: float = 0.5) -> None:
    """Log a saved run (flight/record.py) as if it were flying."""
    view.log_scene(run.graph)
    view.log_plan(run.waypoints)
    for f in run.flights:
        on_record = view.listener(f.label, battery=f.backend != "sim")
        rows = np.asarray(f.trajectory, dtype=float).reshape(-1, 4)
        # in half-second chunks, so the trail grows when scrubbing the timeline
        start = 0
        while start < len(rows):
            end = int(np.searchsorted(rows[:, 0], rows[start, 0] + chunk_s, side="right"))
            end = max(end, start + 1)
            view.log_samples(f.label, rows[start:end], full_track=rows[:end])
            start = end
        if f.telemetry:
            view.log_telemetry(f.label, f.telemetry)
        for st in f.status:
            on_record("status", st)
        for ev in f.events:
            rec = dict(ev)
            if ev["kind"] in ("abort", "error") and len(rows):
                i = int(np.clip(np.searchsorted(rows[:, 0], ev["t"]), 0, len(rows) - 1))
                rec["position"] = rows[i, 1:4].tolist()
            on_record("event", rec)
