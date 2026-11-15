"""Flight report: one self-contained HTML page per run folder (flight/record.py).

Shows the plan next to every flight of it (simulation dry run, real flight): a top view
on the occupancy map, deviation from the plan and height over time, battery, a 3D view,
and the event log. Charts are Plotly, loaded from a CDN, or inlined with `offline=True`
(needs `pip install plotly`, adds ~4.5 MB).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from .flight.monitor import FlightMonitor
from .flight.record import FlightLog, FlightRun, bad_states, load_run
from .graph.model import VoxelState

PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"
MAX_TRAJECTORY_POINTS = 4000
MAX_VOXELS_3D = 15000


def deviations(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance of each point (n, 3) to the polyline (m, 3), vectorized."""
    if len(points) == 0:
        return np.zeros(0)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    pa = points[:, None, :] - a[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom > 0, np.einsum("nij,ij->ni", pa, ab) / denom, 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _downsample(rows: np.ndarray, limit: int) -> np.ndarray:
    if len(rows) <= limit:
        return rows
    idx = np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))
    return rows[idx]


def _round(values: Any, digits: int = 3) -> Any:
    """Rounded list; NaN becomes None (a gap in the chart)."""
    arr = np.round(np.asarray(values, dtype=float), digits)
    return [None if np.isnan(v) else float(v) for v in arr.ravel()]


def _surface_voxels(occupied: np.ndarray) -> np.ndarray:
    """Occupied voxels with at least one face that is not occupied (the visible shell)."""
    padded = np.pad(occupied, 1, constant_values=False)
    inner = np.ones_like(occupied)
    for axis in range(3):
        for shift in (-1, 1):
            inner &= np.roll(padded, shift, axis=axis)[1:-1, 1:-1, 1:-1]
    return occupied & ~inner


def _flight_data(f: FlightLog, waypoints: np.ndarray) -> dict[str, Any]:
    traj = np.asarray(f.trajectory, dtype=float).reshape(-1, 4)
    traj = _downsample(traj, MAX_TRAJECTORY_POINTS)
    dev = deviations(traj[:, 1:4], waypoints) if len(traj) else np.zeros(0)
    # like compare_to_plan: take-off and landing (below 80% of flight height) are vertical
    # and would only show up as distance from the plan, so they are left out of the chart
    if len(traj) and len(waypoints):
        low = traj[:, 3] < 0.8 * waypoints[:, 2].min()
        if not low.all():
            dev = np.where(low, np.nan, dev)
    abort_point = None
    if f.aborted and len(traj):
        abort_ev = next((e for e in f.events if e["kind"] in ("abort", "error")), None)
        i = len(traj) - 1
        if abort_ev is not None:
            i = int(np.clip(np.searchsorted(traj[:, 0], abort_ev["t"]), 0, len(traj) - 1))
        abort_point = _round(traj[i, 1:4])
    # battery: the 20 Hz telemetry when the backend logged it, else the per-waypoint status
    if f.telemetry:
        tel = _downsample(np.asarray(f.telemetry, dtype=float).reshape(-1, 3), MAX_TRAJECTORY_POINTS)
        battery = {"t": _round(tel[:, 0], 2), "v": _round(tel[:, 1]), "source": "telemetry"}
    else:
        st = [s for s in f.status if s.get("battery_v") is not None]
        battery = {"t": [round(s["t"], 2) for s in st], "v": [s["battery_v"] for s in st], "source": "status"}
    batteries = [v for v in battery["v"] if v is not None]
    # the drone entering or leaving a bad state (locked, crashed, tumbled), from the telemetry
    state_events, last = [], []
    for t, _v, info in f.telemetry:
        bad = bad_states(info)
        if bad != last:
            state_events.append({"t": t, "kind": "state", "message": ", ".join(bad) if bad else "ok"})
            last = bad
    events = sorted(f.events + state_events, key=lambda e: e["t"])
    return {
        "label": f.label,
        "backend": f.backend,
        "ok": f.ok,
        "aborted": f.aborted,
        "collisions": f.collisions,
        "comparison": f.comparison,
        "t": _round(traj[:, 0], 2),
        "x": _round(traj[:, 1]),
        "y": _round(traj[:, 2]),
        "z": _round(traj[:, 3]),
        "deviation": _round(dev),
        "abort_point": abort_point,
        "status": [
            {"t": round(s["t"], 2), "battery_v": s.get("battery_v"), "flags": s.get("flags", {})}
            for s in f.status
        ],
        "battery": battery,
        # the simulator has no battery model and always reports the same value
        "battery_real": f.backend != "sim" and bool(batteries),
        "min_battery_v": min(batteries) if batteries and f.backend != "sim" else None,
        "events": [{**e, "t": round(e["t"], 2)} for e in events],
    }


def report_data(run: FlightRun) -> dict[str, Any]:
    """Everything the page needs, as plain JSON data."""
    g = run.graph
    grid = g.voxels
    waypoints = np.asarray(run.waypoints, dtype=float).reshape(-1, 3)
    height = float(waypoints[:, 2].min()) if len(waypoints) else 0.5
    k = int(np.clip(grid.index((0.0, 0.0, height))[2], 0, grid.shape[2] - 1))
    res = grid.resolution
    xs = grid.origin[0] + (np.arange(grid.shape[0]) + 0.5) * res
    ys = grid.origin[1] + (np.arange(grid.shape[1]) + 0.5) * res

    occupied = grid.states == int(VoxelState.OCCUPIED)
    # cut 1 m above flight height, so the ceiling does not hide the route
    cut_z = height + 1.0
    k_cut = int(np.clip(grid.index((0.0, 0.0, cut_z))[2], 1, grid.shape[2]))
    occupied = occupied.copy()
    occupied[:, :, k_cut:] = False
    shell = np.argwhere(_surface_voxels(occupied))
    if len(shell) > MAX_VOXELS_3D:
        shell = shell[:: int(np.ceil(len(shell) / MAX_VOXELS_3D))]
    centers = np.asarray(grid.origin) + (shell + 0.5) * res

    monitor = FlightMonitor()
    meta = dict(run.meta)
    return {
        "name": run.folder.name if run.folder else "",
        "meta": meta,
        "plan": run.plan,
        "height": height,
        "limits": {
            "max_deviation_m": float(meta.get("max_deviation", monitor.max_deviation_m)),
            "min_battery_v": monitor.min_battery_v,
        },
        "map": {
            "x": _round(xs),
            "y": _round(ys),
            # rows = y, columns = x, values = VoxelState at flight height
            "states": grid.states[:, :, k].T.astype(int).tolist(),
            "slice_z": round(grid.origin[2] + (k + 0.5) * res, 3),
            "resolution": res,
        },
        "rooms": [
            {
                "id": r.id,
                "name": r.name,
                "min": list(r.bounds.min),
                "max": list(r.bounds.max),
                "polygon": [list(p) for p in r.polygon] if r.polygon else None,
                "center": list(r.center),
            }
            for r in g.rooms.values()
        ],
        "doors": [
            {"id": d.id, "rooms": [d.room_a, d.room_b], "position": list(d.position), "width": d.width}
            for d in g.doors.values()
        ],
        "voxels3d": {"cut_z": round(cut_z, 2), "x": _round(centers[:, 0]), "y": _round(centers[:, 1]), "z": _round(centers[:, 2])},
        "flights": [_flight_data(f, waypoints) for f in run.flights],
    }


def render_report(run: FlightRun, offline: bool = False) -> str:
    data = report_data(run)
    if offline:
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as e:  # pragma: no cover
            raise ImportError("offline reports need: pip install plotly") from e
        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    else:
        plotly_tag = f'<script src="{PLOTLY_CDN}"></script>'
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = f"Flight report {data['name']}".strip()
    return (
        _TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__PLOTLY__", plotly_tag)
        .replace("__DATA__", payload)
    )


def write_report(folder: str | Path, out: str | Path | None = None, offline: bool = False) -> Path:
    run = load_run(folder)
    out = Path(out) if out else Path(folder) / "report.html"
    out.write_text(render_report(run, offline=offline), encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__PLOTLY__
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --good: #0ca30c; --good-text: #006300; --serious: #ec835a; --critical: #d03b3b;
  --occupied: #52514e; --unknown: #e1e0d9; --free: #fcfcfb;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --good-text: #0ca30c; --occupied: #c3c2b7; --unknown: #383835; --free: #1a1a19;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --good-text: #0ca30c; --occupied: #c3c2b7; --unknown: #383835; --free: #1a1a19;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.sub { color: var(--ink-2); margin: 0; }
.note { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
header { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: flex-end;
  justify-content: space-between; margin-bottom: 16px; }
.meta { color: var(--ink-2); font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 16px; }
.meta b { color: var(--ink); font-weight: 600; }
.verdicts { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.verdict { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
.verdict .icon { width: 18px; height: 18px; border-radius: 50%; display: grid; place-items: center;
  color: #fff; font-size: 12px; font-weight: 700; flex: none; }
.verdict .line-key { width: 16px; height: 3px; border-radius: 2px; flex: none; }
.verdict small { color: var(--ink-2); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px;
  margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .value small { font-size: 13px; font-weight: 400; color: var(--ink-2); }
.tile .hint { color: var(--muted); font-size: 12px; }
.tile .hint.bad { color: var(--critical); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; min-width: 0; }
.grid2 { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 12px; margin-bottom: 12px;
  align-items: start; }
.stack { display: grid; gap: 12px; }
.row { margin-bottom: 12px; }
.chart { width: 100%; height: 300px; }
.chart.tall { height: 560px; }
#map.chart.tall { height: auto; min-height: 260px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 12px; margin-top: 8px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--border); }
.lk { width: 16px; height: 3px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tablewrap { overflow-x: auto; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--ink-2); font-weight: 600; padding: 4px 0; }
.kind { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
button.theme { background: var(--surface); color: var(--ink-2); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; font: inherit; font-size: 12px; cursor: pointer; }
@media (max-width: 900px) { .grid2 { grid-template-columns: minmax(0, 1fr); } .chart.tall { height: 420px; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1 id="title">Flight report</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
      <div class="meta" id="meta"></div>
      <button class="theme" id="theme" type="button">Theme: auto</button>
    </div>
  </header>

  <div class="verdicts" id="verdicts"></div>
  <div class="tiles" id="tiles"></div>
  <div class="card row" id="compare-card" hidden>
    <h2>Flights compared</h2>
    <p class="note">Measured at flight height; take-off and landing are left out.</p>
    <div class="tablewrap"><table id="compare"></table></div>
  </div>

  <div class="grid2">
    <div class="card">
      <h2>Top view</h2>
      <p class="note" id="map-note"></p>
      <div class="chart tall" id="map"></div>
      <div class="legend" id="map-legend"></div>
    </div>
    <div class="stack">
      <div class="card">
        <h2>Distance from the planned path</h2>
        <p class="note">At flight height (take-off and landing left out). The flight monitor lands the drone above the limit.</p>
        <div class="chart" id="deviation"></div>
      </div>
      <div class="card">
        <h2>Height</h2>
        <p class="note">Planned flight height for reference.</p>
        <div class="chart" id="height"></div>
      </div>
    </div>
  </div>

  <div class="card row" id="battery-card" hidden>
    <h2>Battery</h2>
    <p class="note" id="battery-note">Real flights only.</p>
    <div class="chart" id="battery"></div>
  </div>

  <div class="card row">
    <h2>3D view</h2>
    <p class="note" id="view3d-note">Occupied voxels (outer shell), plan and flights. Drag to rotate.</p>
    <div class="chart tall" id="view3d"></div>
  </div>

  <div class="card row">
    <h2>Events</h2>
    <div class="tablewrap"><table id="events"></table></div>
    <details>
      <summary>Planned waypoints</summary>
      <div class="tablewrap"><table id="waypoints"></table></div>
    </details>
  </div>
</main>

<script>
const D = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const flightVar = (i) => ["--s2", "--s3"][Math.min(i, 1)];   // plan is --s1
const main = D.flights.length ? D.flights[D.flights.length - 1] : null;

// ---- theme toggle (auto / light / dark), remembered per viewer --------------
const THEMES = ["auto", "light", "dark"];
let theme = "auto";
try { theme = localStorage.getItem("ddsg-report-theme") || "auto"; } catch (e) {}
function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("theme").textContent = "Theme: " + theme;
}
$("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try { localStorage.setItem("ddsg-report-theme", theme); } catch (e) {}
  applyTheme(); drawCharts();
});
applyTheme();

// ---- header, verdicts, tiles ---------------------------------------------------
const m = D.meta;
$("title").textContent = "Flight report" + (D.name ? " · " + D.name : "");
$("subtitle").textContent = (m.text ? `“${m.text}”  ` : "") + "Route " + D.plan.rooms.join(" → ");
$("meta").innerHTML = [
  m.created && `<span>${esc(m.created.replace("T", " "))}</span>`,
  `<span>Height <b>${fmt(D.height)} m</b></span>`,
  m.velocity !== undefined && `<span>Speed <b>${fmt(m.velocity)} m/s</b></span>`,
  m.margin !== undefined && `<span>Margin <b>${fmt(m.margin)} m</b></span>`,
  `<span>Waypoints <b>${D.plan.waypoints.length}</b></span>`,
].filter(Boolean).join("");

function verdict(f) {
  if (f.aborted) return { icon: "✕", color: "--critical", text: "Aborted", detail: f.aborted };
  if (f.collisions) return { icon: "!", color: "--serious", text: `${f.collisions} wall contacts`, detail: "in the simulation" };
  return { icon: "✓", color: "--good", text: "Completed", detail: "" };
}
$("verdicts").innerHTML = D.flights.map((f, i) => {
  const v = verdict(f);
  return `<div class="verdict"><span class="line-key" style="background:var(${flightVar(i)})"></span>
    <span>${esc(f.label)}</span>
    <span class="icon" style="background:var(${v.color})" aria-hidden="true">${v.icon}</span>
    <b>${esc(v.text)}</b>${v.detail ? `<small>${esc(v.detail)}</small>` : ""}</div>`;
}).join("") || `<div class="verdict">No flights recorded</div>`;

if (main && main.comparison) {
  const c = main.comparison, lim = D.limits.max_deviation_m;
  const tiles = [
    ["Duration", fmt(c.duration_s, 1), "s", `${c.samples} samples`],
    ["Max deviation", fmt(c.max_deviation_m), "m", `limit ${fmt(lim)} m`, c.max_deviation_m > lim],
    ["Mean deviation", fmt(c.mean_deviation_m), "m", ""],
    ["End error (xy)", fmt(c.end_error_m), "m", "last point vs goal"],
    ["Path flown", fmt(c.flown_length_m, 1), "m", `planned ${fmt(c.planned_length_m, 1)} m`],
  ];
  if (main.min_battery_v !== null) tiles.push(["Lowest battery", fmt(main.min_battery_v), "V", `land below ${fmt(D.limits.min_battery_v, 1)} V`, main.min_battery_v < D.limits.min_battery_v]);
  $("tiles").innerHTML = `<p class="note" style="grid-column:1/-1;margin:0">Figures for <b style="color:var(--ink-2)">${esc(main.label)}</b></p>` +
    tiles.map(([label, value, unit, hint, bad]) => `<div class="tile"><div class="label">${label}</div>
      <div class="value">${value} <small>${unit}</small></div><div class="hint${bad ? " bad" : ""}">${bad ? "⚠ " : ""}${esc(hint)}</div></div>`).join("");
}

if (D.flights.length > 1) {
  $("compare-card").hidden = false;
  const rows = [
    ["Result", (f) => esc(verdict(f).text)],
    ["Duration (s)", (f) => fmt(f.comparison?.duration_s, 1)],
    ["Max deviation (m)", (f) => fmt(f.comparison?.max_deviation_m)],
    ["Mean deviation (m)", (f) => fmt(f.comparison?.mean_deviation_m)],
    ["End error xy (m)", (f) => fmt(f.comparison?.end_error_m)],
    ["Path flown (m)", (f) => fmt(f.comparison?.flown_length_m, 1)],
    ["Lowest battery (V)", (f) => fmt(f.min_battery_v)],
  ];
  $("compare").innerHTML = `<thead><tr><th></th>${D.flights.map((f, i) =>
    `<th class="num"><span class="kind"><span class="lk" style="background:var(${flightVar(i)})"></span>${esc(f.label)}</span></th>`).join("")}</tr></thead>
    <tbody>${rows.map(([name, get]) => `<tr><th>${name}</th>${D.flights.map((f) => `<td class="num">${get(f)}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

// ---- tables --------------------------------------------------------------------
const kindColor = { abort: "--critical", error: "--critical", state: "--serious", takeoff: "--muted", waypoint: "--muted", land: "--muted", landed: "--muted" };
const events = D.flights.flatMap((f, i) => f.events.map((e) => ({ ...e, flight: f.label, fi: i })));
$("events").innerHTML = `<thead><tr><th>Flight</th><th class="num">t (s)</th><th>Event</th><th>Details</th></tr></thead><tbody>` +
  events.map((e) => `<tr><td><span class="kind"><span class="lk" style="background:var(${flightVar(e.fi)})"></span>${esc(e.flight)}</span></td>
    <td class="num">${fmt(e.t, 1)}</td><td><span class="kind"><span class="dot" style="background:var(${e.kind === "state" && e.message === "ok" ? "--muted" : kindColor[e.kind] || "--muted"})"></span>${esc(e.kind)}</span></td>
    <td>${esc(e.message)}</td></tr>`).join("") + "</tbody>";
$("waypoints").innerHTML = `<thead><tr><th class="num">#</th><th class="num">x (m)</th><th class="num">y (m)</th><th class="num">z (m)</th></tr></thead><tbody>` +
  D.plan.waypoints.map((p, i) => `<tr><td class="num">${i + 1}</td>${p.map((v) => `<td class="num">${fmt(v)}</td>`).join("")}</tr>`).join("") + "</tbody>";

// ---- charts ---------------------------------------------------------------------
const config = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function baseLayout(extra = {}) {
  const axis = { gridcolor: css("--grid"), linecolor: css("--axis"), zerolinecolor: css("--grid"),
    tickfont: { color: css("--muted") }, title: { font: { color: css("--ink-2"), size: 12 } } };
  return {
    paper_bgcolor: css("--surface"), plot_bgcolor: css("--surface"),
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css("--ink-2"), size: 12 },
    margin: { l: 52, r: 16, t: 28, b: 44 }, showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: css("--ink-2") } },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--border"), font: { color: css("--ink") } },
    xaxis: { ...axis, ...(extra.xaxis || {}) }, yaxis: { ...axis, ...(extra.yaxis || {}) },
    ...Object.fromEntries(Object.entries(extra).filter(([k]) => k !== "xaxis" && k !== "yaxis")),
  };
}

function refLine(y, color, label) {
  return {
    shape: { type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { color, width: 1 } },
    annotation: { xref: "paper", x: 1, y, xanchor: "right", yanchor: "bottom", text: label, showarrow: false,
      font: { color: css("--ink-2"), size: 11 } },
  };
}

function drawMap() {
  const surf = css("--surface");
  const states = { 0: "unknown", 1: "free", 2: "occupied" };
  const traces = [{
    type: "heatmap", x: D.map.x, y: D.map.y, z: D.map.states, zmin: 0, zmax: 2, showscale: false,
    colorscale: [[0, css("--unknown")], [0.333, css("--unknown")], [0.334, css("--free")], [0.666, css("--free")], [0.667, css("--occupied")], [1, css("--occupied")]],
    customdata: D.map.states.map((r) => r.map((v) => states[v])),
    hovertemplate: "x %{x:.2f} m, y %{y:.2f} m<br>%{customdata}<extra></extra>", showlegend: false,
  }];
  const shapes = [], annotations = [];
  for (const r of D.rooms) {
    if (r.polygon) {
      shapes.push({ type: "path", path: "M " + r.polygon.map((p) => p.join(",")).join(" L ") + " Z", line: { color: css("--muted"), width: 1 } });
    } else {
      shapes.push({ type: "rect", x0: r.min[0], y0: r.min[1], x1: r.max[0], y1: r.max[1], line: { color: css("--muted"), width: 1 } });
    }
    annotations.push({ x: r.polygon ? r.center[0] : r.min[0], y: r.polygon ? r.center[1] : r.max[1], xanchor: r.polygon ? "center" : "left",
      yanchor: r.polygon ? "middle" : "top", text: esc(r.name), showarrow: false,
      xshift: r.polygon ? 0 : 14, yshift: r.polygon ? 0 : -12,
      bgcolor: css("--surface"), borderpad: 2, font: { color: css("--ink-2"), size: 12 } });
  }
  if (D.doors.length) traces.push({
    type: "scatter", mode: "markers", name: "Doors", x: D.doors.map((d) => d.position[0]), y: D.doors.map((d) => d.position[1]),
    marker: { symbol: "diamond", size: 10, color: css("--ink-2"), line: { color: surf, width: 2 } },
    customdata: D.doors.map((d) => [d.id, d.rooms.join(" ↔ "), d.width]),
    hovertemplate: "%{customdata[0]}<br>%{customdata[1]}, %{customdata[2]:.2f} m wide<extra></extra>",
  });
  const wp = D.plan.waypoints;
  traces.push({ type: "scatter", mode: "lines", name: "Plan", x: wp.map((p) => p[0]), y: wp.map((p) => p[1]),
    line: { color: css("--s1"), width: 2 }, hovertemplate: "plan<br>x %{x:.2f}, y %{y:.2f}<extra></extra>" });
  const kp = D.plan.keypoints;
  traces.push({ type: "scatter", mode: "markers", name: "Start / doors / goal", showlegend: false, x: kp.map((p) => p[0]), y: kp.map((p) => p[1]),
    marker: { size: 9, color: css("--s1"), line: { color: surf, width: 2 } },
    text: kp.map((_, i) => i === 0 ? "start " + D.plan.rooms[0] : i === kp.length - 1 ? "goal " + D.plan.rooms.at(-1) : "door " + D.plan.doors[i - 1]),
    hovertemplate: "%{text}<extra></extra>" });
  D.flights.forEach((f, i) => {
    const color = css(flightVar(i));
    traces.push({ type: "scatter", mode: "lines", name: f.label, x: f.x, y: f.y, line: { color, width: 2 },
      customdata: f.t.map((t, j) => [t, f.z[j], f.deviation[j]]),
      hovertemplate: `${esc(f.label)}<br>t %{customdata[0]:.1f} s<br>x %{x:.2f}, y %{y:.2f}, z %{customdata[1]:.2f} m<br>%{customdata[2]:.2f} m from plan<extra></extra>` });
    if (f.x.length) traces.push({ type: "scatter", mode: "markers", showlegend: false, x: [f.x.at(-1)], y: [f.y.at(-1)],
      marker: { size: 9, color, line: { color: surf, width: 2 } }, hovertemplate: `${esc(f.label)}: end<extra></extra>` });
    if (f.abort_point) traces.push({ type: "scatter", mode: "markers", name: "Abort", x: [f.abort_point[0]], y: [f.abort_point[1]],
      marker: { symbol: "x", size: 13, color: css("--critical"), line: { color: surf, width: 1 } },
      hovertemplate: `${esc(f.label)} aborted<br>${esc(f.aborted)}<extra></extra>` });
  });
  const el = $("map"), xr = D.map.x.at(-1) - D.map.x[0], yr = D.map.y.at(-1) - D.map.y[0];
  const plotW = Math.max(el.clientWidth - 52, 100);
  el.style.height = Math.round(Math.min(620, Math.max(260, plotW * (yr / xr) + 44 + 16))) + "px";
  Plotly.react("map", traces, baseLayout({
    shapes, annotations, showlegend: false, hovermode: "closest", margin: { l: 44, r: 8, t: 8, b: 36 },
    xaxis: { title: { text: "x (m)" }, constrain: "domain", showgrid: false, range: [D.map.x[0] - D.map.resolution / 2, D.map.x.at(-1) + D.map.resolution / 2] },
    yaxis: { title: { text: "y (m)" }, scaleanchor: "x", scaleratio: 1, showgrid: false, constrain: "domain" },
  }), config);
  $("map-note").textContent = `Occupancy at z = ${fmt(D.map.slice_z)} m (${fmt(D.map.resolution)} m voxels). Hover a line for time and distance from the plan.`;
  $("map-legend").innerHTML = [
    `<span><span class="lk" style="background:var(--s1)"></span>Plan</span>`,
    ...D.flights.map((f, i) => `<span><span class="lk" style="background:var(${flightVar(i)})"></span>${esc(f.label)}</span>`),
    D.flights.some((f) => f.abort_point) && `<span><b style="color:var(--critical)">✕</b>Abort</span>`,
    D.doors.length && `<span><span style="color:var(--ink-2)">◆</span>Door</span>`,
    `<span><span class="sw" style="background:var(--occupied)"></span>Occupied</span>`,
    `<span><span class="sw" style="background:var(--unknown)"></span>Not observed</span>`,
    `<span><span class="sw" style="background:var(--free)"></span>Free</span>`,
  ].filter(Boolean).join("");
}

function timeChart(id, series, refs, yTitle) {
  const traces = series.map((s) => ({ type: "scatter", mode: "lines", name: s.name, x: s.x, y: s.y,
    line: { color: s.color, width: 2 }, hovertemplate: `%{y:.2f} ${s.unit}<extra>${esc(s.name)}</extra>` }));
  Plotly.react(id, traces, baseLayout({
    hovermode: "x unified", showlegend: series.length > 1,
    shapes: refs.map((r) => r.shape), annotations: refs.map((r) => r.annotation),
    xaxis: { title: { text: "time (s)" }, showspikes: true, spikemode: "across", spikethickness: 1, spikecolor: css("--axis"), spikedash: "solid" },
    yaxis: { title: { text: yTitle }, rangemode: "tozero" },
  }), config);
}

function draw3d() {
  const traces = [{ type: "scatter3d", mode: "markers", name: "Occupied", x: D.voxels3d.x, y: D.voxels3d.y, z: D.voxels3d.z,
    marker: { size: 2, color: css("--muted"), opacity: 0.35 }, hoverinfo: "skip" }];
  const wp = D.plan.waypoints;
  traces.push({ type: "scatter3d", mode: "lines", name: "Plan", x: wp.map((p) => p[0]), y: wp.map((p) => p[1]), z: wp.map((p) => p[2]),
    line: { color: css("--s1"), width: 5 } });
  D.flights.forEach((f, i) => traces.push({ type: "scatter3d", mode: "lines", name: f.label, x: f.x, y: f.y, z: f.z,
    line: { color: css(flightVar(i)), width: 5 }, customdata: f.t, hovertemplate: "t %{customdata:.1f} s<br>%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>" }));
  $("view3d-note").textContent = `Occupied voxels up to ${fmt(D.voxels3d.cut_z, 1)} m (outer shell), plan and flights. Drag to rotate, scroll to zoom.`;
  const ax = { backgroundcolor: css("--surface"), gridcolor: css("--grid"), color: css("--muted"), showbackground: true };
  Plotly.react("view3d", traces, baseLayout({
    margin: { l: 0, r: 0, t: 0, b: 0 }, legend: { orientation: "h", y: 0, font: { color: css("--ink-2") } },
    scene: { aspectmode: "data", xaxis: { ...ax, title: "x (m)" }, yaxis: { ...ax, title: "y (m)" }, zaxis: { ...ax, title: "z (m)" },
      camera: { eye: { x: -0.9, y: -1.6, z: 1.1 } } },
  }), config);
}

function drawCharts() {
  if (typeof Plotly === "undefined") {
    for (const id of ["map", "deviation", "height", "view3d"]) $(id).innerHTML = '<p class="note">Charts need Plotly from the internet. Rebuild the report with --offline to embed it.</p>';
    return;
  }
  drawMap();
  timeChart("deviation", D.flights.map((f, i) => ({ name: f.label, x: f.t, y: f.deviation, color: css(flightVar(i)), unit: "m" })),
    [refLine(D.limits.max_deviation_m, css("--critical"), `abort limit ${fmt(D.limits.max_deviation_m)} m`)], "distance (m)");
  timeChart("height", D.flights.map((f, i) => ({ name: f.label, x: f.t, y: f.z, color: css(flightVar(i)), unit: "m" })),
    [refLine(D.height, css("--s1"), `planned ${fmt(D.height)} m`)], "height (m)");
  const bat = D.flights.map((f, i) => ({ f, i })).filter(({ f }) => f.battery_real);
  if (bat.length) {
    $("battery-card").hidden = false;
    $("battery-note").textContent = (bat.every(({ f }) => f.battery.source === "telemetry")
      ? "Logged at 20 Hz with the position. Voltage sags under motor load."
      : "Sampled at take-off and at every waypoint.") + " Real flights only.";
    timeChart("battery", bat.map(({ f, i }) => ({ name: f.label, x: f.battery.t, y: f.battery.v, color: css(flightVar(i)), unit: "V" })),
      [refLine(D.limits.min_battery_v, css("--critical"), `land below ${fmt(D.limits.min_battery_v, 1)} V`)], "voltage (V)");
    Plotly.relayout("battery", { "yaxis.rangemode": "normal" });
  }
  draw3d();
}
drawCharts();
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawCharts);
</script>
</body>
</html>
"""
