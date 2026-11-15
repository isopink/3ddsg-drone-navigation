"""Runs dashboard: a local web page that shows every flight in runs/ as it lands.

    cf-env\\Scripts\\python scripts\\dashboard.py            # or double-click start_dashboard.bat
    cf-env\\Scripts\\python scripts\\dashboard.py --runs D:\\flights --port 8800

Keep it open next to the terminal while you experiment: each `fly_route.py` run appears
within two seconds and its report opens by itself. "Open in Rerun" replays a run in the
3D viewer. It only reads results; flights are still started from the terminal.

Serves on 127.0.0.1 only (this computer). Reports are built on demand when missing or
older than the run.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # also works without `pip install -e .`

from ddsg_nav.report import write_report  # noqa: E402

NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")
_report_lock = threading.Lock()


def run_summary(folder: Path) -> dict | None:
    """What the run list shows, read from run.json. None while the run is still being written."""
    try:
        doc = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    flights = []
    for f in doc.get("flights", []):
        c = f.get("comparison") or {}
        flights.append(
            {
                "label": f.get("label", ""),
                "backend": f.get("backend", ""),
                "aborted": f.get("aborted"),
                "collisions": f.get("collisions"),
                "max_deviation_m": c.get("max_deviation_m"),
                "duration_s": c.get("duration_s"),
            }
        )
    meta = doc.get("meta", {})
    return {
        "name": folder.name,
        "created": meta.get("created", ""),
        "text": meta.get("text", ""),
        "rooms": doc.get("plan", {}).get("rooms", []),
        "flights": flights,
        "has_rrd": (folder / "flight.rrd").is_file(),
        "mtime": (folder / "run.json").stat().st_mtime,
    }


def list_runs(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    runs = [s for p in root.iterdir() if p.is_dir() and NAME.match(p.name) and (s := run_summary(p))]
    return sorted(runs, key=lambda r: (r["created"], r["name"]), reverse=True)


def ensure_report(folder: Path) -> Path:
    report, run_json = folder / "report.html", folder / "run.json"
    with _report_lock:
        if not report.is_file() or report.stat().st_mtime < run_json.stat().st_mtime:
            write_report(folder)
    return report


def open_in_rerun(folder: Path) -> str:
    """Open the run in the Rerun viewer: its live recording if there is one, else a replay."""
    exe_dir = Path(sys.executable).parent
    rrd = folder / "flight.rrd"
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    viewer = next((exe_dir / n for n in ("rerun.exe", "rerun") if (exe_dir / n).is_file()), None)
    if rrd.is_file() and viewer is not None:
        subprocess.Popen([str(viewer), str(rrd)], cwd=ROOT, **flags)
        return "opened the recording"
    subprocess.Popen([sys.executable, str(ROOT / "scripts" / "replay_run.py"), str(folder)], cwd=ROOT, **flags)
    return "replaying"


class Handler(BaseHTTPRequestHandler):
    runs_root: Path = ROOT / "runs"

    def log_message(self, fmt, *args):  # keep the terminal quiet; errors still print
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _run_folder(self, name: str) -> Path | None:
        name = unquote(name)
        folder = self.runs_root / name
        if not NAME.match(name) or not (folder / "run.json").is_file():
            return None
        return folder

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            page = PAGE.replace("__RUNS__", json.dumps(str(self.runs_root.resolve()), ensure_ascii=False))
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/runs":
            return self._json({"root": str(self.runs_root.resolve()), "runs": list_runs(self.runs_root)})
        m = re.fullmatch(r"/run/([^/]+)/report", path)
        if m:
            folder = self._run_folder(m.group(1))
            if folder is None:
                return self._send(404, b"run not found", "text/plain; charset=utf-8")
            try:
                body = ensure_report(folder).read_bytes()
            except Exception as e:  # show the problem in the frame instead of a blank page
                return self._send(500, f"could not build the report: {e}".encode(), "text/plain; charset=utf-8")
            return self._send(200, body, "text/html; charset=utf-8")
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        # only the dashboard page itself may start the viewer (not another website)
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin", "")
        if self.headers.get("X-Dashboard") != "1" or (origin and urlparse(origin).netloc != host):
            return self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
        m = re.fullmatch(r"/api/rerun/([^/]+)", urlparse(self.path).path)
        folder = self._run_folder(m.group(1)) if m else None
        if folder is None:
            return self._json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
        try:
            return self._json({"ok": True, "message": open_in_rerun(folder)})
        except Exception as e:
            return self._json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def serve(runs_root: Path, port: int) -> ThreadingHTTPServer:
    Handler.runs_root = runs_root
    for p in range(port, port + 20):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Handler)
        except OSError:
            continue
    raise OSError(f"no free port in {port}-{port + 19}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"), help="folder with run folders (default: runs)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    runs_root = Path(a.runs)
    runs_root.mkdir(parents=True, exist_ok=True)
    server = serve(runs_root, a.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Runs dashboard: {url}")
    print(f"Watching {runs_root.resolve()}. Close this window or press Ctrl+C to stop.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flight runs</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --border: rgba(11,11,11,0.10); --hover: rgba(11,11,11,0.04); --sel: rgba(42,120,214,0.10);
  --accent: #2a78d6; --good: #0ca30c; --serious: #ec835a; --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.04); --sel: rgba(57,135,229,0.18);
    --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --border: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.04); --sel: rgba(57,135,229,0.18);
  --accent: #3987e5;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; display: flex; flex-direction: column; }
header { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; padding: 12px 16px;
  border-bottom: 1px solid var(--border); background: var(--surface); }
h1 { font-size: 16px; font-weight: 600; margin: 0; }
.watch { color: var(--ink-2); font-size: 12px; display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
.watch code { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 46vw; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--good); flex: none; }
.pulse.off { background: var(--critical); }
.spacer { flex: 1; }
label.toggle { color: var(--ink-2); font-size: 12px; display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
button { background: var(--surface); color: var(--ink-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 10px; font: inherit; font-size: 12px; cursor: pointer; }
button:hover { background: var(--hover); }
main { flex: 1; display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 0; }
aside { border-right: 1px solid var(--border); overflow-y: auto; padding: 8px; }
.run { display: block; width: 100%; text-align: left; border: 1px solid transparent; border-radius: 10px; padding: 10px 12px;
  margin-bottom: 4px; background: transparent; color: var(--ink); cursor: pointer; font-size: 13px; }
.run:hover { background: var(--hover); }
.run.sel { background: var(--sel); border-color: var(--border); }
.run.new { animation: flash 2s ease-out; }
@keyframes flash { from { background: var(--sel); } }
.run .top { display: flex; justify-content: space-between; gap: 8px; }
.run .when { font-weight: 600; }
.run .text { color: var(--ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 2px; }
.run .flights { display: flex; flex-direction: column; gap: 2px; margin-top: 6px; }
.fl { display: flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 12px; }
.fl .icon { width: 14px; height: 14px; border-radius: 50%; display: grid; place-items: center; color: #fff;
  font-size: 9px; font-weight: 700; flex: none; }
.fl .num { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.viewer { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 8px 12px; border-bottom: 1px solid var(--border); }
.bar .title { font-weight: 600; }
.bar .msg { color: var(--muted); font-size: 12px; }
iframe { flex: 1; width: 100%; border: 0; background: var(--page); }
.empty { color: var(--ink-2); padding: 32px; max-width: 640px; }
.empty code { display: block; background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; margin: 8px 0; color: var(--ink); white-space: pre-wrap; word-break: break-all; }
@media (max-width: 800px) {
  main { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto minmax(0, 1fr); }
  aside { border-right: 0; border-bottom: 1px solid var(--border); max-height: 38vh; }
}
</style>
</head>
<body>
<header>
  <h1>Flight runs</h1>
  <span class="watch"><span class="pulse" id="pulse"></span><span id="status">Watching</span> <code id="root"></code></span>
  <span class="spacer"></span>
  <label class="toggle"><input type="checkbox" id="auto" checked> Open new runs automatically</label>
  <button type="button" id="theme">Theme: auto</button>
</header>
<main>
  <aside id="list"></aside>
  <section class="viewer">
    <div class="bar" id="bar" hidden>
      <span class="title" id="sel-title"></span>
      <span class="spacer"></span>
      <span class="msg" id="msg"></span>
      <button type="button" id="rerun">Open in Rerun</button>
      <button type="button" id="newtab">Open in new tab</button>
    </div>
    <div class="empty" id="empty">
      <p><b>No runs yet.</b> Fly one in a terminal from the repository folder and it appears here within two seconds:</p>
      <code>cf-env\Scripts\python scripts\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"</code>
      <p>Add <b>--live</b> to also watch it in Rerun while it flies.</p>
    </div>
    <iframe id="frame" title="Flight report" hidden></iframe>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d = 2) => (v === null || v === undefined) ? "–" : Number(v).toFixed(d);
let runs = [], known = null, selected = null;

const THEMES = ["auto", "light", "dark"];
let theme = "auto";
try { theme = localStorage.getItem("ddsg-dashboard-theme") || "auto"; } catch (e) {}
try { $("auto").checked = localStorage.getItem("ddsg-dashboard-auto") !== "0"; } catch (e) {}
function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("theme").textContent = "Theme: " + theme;
}
$("theme").onclick = () => { theme = THEMES[(THEMES.indexOf(theme) + 1) % 3];
  try { localStorage.setItem("ddsg-dashboard-theme", theme); } catch (e) {} applyTheme(); };
$("auto").onchange = () => { try { localStorage.setItem("ddsg-dashboard-auto", $("auto").checked ? "1" : "0"); } catch (e) {} };
applyTheme();

function verdict(f) {
  if (f.aborted) return ["✕", "--critical", "Aborted: " + f.aborted];
  if (f.collisions) return ["!", "--serious", f.collisions + " wall contacts"];
  return ["✓", "--good", "Completed"];
}
function when(r) { return r.created ? r.created.replace("T", " ").slice(5, 16) : r.name; }

function render(fresh = new Set()) {
  $("list").innerHTML = runs.map((r) => `<button class="run${r.name === selected ? " sel" : ""}${fresh.has(r.name) ? " new" : ""}" data-name="${esc(r.name)}">
    <div class="top"><span class="when">${esc(when(r))}</span><span style="color:var(--muted)">${esc(r.rooms.join(" → "))}</span></div>
    ${r.text ? `<div class="text">“${esc(r.text)}”</div>` : ""}
    <div class="flights">${r.flights.map((f) => { const [icon, color, text] = verdict(f);
      return `<span class="fl"><span class="icon" style="background:var(${color})" aria-hidden="true">${icon}</span>
        <span>${esc(f.label)}: ${esc(text)}</span><span class="num" title="largest distance from the planned path">max ${fmt(f.max_deviation_m)} m</span></span>`; }).join("")}</div>
  </button>`).join("") || `<p class="empty" style="padding:12px">Waiting for the first run…</p>`;
  for (const el of document.querySelectorAll(".run")) el.onclick = () => select(el.dataset.name);
  $("empty").hidden = runs.length > 0;
}

function select(name) {
  selected = name;
  const r = runs.find((x) => x.name === name);
  $("bar").hidden = $("frame").hidden = !r;
  if (!r) return;
  $("sel-title").textContent = `${when(r)}  ${r.text ? "“" + r.text + "”" : ""}`;
  $("rerun").textContent = r.has_rrd ? "Open recording in Rerun" : "Replay in Rerun";
  $("msg").textContent = "";
  $("frame").src = `/run/${encodeURIComponent(name)}/report?t=${r.mtime}`;
  render();
}

$("newtab").onclick = () => selected && window.open(`/run/${encodeURIComponent(selected)}/report`, "_blank");
$("rerun").onclick = async () => {
  if (!selected) return;
  $("msg").textContent = "Starting Rerun…";
  try {
    const res = await fetch(`/api/rerun/${encodeURIComponent(selected)}`, { method: "POST", headers: { "X-Dashboard": "1" } });
    const data = await res.json();
    $("msg").textContent = data.ok ? "Rerun: " + data.message : "Rerun: " + data.error;
  } catch (e) { $("msg").textContent = "Rerun: " + e; }
};

async function poll() {
  try {
    const res = await fetch("/api/runs", { cache: "no-store" });
    const data = await res.json();
    $("root").textContent = data.root;
    const names = new Set(data.runs.map((r) => r.name));
    const fresh = known ? new Set([...names].filter((n) => !known.has(n))) : new Set();
    const changed = JSON.stringify(data.runs) !== JSON.stringify(runs);
    runs = data.runs; known = names;
    if (fresh.size && $("auto").checked) {
      select(runs.find((r) => fresh.has(r.name)).name);
    } else if (!selected && runs.length) {
      select(runs[0].name);
    } else if (changed) {
      const cur = runs.find((r) => r.name === selected);
      if (selected && !cur) select(runs[0]?.name);
      else if (cur && $("frame").src.indexOf(`t=${cur.mtime}`) < 0) select(cur.name);  // rerun rewrote it
      else render(fresh);
    }
    if (fresh.size) render(fresh);
    $("pulse").classList.remove("off");
    $("status").textContent = "Watching · " + new Date().toLocaleTimeString();
  } catch (e) {
    $("pulse").classList.add("off");
    $("status").textContent = "Dashboard stopped (start it again)";
  }
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
