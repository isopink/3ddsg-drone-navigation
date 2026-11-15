"""Flight records, the HTML flight report and the Rerun live view (no drone needed)."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ddsg_nav.flight.backend import DroneBackend, Status
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.flight.monitor import FlightAbort, FlightMonitor
from ddsg_nav.flight.record import FlightRecorder, FlightRun, load_run, new_run_folder, plan_dict, save_run
from ddsg_nav.graph.fake import build_row_graph
from ddsg_nav.planner.route import plan_route
from ddsg_nav.report import deviations, render_report, report_data, write_report

ROOT = Path(__file__).resolve().parents[1]


class TrackBackend(DroneBackend):
    """Teleports to each target and records a trajectory on its own clock."""

    def __init__(self, battery=(4.1,), drift=0.0):
        self.t = 0.0
        self.pos = (0.0, 0.0, 0.0)
        self.trajectory = [(0.0, *self.pos)]
        self.battery = list(battery)
        self.drift = drift
        self.landed = 0

    def clock(self):
        return self.t

    def _move(self, p):
        self.t += 1.0
        self.pos = p
        self.trajectory.append((self.t, *p))

    def takeoff(self, height):
        self._move((self.pos[0], self.pos[1], height))

    def go_to(self, x, y, z, velocity):
        self._move((x, y + self.drift, z))

    def position(self):
        return self.pos

    def status(self):
        v = self.battery.pop(0) if len(self.battery) > 1 else self.battery[0]
        return Status(battery_v=v, flags={}, position=self.pos)

    def land(self):
        self.landed += 1
        self._move((self.pos[0], self.pos[1], 0.0))

    def close(self):
        pass


WPS = [(0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (2.0, 0.0, 0.5)]


def test_recorder_collects_status_and_events():
    b = TrackBackend()
    seen = []
    rec = FlightRecorder("test", "fake", clock=b.clock, listeners=[lambda k, r: seen.append(k)])
    follow_path(b, WPS, recorder=rec)
    kinds = [e["kind"] for e in rec.log.events]
    assert kinds == ["takeoff", "waypoint", "waypoint", "waypoint", "land", "landed"]
    assert [e["index"] for e in rec.log.events if e["kind"] == "waypoint"] == [0, 1, 2]
    assert len(rec.log.status) == 4  # after take-off and after each waypoint
    assert rec.log.status[-1]["position"] == [2.0, 0.0, 0.5]
    # status and events use the backend clock, so they line up with the trajectory
    assert [s["t"] for s in rec.log.status] == [1.0, 2.0, 3.0, 4.0]
    assert seen.count("status") == 4 and seen.count("event") == 6
    log = rec.finish(b.trajectory, WPS)
    assert log.ok and log.comparison["max_deviation_m"] == pytest.approx(0.0)


def test_recorder_records_abort_and_still_lands():
    b = TrackBackend(battery=(4.1, 4.0, 3.2))
    rec = FlightRecorder("test", "fake", clock=b.clock)
    with pytest.raises(FlightAbort):
        follow_path(b, WPS, monitor=FlightMonitor(), recorder=rec)
    kinds = [e["kind"] for e in rec.log.events]
    assert "abort" in kinds and kinds[-2:] == ["land", "landed"]
    assert "battery" in next(e for e in rec.log.events if e["kind"] == "abort")["message"]
    assert b.landed == 1


def test_recorder_records_errors():
    class Broken(TrackBackend):
        def go_to(self, *a):
            raise RuntimeError("radio lost")

    b = Broken()
    rec = FlightRecorder("test", "fake", clock=b.clock)
    with pytest.raises(RuntimeError):
        follow_path(b, WPS, recorder=rec)
    err = next(e for e in rec.log.events if e["kind"] == "error")
    assert "radio lost" in err["message"] and b.landed == 1


def test_broken_listener_does_not_stop_the_flight(capsys):
    def bad(kind, record):
        raise ValueError("viewer gone")

    b = TrackBackend()
    follow_path(b, WPS, recorder=FlightRecorder("t", "fake", clock=b.clock, listeners=[bad]))
    assert b.landed == 1 and "viewer gone" in capsys.readouterr().out


def make_run(tmp_path, drift=0.0, aborted=None):
    graph = build_row_graph([2.0, 2.0], 2.0, 1.0)
    plan = plan_route(graph, "A", "B", height=0.5)
    flights = []
    for label, kind, d in (("simulation (dry run)", "sim", 0.0), ("real flight", "crazyflie", drift)):
        b = TrackBackend(battery=(4.1, 4.0, 3.9, 3.8, 3.7, 3.6), drift=d)
        b.pos = (plan.waypoints[0][0], plan.waypoints[0][1], 0.0)
        rec = FlightRecorder(label, kind, clock=b.clock)
        follow_path(b, plan.waypoints, recorder=rec)
        flights.append(rec.finish(b.trajectory, plan.waypoints, aborted if kind != "sim" else None, 0 if kind == "sim" else None))
    run = FlightRun(graph=graph, plan=plan_dict(plan), flights=flights, meta={"text": "A -> B", "velocity": 0.2})
    return save_run(run, tmp_path / "run")


def test_run_folder_round_trip(tmp_path):
    folder = make_run(tmp_path, drift=0.1)
    assert {p.name for p in folder.iterdir()} >= {"run.json", "graph", "flight_1.csv", "flight_2.csv"}
    run = load_run(folder)
    assert [f.label for f in run.flights] == ["simulation (dry run)", "real flight"]
    assert run.meta["text"] == "A -> B" and "created" in run.meta
    assert run.flights[1].comparison["max_deviation_m"] == pytest.approx(0.1, abs=1e-6)
    assert len(run.flights[1].trajectory) == len(run.flights[1].trajectory)
    assert run.graph.rooms.keys() == {"A", "B"}
    assert run.waypoints[0] == tuple(run.plan["waypoints"][0])


def test_load_run_rejects_other_folders(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_run(tmp_path)


def test_new_run_folder_never_reuses_a_folder(tmp_path):
    a = new_run_folder(tmp_path, "20260101_000000")
    a.mkdir(parents=True)
    b = new_run_folder(tmp_path, "20260101_000000")
    assert a != b and b.name == "20260101_000000_2"


def test_deviations_matches_scalar_version():
    from ddsg_nav.flight.compare import distance_to_polyline

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 3, (50, 3))
    line = np.array([(0, 0, 0.5), (1, 0, 0.5), (1, 2, 0.5), (1, 2, 0.5)])  # incl. a zero-length segment
    expected = [distance_to_polyline(tuple(p), [tuple(q) for q in line]) for p in pts]
    assert np.allclose(deviations(pts, line), expected)
    assert np.allclose(deviations(pts, line[:1]), np.linalg.norm(pts - line[0], axis=1))


def test_report_data(tmp_path):
    run = load_run(make_run(tmp_path, drift=0.1, aborted="battery 3.40 V is below 3.5 V"))
    d = report_data(run)
    sim, real = d["flights"]
    assert sim["ok"] and not real["ok"] and real["abort_point"] is not None
    assert sim["battery_real"] is False and real["battery_real"] is True
    assert real["min_battery_v"] == min(s["battery_v"] for s in run.flights[1].status) < 4.1
    # take-off and landing samples are gaps in the deviation chart
    assert real["deviation"][0] is None and real["deviation"][-1] is None
    assert max(v for v in real["deviation"] if v is not None) == pytest.approx(0.1, abs=1e-3)
    assert len(d["map"]["states"]) == len(d["map"]["y"]) and len(d["map"]["states"][0]) == len(d["map"]["x"])
    assert len(d["voxels3d"]["x"]) > 0 and max(d["voxels3d"]["z"]) <= d["voxels3d"]["cut_z"]
    json.dumps(d, allow_nan=False)  # valid JSON for the page: no NaN


def test_report_html(tmp_path):
    folder = make_run(tmp_path)
    out = write_report(folder)
    text = out.read_text(encoding="utf-8")
    assert out.name == "report.html" and text.startswith("<!doctype html>")
    assert "cdnjs.cloudflare.com/ajax/libs/plotly.js" in text
    assert "__DATA__" not in text and "real flight" in text
    # a label cannot close the script tag
    run = load_run(folder)
    run.flights[0].label = "</script><b>x"
    assert "</script><b>x" not in render_report(run)


def test_flight_report_script(tmp_path):
    folder = make_run(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "flight_report.py"), str(tmp_path), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert (folder / "report.html").is_file()


def test_fly_route_sim_saves_run_and_report(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fly_route.py"), "--backend", "sim", "--rooms", "2", "2",
         "--runs", str(tmp_path), "A -> B"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = load_run(folder)
    assert len(run.flights) == 1 and run.flights[0].ok and run.flights[0].backend == "sim"
    assert [e["kind"] for e in run.flights[0].events][0] == "takeoff"
    assert (folder / "report.html").is_file()


def test_live_view_replay_to_file(tmp_path):
    pytest.importorskip("rerun")
    from ddsg_nav.viz.live import LiveView, replay_run, slug

    assert slug("simulation (dry run)") == "simulation_dry_run"
    run = load_run(make_run(tmp_path, drift=0.1, aborted="stopped"))
    rrd = tmp_path / "replay.rrd"
    view = LiveView(spawn=False, save=rrd)
    replay_run(run, view)
    import rerun as rr

    rr.disconnect()
    assert rrd.is_file() and rrd.stat().st_size > 1000


def test_live_view_follows_a_flight(tmp_path):
    pytest.importorskip("rerun")
    from ddsg_nav.viz.live import LiveView

    graph = build_row_graph([2.0, 2.0], 2.0, 1.0)
    plan = plan_route(graph, "A", "B", height=0.5)
    view = LiveView(spawn=False, save=tmp_path / "live.rrd")
    view.log_scene(graph, plan)
    b = TrackBackend()
    rec = FlightRecorder("sim", "sim", clock=b.clock, listeners=[view.listener("sim")])
    with view.follow("sim", b, period_s=0.01):
        follow_path(b, plan.waypoints, recorder=rec)
    import rerun as rr

    rr.disconnect()
    assert (tmp_path / "live.rrd").stat().st_size > 1000
    assert math.isclose(rec.log.events[-1]["t"], b.t)


# ---- 20 Hz telemetry (battery and drone state) -------------------------------

TUMBLED = 1 << 5  # supervisor.info bit of isTumbled (flight/safety.py)


class TelemetryBackend(TrackBackend):
    """Like the real Crazyflie: battery and state logged with every position sample."""

    def __init__(self, tumble_at=None, **kw):
        super().__init__(**kw)
        self.telemetry = []
        self.tumble_at = tumble_at

    def _move(self, p):
        super()._move(p)
        info = TUMBLED if self.tumble_at is not None and self.t >= self.tumble_at else 0
        self.telemetry.append((self.t, 4.1 - 0.05 * self.t, info))


def make_telemetry_run(tmp_path, tumble_at=None):
    graph = build_row_graph([2.0, 2.0], 2.0, 1.0)
    plan = plan_route(graph, "A", "B", height=0.5)
    b = TelemetryBackend(tumble_at=tumble_at)
    rec = FlightRecorder("real flight", "crazyflie", clock=b.clock)
    follow_path(b, plan.waypoints, recorder=rec)
    log = rec.finish(b.trajectory, plan.waypoints, telemetry=b.telemetry)
    folder = save_run(FlightRun(graph=graph, plan=plan_dict(plan), flights=[log]), tmp_path / "run")
    return folder, b


def test_telemetry_round_trip(tmp_path):
    folder, b = make_telemetry_run(tmp_path)
    assert (folder / "flight_1_telemetry.csv").is_file()
    run = load_run(folder)
    assert run.flights[0].telemetry == [(float(t), float(v), int(s)) for t, v, s in b.telemetry]


def test_runs_without_telemetry_have_no_telemetry_file(tmp_path):
    folder = make_run(tmp_path)
    assert not list(folder.glob("*_telemetry.csv"))
    assert load_run(folder).flights[0].telemetry == []


def test_report_uses_telemetry_for_battery_and_state(tmp_path):
    folder, b = make_telemetry_run(tmp_path, tumble_at=3.0)
    (f,) = report_data(load_run(folder))["flights"]
    assert f["battery"]["source"] == "telemetry"
    assert len(f["battery"]["t"]) == len(b.telemetry) > len(load_run(folder).flights[0].status)
    assert f["min_battery_v"] == pytest.approx(min(v for _, v, _ in b.telemetry))
    states = [e for e in f["events"] if e["kind"] == "state"]
    assert [e["message"] for e in states] == ["isTumbled"] and states[0]["t"] == 3.0
    assert [e["t"] for e in f["events"]] == sorted(e["t"] for e in f["events"])


def test_report_falls_back_to_status_battery(tmp_path):
    (_, real) = report_data(load_run(make_run(tmp_path)))["flights"]
    assert real["battery"]["source"] == "status" and len(real["battery"]["v"]) == 4


def test_live_view_streams_telemetry(tmp_path):
    pytest.importorskip("rerun")
    from ddsg_nav.viz.live import LiveView

    graph = build_row_graph([2.0, 2.0], 2.0, 1.0)
    plan = plan_route(graph, "A", "B", height=0.5)
    view = LiveView(spawn=False, save=tmp_path / "live.rrd")
    view.log_scene(graph, plan)
    b = TelemetryBackend(tumble_at=2.0)
    with view.follow("real flight", b, period_s=0.01):
        follow_path(b, plan.waypoints)
    assert view._bad["real flight"] == ["isTumbled"]  # the state change reached the viewer
    import rerun as rr

    rr.disconnect()
    assert (tmp_path / "live.rrd").stat().st_size > 1000


def test_crazyflie_logs_battery_and_state_with_the_position(monkeypatch):
    """No radio: a fake log block checks the variables, the block size and the callback."""
    pytest.importorskip("cflib")
    from ddsg_nav.flight.backends import crazyflie as cfmod
    from ddsg_nav.graph.transform import Transform

    sizes = {"float": 4, "uint16_t": 2}

    class FakeLog:
        def __init__(self, name, period_in_ms):
            self.vars, self.callbacks = [], []
            self.data_received_cb = type("C", (), {"add_callback": lambda _s, cb: self.callbacks.append(cb)})()

        def add_variable(self, name, typ):
            self.vars.append((name, typ))

        def start(self):
            pass

    class FakeScf:
        cf = type("CF", (), {"log": type("L", (), {"add_config": lambda _s, lg: None})()})()

    monkeypatch.setattr(cfmod, "LogConfig", FakeLog)
    drone = cfmod.CrazyflieBackend(frame=Transform((1.0, 2.0, 0.0), 0.0))
    drone._scf = FakeScf()
    drone._start_trajectory_log()
    lg = drone._traj_log
    names = [n for n, _ in lg.vars]
    assert {"pm.vbat", "supervisor.info", "stateEstimate.x"} <= set(names)
    assert sum(sizes[t] for _, t in lg.vars) <= 26  # Crazyflie log block payload limit
    lg.callbacks[0](0, {"stateEstimate.x": 0.5, "stateEstimate.y": 0.0, "stateEstimate.z": 0.4,
                        "pm.vbat": 3.95, "supervisor.info": TUMBLED}, lg)
    assert drone.trajectory[0][1:] == pytest.approx((1.5, 2.0, 0.4))
    t, vbat, info = drone.telemetry[0]
    assert t == drone.trajectory[0][0] and vbat == pytest.approx(3.95) and info == TUMBLED
    drone._scf = None  # nothing to close
