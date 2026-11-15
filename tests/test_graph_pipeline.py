"""Graph pipeline (load, validate, plan, simulate) on generated graphs: the fake graph,
a saved folder, and broken variants. No database, no IFC, no building data.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import graph_pipeline  # noqa: E402

from ddsg_nav import pipeline  # noqa: E402
from ddsg_nav.graph.fake import build_fake_graph, build_row_graph  # noqa: E402
from ddsg_nav.graph.io import load_graph, save_graph  # noqa: E402
from ddsg_nav.graph.model import Box, Door, Room, SceneGraph, VoxelGrid, VoxelState  # noqa: E402
from ddsg_nav.graph.source import FakeGraphSource, FileGraphSource  # noqa: E402
from ddsg_nav.validate import validate_graph  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "graph_pipeline.py")
def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


needs_sim = pytest.mark.skipif(not _has_mujoco(), reason='needs pip install -e ".[sim]"')


class GivenGraph:
    """A GraphSource for a graph built in the test."""

    def __init__(self, graph):
        self.graph = graph

    def load(self):
        return self.graph


def without_door(graph, door_id):
    """The same graph with one door missing from the data."""
    g = SceneGraph(voxels=graph.voxels)
    for room in graph.rooms.values():
        g.add_room(room)
    for door in graph.doors.values():
        if door.id != door_id:
            g.add_door(door)
    for box in graph.solids:
        g.add_solid(box)
    return g


# ---- the shared module -----------------------------------------------------

@needs_sim
def test_fake_graph_passes():
    lines = []
    result = pipeline.run_source(FakeGraphSource(), log=lines.append)
    assert result.ok, result.failures
    assert [r.rooms for r in result.routes] == [["A", "B", "C"]]
    assert result.routes[0].flown and result.wall_contacts == 0 and result.out is None
    assert [line[:5] for line in lines] == ["[1/4]", "[2/4]", "[3/4]", "[4/4]"]
    assert "3 rooms, 2 doors" in lines[0] and "0.0% unknown" in lines[0]
    report = pipeline.format_report(result)
    assert "rooms: 3" in report and "doors: 2" in report
    assert "grid: 92 x 32 x 25 voxels of 0.1 m (9.2 x 3.2 x 2.5 m)" in report
    assert "unknown voxels: 0.0%" in report
    assert "rooms that cannot be reached: 0" in report
    assert "wall contacts: 0" in report and report.endswith("RESULT: PASS (0 failures)")


@needs_sim
def test_saved_folder_gives_the_same_result(tmp_path):
    first = pipeline.run_source(FakeGraphSource(obstacle=True), save=tmp_path / "g")
    assert first.ok and first.out == tmp_path / "g"
    saved = load_graph(first.out)  # --save writes a complete graph export
    assert saved.rooms == first.graph.rooms and saved.solids == first.graph.solids
    second = pipeline.run_source(FileGraphSource(first.out))
    assert second.ok, second.failures
    assert [r.rooms for r in second.routes] == [r.rooms for r in first.routes]
    assert second.routes[0].length_m == pytest.approx(first.routes[0].length_m)


@needs_sim
def test_given_pair_by_name():
    result = pipeline.run_source(FakeGraphSource(), start="Room B", goal="A", velocity=0.5)
    assert result.ok, result.failures
    assert [r.rooms for r in result.routes] == [["B", "A"]]


def test_bad_pair_is_explained():
    half = pipeline.run_source(FakeGraphSource(), start="A")
    assert half.failures == ["plan: give both --from and --to, or neither"]
    unknown = pipeline.run_source(FakeGraphSource(), start="A", goal="Z")
    assert "unknown room 'Z'" in unknown.failures[0] and "of this graph: A, B, C" in unknown.failures[0]


def test_load_failure_is_reported(tmp_path):
    missing = pipeline.run_source(FileGraphSource(tmp_path / "none"))
    assert not missing.ok and missing.failures[0].startswith("load: ") and "not found" in missing.failures[0]
    assert missing.graph is None
    assert "RESULT: FAIL (1 failures)" in pipeline.format_report(missing)
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "graph.json").write_text("{}")
    assert pipeline.run_source(FileGraphSource(tmp_path / "bad")).failures[0].startswith("load: ")


def test_validation_errors_stop_before_planning():
    g = build_fake_graph()
    door = g.doors["door_AB"]
    x, y, _ = door.position
    g.voxels.fill_box(Box((x - 0.1, y - 0.5, 0.0), (x + 0.1, y + 0.5, 2.0)), VoxelState.OCCUPIED)
    result = pipeline.run_source(GivenGraph(g))
    assert not result.ok and not result.routes
    assert any(f.startswith("validate: door-blocked: door door_AB") for f in result.failures)
    assert result.unreachable == {"A": "blocked or out of reach: door door_AB"}
    assert "  A: blocked or out of reach: door door_AB" in pipeline.format_report(result)


# ---- rooms that cannot be reached ------------------------------------------

@needs_sim
def test_room_without_a_door_is_listed_and_the_rest_passes():
    g = without_door(build_row_graph([2, 2, 2], 2.0, 1.0), "door_BC")
    result = pipeline.run_source(GivenGraph(g), velocity=0.5)
    assert result.ok, result.failures
    assert [r.rooms for r in result.routes] == [["A", "B"]]
    assert result.doorless == ["C"] and result.unreachable == {"C": "no door"}
    report = pipeline.format_report(result)
    assert "rooms that cannot be reached: 1\n  C: no door" in report
    assert "rooms without a door" not in report  # that line belongs to the IFC report


def test_unreachable_rooms_and_reasons():
    grid = VoxelGrid.empty(origin=(0.0, 0.0, 0.0), size=(12.0, 2.0, 2.0), resolution=0.5, state=VoxelState.FREE)
    g = SceneGraph(voxels=grid)
    for i, rid in enumerate("ABCDEF"):
        g.add_room(Room(rid, f"Room {rid}", Box((2.0 * i, 0.0, 0.0), (2.0 * i + 2.0, 2.0, 2.0))))
    for a, b, x in [("A", "B", 2.0), ("B", "C", 4.0), ("D", "E", 8.0)]:  # C | D is a wall, F has no door
        g.add_door(Door(f"door_{a}{b}", a, b, (x, 1.0, 1.0), 1.0))
    assert pipeline.unreachable_rooms(g) == {
        "D": "connected only to E, not to the main group of 3 rooms",
        "E": "connected only to D, not to the main group of 3 rooms",
        "F": "no door",
    }
    assert pipeline.unreachable_rooms(build_fake_graph()) == {}
    assert pipeline.unreachable_rooms(build_row_graph([2.0])) == {}  # one room: nothing to reach


def test_door_names_are_matched_whole():
    """An error about door_1 must not cut off door_10."""
    grid = VoxelGrid.empty(origin=(0.0, 0.0, 0.0), size=(6.0, 2.0, 2.0), resolution=0.5, state=VoxelState.FREE)
    g = SceneGraph(voxels=grid)
    for i, rid in enumerate("ABC"):
        g.add_room(Room(rid, rid, Box((2.0 * i, 0.0, 0.0), (2.0 * i + 2.0, 2.0, 2.0))))
    g.add_door(Door("door_1", "A", "B", (2.0, 1.0, 1.0), 1.0))
    g.add_door(Door("door_10", "B", "C", (4.0, 1.0, 1.0), 1.0))
    report = validate_graph(g, check_connectivity=False)
    report.add("error", "door-blocked", "door door_1 center is inside an obstacle at 0.5 m")
    assert pipeline.unreachable_rooms(g, report) == {"A": "blocked or out of reach: door door_1"}


# ---- unknown voxels and the policy -----------------------------------------

def unobserved_graph():
    """The fake graph with the room interiors never observed (walls are known)."""
    g = build_fake_graph()
    states = g.voxels.states
    states[states == int(VoxelState.FREE)] = int(VoxelState.UNKNOWN)
    return g


def test_unknown_share():
    g = build_fake_graph()
    assert pipeline.unknown_share(g) == 0.0
    g.voxels.states[:, :, 20:] = int(VoxelState.UNKNOWN)  # top 5 of 25 layers
    assert pipeline.unknown_share(g) == pytest.approx(0.2)
    assert "unknown voxels: 20.0%" in pipeline.format_report(pipeline.PipelineResult(graph=g))


@needs_sim
def test_policy_decides_about_unobserved_rooms():
    blocked = pipeline.run_source(GivenGraph(unobserved_graph()))  # default: unknown is an obstacle
    assert not blocked.ok and all(f.startswith("validate: ") for f in blocked.failures)
    assert any("mostly-unknown" in line for line in pipeline.format_report(blocked).splitlines())
    trusted = pipeline.run_source(GivenGraph(unobserved_graph()), policy="free_in_rooms", velocity=0.5)
    assert trusted.ok, trusted.failures
    assert [r.rooms for r in trusted.routes] == [["A", "B", "C"]] and trusted.wall_contacts == 0


# ---- command line ----------------------------------------------------------

@needs_sim
def test_command_line_fake_then_saved_folder(tmp_path):
    def run(*args):
        return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=tmp_path)

    fake = run("--fake", "--save", "out")
    assert fake.returncode == 0, fake.stdout + fake.stderr
    assert "Graph pipeline: built-in fake graph" in fake.stdout and "RESULT: PASS" in fake.stdout
    assert (tmp_path / "out" / "graph.json").is_file() and (tmp_path / "out" / "voxels.npz").is_file()

    folder = run("out", "--from", "C", "--to", "A", "--height", "0.8", "--margin", "0.2",
                 "--policy", "free_in_rooms", "--velocity", "0.5")
    assert folder.returncode == 0, folder.stdout + folder.stderr
    assert "Graph pipeline: out" in folder.stdout
    assert "1 routes at 0.8 m, 0.2 m from walls" in folder.stdout
    assert "  C > B > A: " in folder.stdout and "wall contacts: 0" in folder.stdout
    assert list(tmp_path.iterdir()) == [tmp_path / "out"]  # without --save nothing is written

    bad = run("out", "--from", "A", "--to", "Z")
    assert bad.returncode == 1 and "FAILED route A -> Z: plan: unknown room 'Z'" in bad.stdout
    assert "RESULT: FAIL (1 failures)" in bad.stdout


@needs_sim
def test_routes_option(tmp_path, capsys):
    save_graph(build_row_graph([1.5, 1.5, 1.5, 1.5], 1.5, 0.8), tmp_path / "row")
    assert graph_pipeline.main([str(tmp_path / "row"), "--routes", "2", "--velocity", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "[3/4] plan      2 routes" in out and "routes flown: 2" in out


def test_command_line_missing_folder(tmp_path, capsys):
    assert graph_pipeline.main([str(tmp_path / "none")]) == 1
    out = capsys.readouterr().out
    assert "FAILED load: " in out and "RESULT: FAIL" in out


@pytest.mark.parametrize("argv", [[], ["folder", "--fake"], ["--neo4j", "bolt://db"], ["--floor", "4F"],
                                  ["folder", "--neo4j", "bolt://db", "--floor", "4F"],
                                  ["--fake", "--policy", "maybe"]])
def test_command_line_rejects_unclear_input(argv, capsys):
    with pytest.raises(SystemExit) as e:
        graph_pipeline.main(argv)
    assert e.value.code == 2
    capsys.readouterr()


def test_neo4j_credentials_come_from_the_environment(monkeypatch, capsys):
    monkeypatch.delenv("DDSG_NEO4J_USER", raising=False)
    monkeypatch.delenv("DDSG_NEO4J_PASSWORD", raising=False)
    assert graph_pipeline.main(["--neo4j", "bolt://db:7687", "--floor", "4F"]) == 1
    assert "set DDSG_NEO4J_USER and DDSG_NEO4J_PASSWORD" in capsys.readouterr().out

    seen = {}

    class FakeNeo4j:
        def __init__(self, uri, user, password, floor):
            seen.update(uri=uri, user=user, password=password, floor=floor)

        def load(self):
            raise ConnectionError("database is not reachable")

    import ddsg_nav.graph.neo4j_source as neo4j_source

    monkeypatch.setattr(neo4j_source, "Neo4jGraphSource", FakeNeo4j)
    monkeypatch.setenv("DDSG_NEO4J_USER", "reader")
    monkeypatch.setenv("DDSG_NEO4J_PASSWORD", "s3cret-pw")
    assert graph_pipeline.main(["--neo4j", "bolt://db:7687", "--floor", "4F"]) == 1
    out = capsys.readouterr().out
    assert seen == {"uri": "bolt://db:7687", "user": "reader", "password": "s3cret-pw", "floor": "4F"}
    assert "Graph pipeline: Neo4j bolt://db:7687, floor 4F" in out
    assert "FAILED load: database is not reachable" in out
    assert "s3cret-pw" not in out
    with pytest.raises(SystemExit):  # there is no option to pass them on the command line
        graph_pipeline.main(["--neo4j", "bolt://db", "--floor", "4F", "--password", "x"])
    capsys.readouterr()


@needs_sim
def test_neo4j_graph_runs_like_any_other(monkeypatch, capsys, tmp_path):
    """The day the adapter returns a graph, the same command checks and saves it."""
    import ddsg_nav.graph.neo4j_source as neo4j_source

    class FakeNeo4j:
        def __init__(self, uri, user, password, floor):
            pass

        def load(self):
            return build_fake_graph()

    monkeypatch.setattr(neo4j_source, "Neo4jGraphSource", FakeNeo4j)
    monkeypatch.setenv("DDSG_NEO4J_USER", "reader")
    monkeypatch.setenv("DDSG_NEO4J_PASSWORD", "pw")
    code = graph_pipeline.main(["--neo4j", "bolt://db", "--floor", "4F", "--save", str(tmp_path / "4F"),
                                "--velocity", "0.5"])
    assert code == 0 and "RESULT: PASS" in capsys.readouterr().out
    assert load_graph(tmp_path / "4F").rooms.keys() == {"A", "B", "C"}


# ---- the IFC command uses the same steps -----------------------------------

def test_ifc_pipeline_shares_the_module():
    pytest.importorskip("ifcopenshell")
    import ifc_pipeline

    assert ifc_pipeline.pick_pairs is pipeline.pick_pairs
    assert ifc_pipeline.PipelineResult is pipeline.PipelineResult
    result = pipeline.PipelineResult(graph=build_fake_graph(), doorless=["X"])
    report = ifc_pipeline.format_report(result)
    assert "rooms without a door: X. They cannot be reached." in report and "--exclude" in report
    assert "grid:" not in report and "unknown voxels" not in report  # its report is unchanged
