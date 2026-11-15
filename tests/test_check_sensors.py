"""scripts/check_sensors.py with a fake cflib: no radio, no drone."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cflib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_sensors.py"
BASE = {"range.zrange": 20, "motion.squal": 40, "pm.vbat": 4.0, "supervisor.info": 0}
RANGES = {"range.front": 1234, "range.back": 250, "range.left": 32767, "range.right": 3999, "range.up": 8000}
RANGE_TEXT = "front 1.23 m   back 0.25 m   left -   right 4.00 m   up -"
DECK = {"deck.bcMultiranger": "1"}


class FakeLogConfig:
    made = []

    def __init__(self, name, period_in_ms):
        self.names = []
        FakeLogConfig.made.append(self)

    def add_variable(self, name, fetch_as=None):
        self.names.append(name)


class FakeParam:
    def __init__(self, values):
        self.values = values

    def get_value(self, name):
        return self.values[name]  # KeyError for an unknown parameter, like cflib


def _load(monkeypatch, row, rows=1000):
    spec = importlib.util.spec_from_file_location("check_sensors", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeSyncLogger:
        def __init__(self, scf, lg):
            self.lg = lg

        def __enter__(self):
            # like the drone: only the variables in the block come back
            def sample(i):
                d = {k: row[k] for k in self.lg.names}
                d["range.zrange"] += i % 2  # a live height sensor jitters
                return d

            return iter([(i, sample(i), self.lg) for i in range(rows)])

        def __exit__(self, *exc):
            return False

    FakeLogConfig.made = []
    monkeypatch.setattr(mod, "LogConfig", FakeLogConfig)
    monkeypatch.setattr(mod, "SyncLogger", FakeSyncLogger)
    return mod


def _scf(**params):
    return SimpleNamespace(cf=SimpleNamespace(param=FakeParam({"deck.bcFlow2": "1", **params})))


def test_unit_conversion_and_out_of_range(monkeypatch):
    mod = _load(monkeypatch, BASE)
    assert mod.range_m(1234) == pytest.approx(1.234)
    assert mod.range_m(0) == 0.0
    assert mod.range_m(7999) == pytest.approx(7.999)
    assert mod.range_m(8000) is None and mod.range_m(32767) is None and mod.range_m(65535) is None
    assert mod.format_ranges(RANGES) == RANGE_TEXT


def test_ground_check_with_the_deck(monkeypatch, capsys):
    mod = _load(monkeypatch, {**BASE, **RANGES})
    assert mod.ground_check(_scf(**DECK)) == 0
    assert f"Multi-ranger: detected   {RANGE_TEXT}" in capsys.readouterr().out
    assert FakeLogConfig.made[0].names == list(BASE) + list(RANGES)


@pytest.mark.parametrize("params", [{}, {"deck.bcMultiranger": "0"}])
def test_ground_check_without_the_deck(monkeypatch, capsys, params):
    mod = _load(monkeypatch, BASE)  # asking for a range variable would raise KeyError
    assert mod.ground_check(_scf(**params)) == 0  # the deck is optional: still ready to fly
    out = capsys.readouterr().out
    assert "Multi-ranger: not detected" in out and "front" not in out
    assert "OK: ready to fly" in out
    assert FakeLogConfig.made[0].names == list(BASE)  # log block as before


def test_watch_with_and_without_the_deck(monkeypatch, capsys):
    mod = _load(monkeypatch, {**BASE, **RANGES}, rows=2)
    assert mod.watch(_scf(**DECK)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and lines[1].endswith(RANGE_TEXT)

    mod = _load(monkeypatch, BASE, rows=2)
    assert mod.watch(_scf()) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "Multi-ranger deck not detected: no range readings"
    assert lines[2] == "height    20 mm   flow  40   battery 4.00 V   state ok"  # line as before


def test_shake_test_is_unchanged(monkeypatch):
    mod = _load(monkeypatch, BASE, rows=10)
    mod.shake_test(_scf(**DECK), 0.5)
    assert FakeLogConfig.made[0].names == list(BASE)
