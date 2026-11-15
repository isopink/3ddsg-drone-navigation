"""Check the Crazyflie's sensors without flying (no arming, no motor commands).

    python scripts\\check_sensors.py              # 3 s on the ground: OK to take off?
    python scripts\\check_sensors.py --shake      # 15 s: pick the drone up, move and tap it
    python scripts\\check_sensors.py --watch      # live values until Ctrl+C

Ground check: Flow deck detected, height sensor alive and low, flow sensor sees the floor,
battery, drone state (locked / crashed / tumbled). Same limits as the pre-flight checks
(flight/safety.py).
With a Multi-ranger deck the ground check and --watch also show the distance to the front,
back, left, right and up in meters ("-" when nothing is in view).
Shake test: the deck connection must survive vibration. Any 0 or error reading, or a jump
of more than 30 cm in 50 ms while you move it by hand, means a loose deck or bad pins.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cflib.crtp  # noqa: E402
from cflib.crazyflie import Crazyflie  # noqa: E402
from cflib.crazyflie.log import LogConfig  # noqa: E402
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: E402
from cflib.crazyflie.syncLogger import SyncLogger  # noqa: E402

from ddsg_nav.flight import safety  # noqa: E402

DEFAULT_URI = "radio://0/80/2M/E7E7E7E7E7"
MIN_SQUAL = 10  # flow quality on a textured floor in light is ~20-50
ERROR_MM = 8000  # the height sensor reports 65535 (or similar) when it fails
VARS = [("range.zrange", "uint16_t"), ("motion.squal", "uint8_t"),
        ("pm.vbat", "float"), ("supervisor.info", "uint16_t")]
RANGE_DIRS = ("front", "back", "left", "right", "up")
# 5 x uint16 = 10 bytes on top of the 9 above: still inside the 26-byte log block limit
RANGE_VARS = [(f"range.{d}", "uint16_t") for d in RANGE_DIRS]


def read(scf, seconds: float, period_ms: int, variables=VARS):
    lg = LogConfig("check", period_ms)
    for name, typ in variables:
        lg.add_variable(name, typ)
    rows, n = [], max(1, int(seconds * 1000 / period_ms))
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            rows.append(data)
            if len(rows) >= n:
                break
    return rows


def deck_found(scf) -> bool:
    try:
        return int(scf.cf.param.get_value("deck.bcFlow2")) == 1
    except Exception:
        return False


def multiranger_found(scf) -> bool:
    try:
        return int(scf.cf.param.get_value("deck.bcMultiranger")) == 1
    except Exception:
        return False


def range_m(mm) -> float | None:
    """Multi-ranger reading in meters; None when nothing is in view (the deck then reports 32767)."""
    return None if mm >= ERROR_MM else mm / 1000.0


def format_ranges(row) -> str:
    parts = []
    for d in RANGE_DIRS:
        m = range_m(row[f"range.{d}"])
        parts.append(f"{d} -" if m is None else f"{d} {m:.2f} m")
    return "   ".join(parts)


def bad_state(info: int) -> list[str]:
    flags = safety.decode_supervisor(int(info))
    return [k for k in ("isLocked", "isCrashed", "isTumbled") if flags.get(k)]


def ground_check(scf) -> int:
    ranger = multiranger_found(scf)
    rows = read(scf, 3.0, 100, VARS + RANGE_VARS if ranger else VARS)
    z = [r["range.zrange"] for r in rows]
    q = [r["motion.squal"] for r in rows]
    vbat = rows[-1]["pm.vbat"]
    state = bad_state(rows[-1]["supervisor.info"])
    problems = []
    deck = deck_found(scf)
    if not deck:
        problems.append("Flow deck not detected: check that it is seated, then power-cycle")
    try:
        safety.check_ground_height(z)
    except safety.SafetyError as e:
        problems.append(str(e))
        if len(set(z)) == 1:
            problems.append("a stuck height sensor repeats one value even when lifted (check with --watch): "
                            "take the battery out for 10 s, power on, and check again. "
                            "If it keeps happening, the Flow deck is failing")
    if min(q) < MIN_SQUAL:
        problems.append(f"flow quality drops to {min(q)} (< {MIN_SQUAL}): more light, textured floor, clean sensor")
    try:
        safety.check_battery(vbat)
    except safety.SafetyError as e:
        problems.append(str(e))
    if state:
        problems.append(f"drone state {state}: power-cycle it on a flat surface")

    print(f"Flow deck   : {'detected' if deck else 'NOT detected'}")
    print(f"Height      : {min(z)}-{max(z)} mm   {z}")
    print(f"Flow quality: {min(q)}-{max(q)}")
    if ranger:
        print(f"Multi-ranger: detected   {format_ranges(rows[-1])}")
    else:
        print("Multi-ranger: not detected (no range readings)")
    print(f"Battery     : {vbat:.2f} V")
    print(f"Drone state : {', '.join(state) if state else 'ok'}")
    print()
    if problems:
        print("NOT READY TO FLY")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: ready to fly (keep a hand near it, no USB cable attached)")
    return 0


def shake_test(scf, seconds: float) -> int:
    print(f"Recording {seconds:.0f} s. Pick the drone up now: move it up and down, shake it, tap the deck.")
    rows = read(scf, seconds, 50)
    z = [r["range.zrange"] for r in rows]
    q = [r["motion.squal"] for r in rows]
    zeros = sum(1 for v in z if v == 0)
    errors = sum(1 for v in z if v >= ERROR_MM)
    jumps = sum(1 for a, b in zip(z, z[1:]) if abs(b - a) > 300 and a < ERROR_MM and b < ERROR_MM)
    valid = [v for v in z if 0 < v < ERROR_MM]
    print(f"Samples     : {len(z)}")
    print(f"Height      : {min(valid) if valid else '-'}-{max(valid) if valid else '-'} mm, "
          f"zeros {zeros}, error values {errors}, jumps > 30 cm {jumps}")
    print(f"Flow quality: min {min(q)}, zeros {q.count(0)}")
    print(f"Every 0.5 s : {z[::10]}")
    print()
    if zeros or errors or jumps or (valid and max(valid) - min(valid) < 100):
        print("FAILED: the deck connection drops out (or the drone was not moved).")
        print("  Check the deck pins and seating, power-cycle, and run this again.")
        return 1
    print("OK: the deck connection survives vibration")
    return 0


def watch(scf) -> int:
    ranger = multiranger_found(scf)
    print("Live values, Ctrl+C to stop")
    if not ranger:
        print("Multi-ranger deck not detected: no range readings")
    lg = LogConfig("watch", 200)
    for name, typ in (VARS + RANGE_VARS if ranger else VARS):
        lg.add_variable(name, typ)
    try:
        with SyncLogger(scf, lg) as logger:
            for _, d, _ in logger:
                state = bad_state(d["supervisor.info"])
                print(f"height {d['range.zrange']:>5} mm   flow {d['motion.squal']:>3}   "
                      f"battery {d['pm.vbat']:.2f} V   state {','.join(state) or 'ok'}"
                      + (f"   {format_ranges(d)}" if ranger else ""))
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=DEFAULT_URI)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--shake", action="store_true", help="15 s vibration test while you handle the drone")
    mode.add_argument("--watch", action="store_true", help="print live values until Ctrl+C")
    ap.add_argument("--seconds", type=float, default=15.0, help="length of the shake test")
    a = ap.parse_args()

    cflib.crtp.init_drivers()
    try:
        with SyncCrazyflie(a.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            time.sleep(0.5)
            if a.shake:
                return shake_test(scf, a.seconds)
            if a.watch:
                return watch(scf)
            return ground_check(scf)
    except Exception as e:
        print(f"Could not connect to {a.uri}: {e}")
        print("Is the Crazyradio plugged in and the drone switched on?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
