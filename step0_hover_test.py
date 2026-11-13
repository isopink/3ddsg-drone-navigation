"""Step 0: Crazyflie 2.1 + Flow deck basic flight check.

Sequence: take off -> hover -> move forward 1 m -> hover -> land.

Run (drone on flat floor, >=2 m clear in front, Crazyradio plugged in):
    cf-env\\Scripts\\python step0_hover_test.py [--uri radio://0/80/2M/E7E7E7E7E7]

Safety: keep a hand near the drone / battery. Ctrl+C lands immediately.
"""
import argparse
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

DEFAULT_URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

HOVER_HEIGHT = 0.4   # m  (Flow deck is reliable around 0.3-0.5 m)
MOVE_DISTANCE = 1.0  # m
VELOCITY = 0.2       # m/s (slow on purpose)
MIN_BATTERY_V = 3.7  # refuse to fly below this


def check_flow_deck(scf: SyncCrazyflie) -> bool:
    """Return True if the Flow v2 deck is detected."""
    result = {}

    def cb(_, value_name, value):
        result["flow"] = int(value)

    scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=cb)
    time.sleep(1.0)
    return result.get("flow", 0) == 1


def read_battery(scf: SyncCrazyflie) -> float:
    lg = LogConfig(name="Battery", period_in_ms=100)
    lg.add_variable("pm.vbat", "float")
    with cflib.crazyflie.syncLogger.SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            return data["pm.vbat"]


def main(uri: str) -> None:
    cflib.crtp.init_drivers()
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        if not check_flow_deck(scf):
            raise SystemExit("Flow v2 deck not detected. Check seating / firmware.")
        vbat = read_battery(scf)
        print(f"Battery: {vbat:.2f} V")
        if vbat < MIN_BATTERY_V:
            raise SystemExit(f"Battery too low (< {MIN_BATTERY_V} V). Charge first.")

        print("Take off...")
        with MotionCommander(scf, default_height=HOVER_HEIGHT) as mc:
            time.sleep(2)
            print(f"Forward {MOVE_DISTANCE} m")
            mc.forward(MOVE_DISTANCE, velocity=VELOCITY)
            time.sleep(2)
            print("Landing...")
        print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=DEFAULT_URI)
    main(ap.parse_args().uri)
