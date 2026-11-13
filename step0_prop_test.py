"""Step 0: propeller test (no take-off).

Triggers the firmware's built-in propeller test (param health.startPropTest):
each motor spins briefly in turn at low power, then stops.

Before running:
  - Drone on a flat surface, nothing touching the propellers, hands clear.
  - Battery charged.

Run:
    cf-env\\Scripts\\python step0_prop_test.py [--uri radio://0/80/2M]
"""
import argparse
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

DEFAULT_URI = "radio://0/80/2M"


def read_log(scf, names, period_ms=100):
    lg = LogConfig(name="read", period_in_ms=period_ms)
    available = scf.cf.log.toc.toc
    for n in names:
        g, v = n.split(".")
        if g in available and v in available[g]:
            lg.add_variable(n)
    if not lg.variables:
        return {}
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            return data


def main(uri: str) -> None:
    cflib.crtp.init_drivers()
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        print("Battery:", read_log(scf, ["pm.vbat"]))
        print("Starting propeller test: motors M1..M4 spin one by one...")
        scf.cf.param.set_value("health.startPropTest", 1)
        time.sleep(8)  # test takes a few seconds
        print("Result:", read_log(scf, ["health.motorPass", "health.motorTestCount"]))
        print("motorPass is a bitmask: 15 (0b1111) = all 4 motors passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=DEFAULT_URI)
    main(ap.parse_args().uri)
