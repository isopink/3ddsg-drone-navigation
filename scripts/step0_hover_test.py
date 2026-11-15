"""Step 0: Crazyflie 2.1 + Flow deck basic flight check.

Sequence: take off -> hover -> move forward 1 m -> hover -> land.

Run (drone on flat floor, >=2 m clear in front, Crazyradio plugged in):
    cf-env\\Scripts\\python step0_hover_test.py [--uri radio://0/80/2M/E7E7E7E7E7]

Add --live to print motor output and sensor values while it flies.

Safety: keep a hand near the drone / battery. Ctrl+C lands immediately.
"""
import argparse
import threading
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper

DEFAULT_URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")

HOVER_HEIGHT = 0.4   # m  (Flow deck is reliable around 0.3-0.5 m)
MOVE_DISTANCE = 1.0  # m
VELOCITY = 0.2       # m/s (slow on purpose)
MIN_BATTERY_V = 3.7  # default: refuse to fly below this (change with --min-battery)
BATTERY_FLOOR_V = 3.3  # never below this: the LiPo is empty and sags under 3 V in flight
MIN_RECOMMENDED_HEIGHT = 0.2  # m, Flow deck v2 documented minimum
MAX_TEST_HEIGHT = 2.0         # m, safety cap for this test script (Flow deck v2 range: up to 3 m)


def check_flow_deck(scf: SyncCrazyflie, timeout: float = 5.0) -> bool:
    """Return True if the Flow v2 deck is detected (param deck.bcFlow2 == 1).

    Same approach as the Bitcraze Motion Commander step-by-step guide.
    """
    attached = Event()

    def cb(_, value):
        if int(value):
            attached.set()

    scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=cb)
    return attached.wait(timeout=timeout)


def read_battery(scf: SyncCrazyflie) -> float:
    lg = LogConfig(name="Battery", period_in_ms=100)
    lg.add_variable("pm.vbat", "float")
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            return data["pm.vbat"]


SUPERVISOR_BITS = ["canBeArmed", "isArmed", "autoArm", "canFly", "isFlying",
                   "isTumbled", "isLocked", "isCrashed"]


def read_supervisor(scf: SyncCrazyflie) -> dict:
    """Decode supervisor.info into {flag: bool}. Empty dict if unavailable."""
    lg = LogConfig(name="Supervisor", period_in_ms=100)
    toc = scf.cf.log.toc.toc
    if "supervisor" not in toc or "info" not in toc["supervisor"]:
        return {}
    lg.add_variable("supervisor.info", "uint16_t")
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            v = int(data["supervisor.info"])
            return {b: bool(v >> i & 1) for i, b in enumerate(SUPERVISOR_BITS)}


FLIGHT_LOG_VARS = [("range.zrange", "uint16_t"), ("stateEstimate.z", "float"),
                   ("motor.m1", "uint32_t"), ("motor.m2", "uint32_t"),
                   ("motor.m3", "uint32_t"), ("motor.m4", "uint32_t")]


XY_LOG_VARS = [("stateEstimate.x", "float"), ("stateEstimate.y", "float"),
               ("stabilizer.yaw", "float"), ("pm.vbat", "float")]


def live_printer(samples: list, xy: list, stop: Event, period_s: float = 0.25) -> threading.Thread:
    """Print the latest motor output and sensor values while flying (--live)."""
    t0 = time.monotonic()

    def run():
        print("   t s | height mm | est z m | motors % m1  m2  m3  m4 |   x m    y m | battery V")
        while not stop.wait(period_s):
            if not samples:
                continue
            s, p = samples[-1], (xy[-1] if xy else {})
            m = [s.get(f"motor.m{i}", 0) * 100 / 65535 for i in (1, 2, 3, 4)]
            warn = "  <- motors near max" if max(m) > 90 else ""
            print(f"{time.monotonic() - t0:6.1f} | {s.get('range.zrange', 0):9d} | {s.get('stateEstimate.z', 0):7.2f} |"
                  f"          {m[0]:3.0f} {m[1]:3.0f} {m[2]:3.0f} {m[3]:3.0f} |"
                  f" {p.get('stateEstimate.x', 0):6.2f} {p.get('stateEstimate.y', 0):6.2f} |"
                  f" {p.get('pm.vbat', 0):9.2f}{warn}", flush=True)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _start_log(scf: SyncCrazyflie, name: str, variables: list, samples: list, configs: list) -> None:
    toc = scf.cf.log.toc.toc
    lg = LogConfig(name=name, period_in_ms=50)
    for var, typ in variables:
        g, n = var.split(".")
        if g in toc and n in toc[g]:
            lg.add_variable(var, typ)
    if not lg.variables:
        return
    lg.data_received_cb.add_callback(lambda ts, data, _: samples.append(dict(data)))
    scf.cf.log.add_config(lg)
    lg.start()
    configs.append(lg)


def start_flight_log(scf: SyncCrazyflie) -> tuple:
    """Record height/motors and x,y,yaw during the flight (two async 20 Hz logs).

    Returns (samples, xy, configs). Call stop_flight_log(configs) afterwards,
    otherwise the Crazyflie keeps the log blocks and the next connection prints
    'Error no LogEntry to handle id=N'.
    """
    samples: list = []
    xy: list = []
    configs: list = []
    _start_log(scf, "flight", FLIGHT_LOG_VARS, samples, configs)
    _start_log(scf, "flightxy", XY_LOG_VARS, xy, configs)
    return samples, xy, configs


def stop_flight_log(configs: list) -> None:
    for lg in configs:
        try:
            lg.stop()
            lg.delete()
        except Exception as e:  # never let cleanup break the landing
            print(f"log cleanup warning: {e}")


def summarize_xy(xy: list) -> None:
    pts = [(s["stateEstimate.x"], s["stateEstimate.y"]) for s in xy
           if "stateEstimate.x" in s and "stateEstimate.y" in s]
    if not pts:
        print("XY log: no samples received.")
        return
    x0, y0 = pts[0]
    far = max(pts, key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2)
    dx, dy = far[0] - x0, far[1] - y0
    yaws = [s["stabilizer.yaw"] for s in xy if "stabilizer.yaw" in s]
    print(f"XY log: {len(pts)} samples. Estimated max displacement: "
          f"forward-ish dx={dx:.2f} m, dy={dy:.2f} m  (|d|={(dx*dx+dy*dy)**0.5:.2f} m)")
    print(f"  end position estimate: dx={pts[-1][0]-x0:.2f} m, dy={pts[-1][1]-y0:.2f} m")
    if yaws:
        print(f"  yaw start {yaws[0]:.1f} deg, end {yaws[-1]:.1f} deg")


def summarize_flight_log(samples: list) -> None:
    if not samples:
        print("Flight log: no samples received.")
        return
    def col(key):
        return [s[key] for s in samples if key in s]
    z_mm = col("range.zrange")
    est = col("stateEstimate.z")
    motors = [max(col(f"motor.m{i}") or [0]) for i in (1, 2, 3, 4)]
    print(f"Flight log: {len(samples)} samples")
    if z_mm:
        print(f"  range.zrange   min {min(z_mm)} mm  max {max(z_mm)} mm")
    if est:
        print(f"  stateEstimate.z max {max(est) * 1000:.0f} mm")
    print(f"  motor max (m1..m4): {motors}  (0 = never spun)")


MAX_GROUND_ZRANGE_MM = 80  # on the ground the Flow deck reads ~20-30 mm


def check_height_sensor(scf: SyncCrazyflie, n: int = 10) -> None:
    """Abort unless range.zrange looks alive: low on the ground and not frozen."""
    lg = LogConfig(name="zcheck", period_in_ms=100)
    toc = scf.cf.log.toc.toc
    if "range" not in toc or "zrange" not in toc["range"]:
        raise SystemExit("range.zrange not available in firmware. Not taking off.")
    lg.add_variable("range.zrange", "uint16_t")
    vals = []
    with SyncLogger(scf, lg) as logger:
        for _, data, _ in logger:
            vals.append(int(data["range.zrange"]))
            if len(vals) >= n:
                break
    print(f"Height sensor on ground: {vals} mm")
    if max(vals) > MAX_GROUND_ZRANGE_MM:
        raise SystemExit(f"Height sensor reads {max(vals)} mm on the ground (expected "
                         f"< {MAX_GROUND_ZRANGE_MM}). Sensor blocked/faulty. Not taking off.")
    if len(set(vals)) == 1:
        raise SystemExit(f"Height sensor value frozen at {vals[0]} mm. Not taking off.")


def main(uri: str, height: float, distance: float, hover_time: float, live: bool = False,
         min_battery: float = MIN_BATTERY_V) -> None:
    if height < MIN_RECOMMENDED_HEIGHT:
        print(f"WARNING: {height} m is below the documented Flow deck v2 range "
              f"({MIN_RECOMMENDED_HEIGHT}-3.0 m). Height hold may be unreliable.")
    if height > MAX_TEST_HEIGHT:
        raise SystemExit(f"--height above {MAX_TEST_HEIGHT} m refused for this test script.")
    cflib.crtp.init_drivers()
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        if not check_flow_deck(scf):
            raise SystemExit("Flow v2 deck not detected. Check seating / firmware.")
        vbat = read_battery(scf)
        print(f"Battery: {vbat:.2f} V")
        if vbat < min_battery:
            raise SystemExit(f"Battery too low (< {min_battery} V). Charge first, "
                             f"or lower the limit with --min-battery (not below {BATTERY_FLOOR_V}).")
        if vbat < MIN_BATTERY_V:
            print(f"WARNING: battery {vbat:.2f} V is below {MIN_BATTERY_V} V. In flight it sags "
                  "about 0.4-0.5 V: expect high motor output, shorter flights, and a drop if it "
                  "reaches the firmware's cut-off. Keep the flight short and low.")

        sup = read_supervisor(scf)
        if sup.get("isLocked") or sup.get("isCrashed") or sup.get("isTumbled"):
            raise SystemExit(
                f"Drone is locked/crashed/tumbled {sup}. "
                "Power-cycle the Crazyflie on a flat surface and try again."
            )

        check_height_sensor(scf)

        # Bitcraze guide: arm before take-off (newer firmware is disarmed by default)
        scf.cf.supervisor.send_arming_request(True)
        time.sleep(1.0)
        sup = read_supervisor(scf)
        if sup and not (sup.get("isArmed") or sup.get("canFly")):
            raise SystemExit(f"Drone did not arm {sup}. Not taking off.")

        samples, xy, log_configs = start_flight_log(scf)
        live_stop = Event()
        if live:
            live_printer(samples, xy, live_stop)

        print("Take off...")
        with MotionCommander(scf, default_height=height) as mc:
            print(f"Hovering at {height} m for {hover_time} s")
            time.sleep(hover_time)
            if distance > 0:
                print(f"Forward {distance} m")
                mc.forward(distance, velocity=VELOCITY)
                time.sleep(2)
            print("Landing...")
        time.sleep(0.5)
        live_stop.set()
        stop_flight_log(log_configs)
        summarize_flight_log(samples)
        summarize_xy(xy)
        print("Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=DEFAULT_URI)
    ap.add_argument("--height", type=float, default=HOVER_HEIGHT, help="hover height in m")
    ap.add_argument("--distance", type=float, default=MOVE_DISTANCE,
                    help="forward distance in m (0 = hover only)")
    ap.add_argument("--hover-time", type=float, default=3.0, help="seconds to hover")
    ap.add_argument("--live", action="store_true",
                    help="print motor output (%% of max), height, position and battery 4x per second")
    ap.add_argument("--min-battery", type=float, default=MIN_BATTERY_V,
                    help=f"refuse to take off below this voltage (default {MIN_BATTERY_V}, "
                         f"lowest allowed {BATTERY_FLOOR_V})")
    a = ap.parse_args()
    if a.min_battery < BATTERY_FLOOR_V:
        ap.error(f"--min-battery below {BATTERY_FLOOR_V} V is not allowed: the battery is empty")
    main(a.uri, a.height, a.distance, a.hover_time, a.live, a.min_battery)
