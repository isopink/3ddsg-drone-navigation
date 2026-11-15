"""Replay a saved run (runs/<time>/ from fly_route.py) in the Rerun viewer.

    cf-env\\Scripts\\python scripts\\replay_run.py runs\\20261115_101500
    cf-env\\Scripts\\python scripts\\replay_run.py runs\\20261115_101500 --save replay.rrd --no-viewer

Scrub the timeline to see where the drone was, how far from the plan, and what the
monitor saw. Needs: pip install rerun-sdk
"""
import argparse
import sys

from ddsg_nav.flight.record import load_run
from ddsg_nav.viz.live import LiveView, replay_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="run folder with run.json")
    ap.add_argument("--save", help="also write an .rrd recording")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the viewer (use with --save)")
    a = ap.parse_args()
    if a.no_viewer and not a.save:
        ap.error("--no-viewer needs --save")
    try:
        run = load_run(a.folder)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1
    view = LiveView(spawn=not a.no_viewer, save=a.save)
    replay_run(run, view)
    print(f"replayed {len(run.flights)} flight(s) from {a.folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
