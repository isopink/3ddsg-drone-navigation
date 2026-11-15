"""Build the flight report (report.html) of a run folder written by fly_route.py.

    cf-env\\Scripts\\python scripts\\flight_report.py runs\\20261115_101500
    cf-env\\Scripts\\python scripts\\flight_report.py runs\\20261115_101500 --open
    cf-env\\Scripts\\python scripts\\flight_report.py runs --all          # every run folder
    cf-env\\Scripts\\python scripts\\flight_report.py runs\\... --offline  # embed Plotly (pip install plotly)
"""
import argparse
import sys
import webbrowser
from pathlib import Path

from ddsg_nav.report import write_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a run folder (with run.json), or with --all a folder of runs")
    ap.add_argument("--all", action="store_true", help="build the report of every run folder inside")
    ap.add_argument("--out", help="output file (default: <run>/report.html)")
    ap.add_argument("--offline", action="store_true", help="embed Plotly so the page works without internet")
    ap.add_argument("--open", action="store_true", help="open the report in the browser")
    a = ap.parse_args()

    root = Path(a.folder)
    folders = sorted(p.parent for p in root.glob("*/run.json")) if a.all else [root]
    if not folders:
        print(f"no run folders in {root}")
        return 1
    out = None
    for folder in folders:
        try:
            out = write_report(folder, None if a.all else a.out, offline=a.offline)
        except (FileNotFoundError, ValueError) as e:
            print(f"{folder}: {e}")
            return 1
        print(out)
    if a.open and out is not None:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
