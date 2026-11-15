# Overview

<p align="center">
  <img src="docs/media/night_flight.gif" alt="The drone flying in the dark" width="240">
  <br>
  <sub>First flight in the dark: it lost its way. Hence rule one below: bright room. · <a href="docs/media/night_flight.mp4">mp4</a></sub>
</p>

An indoor drone (**Crazyflie 2.1 + Flow deck**) that flies from room A to room C without hitting a wall. Its map is an IFC-based **3D Dynamic Scene Graph (3DDSG)**.

This repo takes the graph and flies the drone. Building the graph and lab data are out of scope.

## How it works

Say "go from A to C". The goal parser picks the two rooms (rules first, optional LLM as fallback; it can only choose rooms that exist in the graph). The planner orders the rooms, then finds a voxel path with a 25 cm wall margin. The executor flies the waypoints in MuJoCo or on the real drone, checks after every waypoint, and lands if the drone leaves the plan.

<p align="center">
  <img src="docs/media/system_overview.svg" alt="System overview: command, goal parsing, scene graph, route planning, execution" width="100%">
</p>

## Goals

Two stages, in this order:

1. **Same floor:** room → bent corridor → room.
2. **Between floors:** room → corridor → floor change (stairwell) → room.

Both already work in simulation on generated test buildings. The end point is a lab demo on the real drone: stage 1 (M4) at the end of November 2026, in the building the 3DDSG describes.

## Roadmap

Done: setup, hover and forward flight on the real drone; goal parsing; route planning; collision check; simulated flight (corridor and stairwell).

| Milestone | | Status |
|---|---|---|
| M1 | Real flight in virtual rooms | Hover and backend review done. **Route flight next** (#10), needs a person |
| M2 | Real building graph (test IFC now, 3DDSG when it arrives) | IFC pipeline ready. Waiting for the IFC and the 3DDSG |
| M3 | Range-sensor obstacle stop (Multi-ranger deck) | Simulation and software done. Waiting for the deck |
| M4 | Demo 1: same floor, real flight in the 3DDSG building | After M1 and M2. **Due end of November** |
| M5 | Demo 2: between floors | After M4 |
| M6 | Dynamic objects, re-planning | After M3 |

M1, M2 and M3 touch different files and run in parallel.

## Tasks

Open tasks are tracked in **[Issues](https://github.com/isopink/3ddsg-drone-navigation/issues)**, grouped by [milestone](https://github.com/isopink/3ddsg-drone-navigation/milestones). Each has a goal, the files it owns and a done criterion. One issue, one branch, one pull request.

## Quick start

Windows, [Python 3.12](https://www.python.org/downloads/). Clone the repo, then double-click:

1. **`setup.bat`**: installs `cf-env` (first run only).
2. **`experiment_terminal.bat`**: a terminal with the environment active. Fly from here.

Command reference: `type scripts\manual.txt`

Manual setup:

```
py -3.12 -m venv cf-env
cf-env\Scripts\activate
pip install cfclient
pip install -e ".[neo4j,viz,sim,dev]"
pytest                      # about 250 tests, under a minute, no drone needed

python scripts\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
```

Real IFC: `pip install -e ".[ifc,sim]"`, then one command converts a storey, validates it, plans routes and flies them in simulation.
`python scripts\ifc_pipeline.py model.ifc` lists the storeys; `python scripts\ifc_pipeline.py model.ifc --storey NAME` runs everything (graph in `data\NAME`).
It exits with 0 only if every step passes with 0 wall contacts; each failure is printed with its reason and what to do.

Real flights need a Crazyflie 2.1 with Flow deck v2 and a Crazyradio PA (libusb driver via Zadig).
Optional LLM goal parsing: copy `.env.example` to `.env` and set `DDSG_LLM_API_KEY` (DeepSeek by default, any OpenAI-compatible API works).

## When the data and hardware arrive

| Arrives | Run |
|---|---|
| IFC | `python scripts\ifc_pipeline.py model.ifc --storey NAME` |
| 3DDSG (Neo4j or exported folder) | fill in `Neo4jMapping` (`docs/graph_format.md`), then `python scripts\graph_pipeline.py --neo4j URI --floor NAME --save data\NAME` |
| A graph folder | `python scripts\check_routes.py --graph data\NAME "from A to C" --png`, then `python scriptsly_route.py --backend sim --graph data\NAME "from A to C"` |
| Multi-ranger deck | `python scripts\check_sensors.py --watch`, then fly with `--check-step 0.1` |

What to ask the 3DDSG builders for: `docs/3ddsg_request.md`. Each closed issue has a checklist for the day its data or hardware arrives.

## Layout

```
src/ddsg_nav/   graph/  goal.py  llm_goal.py  validate.py  planner/  sim/  flight/ (backends/)  viz/
scripts/        fly_route.py  hover_test.py  check_sensors.py  graph_tools.py  manual.txt  ...
tests/          run without a drone
data/           local graphs, never committed
docs/           graph_format.md (graph exchange format), media/
```

Never commit lab data (graphs, IFC, point clouds).

## Before flying the real drone

- **Bright room, textured floor.** In the dark the Flow deck fails and the drone crashes. Valid height is 0.2 to 3.0 m.
- **Simulate first.** `--backend crazyflie` runs a dry run and refuses to fly if it touches a wall.
- Start at **0.3 m/s** on a short route. Keep a hand near the drone. `Ctrl+C` lands it.
- Battery at 3.7 V or more. After a crash, check the battery, propellers and deck pins.
- The simulator has no sensor model: it does not reproduce Flow deck drift.
- `backends/crazyflie.py` has not been flight-tested; only `scripts/hover_test.py` was flown.
