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

## Status

| Step | | |
|---|---|---|
| 0 | Setup, hover, forward flight | Done on the real drone |
| 1 | Goal parsing, route planning | Done |
| 2 | Collision check, visualization | Done |
| 3 | Virtual-room flight | Simulation done. **Real flight pending** |
| 4 | Real graph (Neo4j) | **Blocked: needs graph export or schema** |
| 5 | Dynamic objects, re-planning | Not started |

## Tasks

Open tasks are tracked in **[Issues](https://github.com/isopink/3ddsg-drone-navigation/issues)**. Each has a goal, the files it touches and a done criterion.

## Quick start

Windows, [Python 3.12](https://www.python.org/downloads/). Clone the repo, then double-click:

1. **`setup.bat`**: installs `cf-env` (first run only).
2. **`experiment_terminal.bat`**: a terminal with the environment active. Fly from here.

Command reference (Korean): `type scripts\manual.txt`

Manual setup:

```
py -3.12 -m venv cf-env
cf-env\Scripts\activate
pip install cfclient
pip install -e ".[neo4j,viz,sim,dev]"
pytest                      # 212 tests, ~2 min, no drone needed

python scripts\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
```

Real flights need a Crazyflie 2.1 with Flow deck v2 and a Crazyradio PA (libusb driver via Zadig).
Optional LLM goal parsing: copy `.env.example` to `.env` and set `DDSG_LLM_API_KEY` (DeepSeek by default, any OpenAI-compatible API works).

## Before flying the real drone

- **Bright room, textured floor.** In the dark the Flow deck fails and the drone crashes. Valid height is 0.2 to 3.0 m.
- **Simulate first.** `--backend crazyflie` runs a dry run and refuses to fly if it touches a wall.
- Start at **0.3 m/s** on a short route. Keep a hand near the drone. `Ctrl+C` lands it.
- Battery at 3.7 V or more. After a crash, check the battery, propellers and deck pins.
- The simulator has no sensor model: it does not reproduce Flow deck drift.
- `backends/crazyflie.py` has not been flight-tested; only `scripts/hover_test.py` was flown.

## Layout

```
src/ddsg_nav/   graph/  goal.py  llm_goal.py  validate.py  planner/  sim/  flight/ (backends/)  viz/
scripts/        fly_route.py  hover_test.py  check_sensors.py  graph_tools.py  manual.txt  ...
tests/          run without a drone
data/           local graphs, never committed
docs/           graph_format.md (graph exchange format), media/
```

Never commit lab data (graphs, IFC, point clouds).
