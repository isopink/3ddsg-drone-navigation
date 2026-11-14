"""Render a simulated flight as a video (MuJoCo, offscreen) plus a GIF preview.

The flight is simulated first, then replayed frame by frame from the recorded positions
(the drone is drawn at its recorded position, without its tilt). The planned route is a red
line; the part already flown grows behind the drone as a blue trail.

Run from the repository root:
    cf-env\\Scripts\\python scripts\\render_sim_video.py "go from A to C" --rooms 2 2 2 --depth 2
    cf-env\\Scripts\\python scripts\\render_sim_video.py "go from A to C" --obstacle   # 3-room fake graph with a box
Needs ffmpeg on the PATH.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ddsg_nav.flight.backends.mujoco_backend import MuJoCoBackend
from ddsg_nav.flight.executor import follow_path
from ddsg_nav.goal import parse_goal
from ddsg_nav.graph.fake import build_fake_graph, build_row_graph
from ddsg_nav.planner.route import plan_route
from ddsg_nav.sim.scene import build_mjcf

RED = (0.85, 0.1, 0.1, 1.0)
BLUE = (0.1, 0.35, 0.95, 1.0)
GREEN = (0.1, 0.75, 0.2, 1.0)
MAGENTA = (0.8, 0.1, 0.7, 1.0)
ORANGE = (1.0, 0.55, 0.0, 1.0)


def add_segment(mujoco, scene, a, b, width, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def add_marker(mujoco, scene, pos, radius, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, float), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def thin(points: np.ndarray, min_gap: float) -> np.ndarray:
    """Keep points that are at least min_gap apart (stationary samples make zero-length segments)."""
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) >= min_gap:
            keep.append(i)
    return np.array(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="go from A to C")
    ap.add_argument("--rooms", type=float, nargs="+", default=[2.0, 2.0, 2.0])
    ap.add_argument("--depth", type=float, default=2.0)
    ap.add_argument("--obstacle", action="store_true", help="use the 3-room fake graph with a box in B")
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--velocity", type=float, default=0.3)
    ap.add_argument("--out", default="docs/media/sim_flight")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--speedup", type=float, default=1.5, help="video speed relative to the simulation")
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-55.0)
    ap.add_argument("--distance", type=float, default=0.0, help="camera distance (0 = automatic)")
    a = ap.parse_args()

    import mujoco

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found on PATH")
        return 1
    width, height = a.size

    graph = build_fake_graph(obstacle=True) if a.obstacle else build_row_graph(a.rooms, a.depth, 1.0)
    goal = parse_goal(a.text, graph)
    plan = plan_route(graph, goal.start, goal.goal, height=a.height)
    start = plan.waypoints[0]
    with MuJoCoBackend(graph, (start[0], start[1], 0.0)) as sim:
        follow_path(sim, plan.waypoints, velocity=a.velocity)
        traj = np.array(sim.trajectory)
    print(f"simulated {traj[-1, 0]:.1f} s, {len(traj)} samples, wall contacts {sim.collisions}")

    record_dt = float(traj[1, 0] - traj[0, 0])
    step = max(1, round(a.speedup / a.fps / record_dt))
    idx = list(range(0, len(traj), step)) + [len(traj) - 1]
    xyz = traj[:, 1:4]
    trail_idx = thin(xyz, 0.04)

    xml = build_mjcf(graph, (start[0], start[1], 0.06))
    # bright, shadow-free look on a white background so route and drone stand out
    xml = xml.replace(
        "<option ",
        '<visual><headlight ambient="0.62 0.62 0.62" diffuse="0.55 0.55 0.55" specular="0 0 0"/>'
        f'<global offwidth="{width}" offheight="{height}"/></visual>\n'
        '  <asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/></asset>\n'
        "  <option ",
        1,
    )
    xml = xml.replace('dir="0 0 -1"/>', 'dir="0 0 -1" castshadow="false"/>', 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    adr = model.jnt_qposadr[model.joint("root").id]

    cam = mujoco.MjvCamera()
    lo = np.min([r.bounds.min for r in graph.rooms.values()], axis=0)
    hi = np.max([r.bounds.max for r in graph.rooms.values()], axis=0)
    cam.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] / 2]  # middle of the walls' height
    cam.azimuth, cam.elevation = a.azimuth, a.elevation
    cam.distance = a.distance or float(max(hi[0] - lo[0], hi[1] - lo[1]) * 1.05 + 1.0)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = out.with_suffix(".mp4"), out.with_suffix(".gif")
    encoder = subprocess.Popen(
        [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
         "-movflags", "+faststart", str(mp4)],
        stdin=subprocess.PIPE,
    )
    with mujoco.Renderer(model, height, width, max_geom=3000) as renderer:
        for i in idx:
            data.qpos[adr:adr + 3] = xyz[i]
            data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            for p, q in zip(plan.waypoints, plan.waypoints[1:]):
                add_segment(mujoco, renderer.scene, p, q, 0.03, RED)
            done = trail_idx[trail_idx <= i]
            for j, k in zip(done, done[1:]):
                add_segment(mujoco, renderer.scene, xyz[j] + (0, 0, 0.04), xyz[k] + (0, 0, 0.04), 0.025, BLUE)
            add_marker(mujoco, renderer.scene, plan.waypoints[0], 0.09, GREEN)
            add_marker(mujoco, renderer.scene, plan.waypoints[-1], 0.09, MAGENTA)
            add_marker(mujoco, renderer.scene, xyz[i], 0.11, ORANGE)  # the drone, drawn large
            encoder.stdin.write(renderer.render().tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        print("ffmpeg failed")
        return 1

    secs = len(idx) / a.fps
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)
    for label, t in (("start", 0.0), ("middle", secs / 2), ("end", max(secs - 0.1, 0.0))):
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                        str(out.with_name(out.name + f"_{label}.png"))], check=True)
    print(f"{len(idx)} frames, {secs:.0f} s -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB), "
          f"{gif} ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
