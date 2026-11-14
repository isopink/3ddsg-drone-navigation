"""Save and load a SceneGraph as a folder with two files:

    graph.json   rooms, doors, solids, voxel grid metadata (human readable, diff-able)
    voxels.npz   the voxel states (uint8: 0 unknown, 1 free, 2 occupied), compressed

This is the exchange format for real data: whoever has the graph database can export
this folder, and everything here (planning, validation, simulation) works on it
without a database connection. See docs/graph_format.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Box, Door, Room, SceneGraph, VoxelGrid

FORMAT = "ddsg-scene-graph"
VERSION = 1


class GraphFormatError(ValueError):
    """The folder is not a valid scene graph export."""


def _box(box: Box) -> dict:
    return {"min": list(box.min), "max": list(box.max)}


def _unbox(d: dict) -> Box:
    return Box(tuple(float(v) for v in d["min"]), tuple(float(v) for v in d["max"]))  # type: ignore[arg-type]


def save_graph(graph: SceneGraph, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = graph.voxels
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "voxels": {
            "origin": list(g.origin),
            "resolution": g.resolution,
            "shape": list(g.shape),
            "file": "voxels.npz",
        },
        "rooms": [
            {
                "id": r.id,
                "name": r.name,
                "aliases": list(r.aliases),
                "bounds": _box(r.bounds),
                "polygon": [list(p) for p in r.polygon] if r.polygon else None,
            }
            for r in graph.rooms.values()
        ],
        "doors": [
            {
                "id": d.id,
                "room_a": d.room_a,
                "room_b": d.room_b,
                "position": list(d.position),
                "width": d.width,
            }
            for d in graph.doors.values()
        ],
        "solids": [_box(b) for b in graph.solids],
    }
    (folder / "graph.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(folder / "voxels.npz", states=g.states)
    return folder


def load_graph(folder: str | Path) -> SceneGraph:
    folder = Path(folder)
    json_path = folder / "graph.json"
    if not json_path.is_file():
        raise GraphFormatError(f"{json_path} not found")
    try:
        doc = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise GraphFormatError(f"{json_path}: invalid JSON ({e})") from e
    if doc.get("format") != FORMAT:
        raise GraphFormatError(f"{json_path}: not a {FORMAT} file")
    if doc.get("version") != VERSION:
        raise GraphFormatError(f"{json_path}: version {doc.get('version')} (supported: {VERSION})")

    try:
        v = doc["voxels"]
        npz_path = folder / v.get("file", "voxels.npz")
        if not npz_path.is_file():
            raise GraphFormatError(f"{npz_path} not found")
        with np.load(npz_path) as npz:
            states = np.asarray(npz["states"], dtype=np.uint8)
        shape = tuple(int(n) for n in v["shape"])
        if states.shape != shape:
            raise GraphFormatError(f"voxel array shape {states.shape} does not match metadata {shape}")
        if states.size and int(states.max()) > 2:
            raise GraphFormatError("voxel states must be 0 (unknown), 1 (free) or 2 (occupied)")
        graph = SceneGraph(
            voxels=VoxelGrid(
                tuple(float(o) for o in v["origin"]),  # type: ignore[arg-type]
                float(v["resolution"]),
                states,
            )
        )
        for r in doc["rooms"]:
            polygon = r.get("polygon")
            graph.add_room(
                Room(
                    id=str(r["id"]),
                    name=str(r["name"]),
                    bounds=_unbox(r["bounds"]),
                    aliases=tuple(str(a) for a in r.get("aliases", [])),
                    polygon=tuple((float(x), float(y)) for x, y in polygon) if polygon else None,
                )
            )
        for d in doc["doors"]:
            graph.add_door(
                Door(
                    id=str(d["id"]),
                    room_a=str(d["room_a"]),
                    room_b=str(d["room_b"]),
                    position=tuple(float(c) for c in d["position"]),  # type: ignore[arg-type]
                    width=float(d.get("width", 0.9)),
                )
            )
        graph.solids = [_unbox(b) for b in doc.get("solids", [])]
    except KeyError as e:
        raise GraphFormatError(f"{json_path}: missing field {e}") from e
    return graph
