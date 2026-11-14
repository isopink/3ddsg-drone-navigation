"""Where a SceneGraph comes from. Everything downstream only needs `GraphSource.load()`."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .fake import build_fake_graph
from .io import load_graph
from .model import SceneGraph


class GraphSource(Protocol):
    def load(self) -> SceneGraph: ...


class FakeGraphSource:
    """The 3-room development graph."""

    def __init__(self, obstacle: bool = False) -> None:
        self.obstacle = obstacle

    def load(self) -> SceneGraph:
        return build_fake_graph(obstacle=self.obstacle)


class FileGraphSource:
    """A folder exported in the format of graph/io.py."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)

    def load(self) -> SceneGraph:
        return load_graph(self.folder)
