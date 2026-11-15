"""
The layout a calibration generator produces, and the geometry helpers
the generators share.

A layout is an ordered list of entries. The order is the order in which
the job runs them. All geometry is in test-local millimeters, Y up, with
the origin at the bottom-left corner of the test; the builder moves the
whole test to its place on the work area.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from raygeo.geo import Geometry, Matrix
from raygeo.geo.shape.text import FontConfig, text_to_geometry

from rayforge.core.step import Step


@dataclass
class VectorShape:
    """Vector outlines of a cell, such as lines or label text."""

    geometry: Geometry


@dataclass
class FilledRect:
    """A filled rectangle, engraved as an area."""

    x: float
    y: float
    width: float
    height: float


@dataclass
class ProceduralShape:
    """
    Content drawn by a function, such as the Material Test Grid preview.
    *origin* is its bottom-left corner in test-local coordinates.
    """

    draw_function: Callable[..., Any]
    size_function: Callable[[dict[str, Any]], tuple[float, float]]
    params: dict[str, Any]
    origin: tuple[float, float] = (0.0, 0.0)


Shape = VectorShape | FilledRect | ProceduralShape


@dataclass
class CellEntry:
    """One workpiece and the step that processes only that workpiece."""

    name: str
    shape: Shape
    step: Step


@dataclass
class CommandEntry:
    """Machine commands that run between two cells."""

    name: str
    lines: list[str]


Entry = CellEntry | CommandEntry


@dataclass
class CalibrationLayout:
    """The entries of a test, in execution order."""

    entries: list[Entry] = field(default_factory=list)

    @property
    def cells(self) -> list[CellEntry]:
        return [e for e in self.entries if isinstance(e, CellEntry)]

    @property
    def commands(self) -> list[CommandEntry]:
        return [e for e in self.entries if isinstance(e, CommandEntry)]


def row_positions(count: int, size: float, spacing: float) -> list[float]:
    """The offsets of *count* items of *size* placed in a row."""
    return [i * (size + spacing) for i in range(count)]


def line(x0: float, y0: float, x1: float, y1: float) -> Geometry:
    geo = Geometry()
    geo.move_to(x0, y0)
    geo.line_to(x1, y1)
    return geo


def text(
    content: str,
    height: float,
    center_x: float,
    top: float,
    max_width: float | None = None,
) -> Geometry:
    """
    Glyph outlines of *content*, *height* mm tall and centered below
    *top*, scaled down further when needed to fit in *max_width*.
    """
    geo = text_to_geometry(content, FontConfig(size=10.0))
    min_x, min_y, max_x, max_y = geo.rect()
    scale = height / max(max_y - min_y, 1e-9)
    if max_width is not None and (max_x - min_x) * scale > max_width:
        scale = max_width / max(max_x - min_x, 1e-9)
    width = (max_x - min_x) * scale
    text_height = (max_y - min_y) * scale
    matrix = (
        Matrix.translation(center_x - width / 2, top - text_height)
        @ Matrix.scale(scale, scale)
        @ Matrix.translation(-min_x, -min_y)
    )
    geo.transform(matrix)
    return geo
