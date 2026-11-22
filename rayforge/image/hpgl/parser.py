"""
A parser for the common subset of HP-GL and HP-GL/2 plot files, as
written by cutting plotters, vinyl cutters and CAD programs.

Coordinates are converted from plotter units to millimetres. The result
keeps HP-GL's own orientation, with Y pointing up.
"""

import logging
import math
import re
from dataclasses import dataclass, field

from raygeo.geo import Geometry

logger = logging.getLogger(__name__)

PLOTTER_UNITS_PER_MM = 40.0
MAX_ARC_SWEEP_DEG = 90.0

_ESCAPE_RE = re.compile(rb"\x1b(?:%-?\d*[AB]|E|[^A-Za-z]*[A-Za-z@])")
_LABEL_RE = re.compile(rb"LB[^\x03]*\x03?", re.IGNORECASE)
_INSTRUCTION_RE = re.compile(r"([A-Za-z]{2})([^A-Za-z;]*)")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")

HARMLESS = frozenset(
    {
        "BP",
        "CO",
        "CT",
        "DT",
        "FS",
        "LT",
        "NP",
        "NR",
        "PC",
        "PG",
        "PS",
        "PW",
        "QL",
        "SD",
        "SS",
        "TR",
        "VS",
        "WU",
    }
)


@dataclass
class HpglDrawing:
    """Geometry per pen, in millimetres, plus unsupported instructions."""

    geometries_by_pen: dict[int, Geometry] = field(default_factory=dict)
    unsupported: set[str] = field(default_factory=set)


@dataclass
class _State:
    x: float = 0.0
    y: float = 0.0
    pen: int = 1
    pen_down: bool = False
    absolute: bool = True
    path_open: bool = False


class _Plotter:
    """Executes HP-GL instructions and collects the drawn geometry."""

    def __init__(self):
        self.state = _State()
        self.drawing = HpglDrawing()

    def _geometry(self) -> Geometry | None:
        if self.state.pen <= 0:
            return None
        return self.drawing.geometries_by_pen.setdefault(
            self.state.pen, Geometry()
        )

    def _start_path(self, geo: Geometry):
        if not self.state.path_open:
            geo.move_to(self.state.x, self.state.y)
            self.state.path_open = True

    def _goto(self, x: float, y: float, draw: bool):
        geo = self._geometry() if draw else None
        if geo is not None:
            self._start_path(geo)
            geo.line_to(x, y)
        else:
            self.state.path_open = False
        self.state.x, self.state.y = x, y

    def _target(self, dx: float, dy: float, relative: bool):
        if relative:
            return self.state.x + dx, self.state.y + dy
        return dx, dy

    def plot(self, values: list[float], relative: bool | None = None):
        if relative is None:
            relative = not self.state.absolute
        for i in range(0, len(values) - 1, 2):
            x, y = self._target(values[i], values[i + 1], relative)
            self._goto(x, y, self.state.pen_down)

    def arc(self, cx: float, cy: float, sweep_deg: float, draw: bool):
        radius = math.hypot(self.state.x - cx, self.state.y - cy)
        if radius <= 0.0 or sweep_deg == 0.0:
            return
        start = math.atan2(self.state.y - cy, self.state.x - cx)
        steps = max(1, math.ceil(abs(sweep_deg) / MAX_ARC_SWEEP_DEG))
        step = math.radians(sweep_deg) / steps
        geo = self._geometry() if draw else None
        if geo is not None:
            self._start_path(geo)
        for k in range(1, steps + 1):
            angle = start + step * k
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if geo is not None:
                geo.arc_to(
                    x,
                    y,
                    cx - self.state.x,
                    cy - self.state.y,
                    clockwise=sweep_deg < 0,
                )
            self.state.x, self.state.y = x, y
        if geo is None:
            self.state.path_open = False

    def circle(self, radius: float):
        cx, cy = self.state.x, self.state.y
        geo = self._geometry()
        if geo is None or radius <= 0.0:
            return
        geo.move_to(cx + radius, cy)
        geo.arc_to(cx - radius, cy, -radius, 0.0, clockwise=False)
        geo.arc_to(cx + radius, cy, radius, 0.0, clockwise=False)
        self.state.path_open = False

    def rectangle(self, x: float, y: float):
        x0, y0 = self.state.x, self.state.y
        geo = self._geometry()
        if geo is None:
            return
        geo.move_to(x0, y0)
        for px, py in ((x, y0), (x, y), (x0, y), (x0, y0)):
            geo.line_to(px, py)
        self.state.path_open = False


def _mm(value: float) -> float:
    return value / PLOTTER_UNITS_PER_MM


def _handle_pen(plotter: _Plotter, name: str, values: list[float]):
    state = plotter.state
    if name in ("PU", "PD"):
        state.pen_down = name == "PD"
        if not state.pen_down:
            state.path_open = False
        plotter.plot([_mm(v) for v in values])
    elif name in ("PA", "PR"):
        state.absolute = name == "PA"
        plotter.plot([_mm(v) for v in values])
    elif name == "SP":
        state.pen = int(values[0]) if values else 0
        state.path_open = False


def _handle_shape(plotter: _Plotter, name: str, values: list[float]):
    state = plotter.state
    if name in ("AA", "AR") and len(values) >= 3:
        cx, cy = plotter._target(
            _mm(values[0]), _mm(values[1]), relative=name == "AR"
        )
        plotter.arc(cx, cy, values[2], state.pen_down)
    elif name == "CI" and values:
        plotter.circle(_mm(values[0]))
    elif name in ("EA", "ER", "RA", "RR") and len(values) >= 2:
        x, y = plotter._target(
            _mm(values[0]), _mm(values[1]), relative=name in ("ER", "RR")
        )
        plotter.rectangle(x, y)


_PEN_INSTRUCTIONS = frozenset({"PU", "PD", "PA", "PR", "SP"})
_SHAPE_INSTRUCTIONS = frozenset({"AA", "AR", "CI", "EA", "ER", "RA", "RR"})


def _execute(plotter: _Plotter, name: str, values: list[float]):
    if name in ("IN", "DF"):
        plotter.state = _State(pen=plotter.state.pen)
    elif name in _PEN_INSTRUCTIONS:
        _handle_pen(plotter, name, values)
    elif name in _SHAPE_INSTRUCTIONS:
        _handle_shape(plotter, name, values)
    elif name not in HARMLESS:
        plotter.drawing.unsupported.add(name)


def parse_hpgl(data: bytes) -> HpglDrawing:
    """
    Parses HP-GL data into geometry per pen, in millimetres.

    Supported are IN, DF, SP, PU, PD, PA, PR, CI, AA, AR and the edge
    and fill rectangles EA, ER, RA and RR (drawn as outlines). Labels
    (LB) are skipped. Other instructions that change the drawing, such
    as scaling, are listed in ``unsupported``.
    """
    cleaned = _ESCAPE_RE.sub(b"", data)
    has_labels = _LABEL_RE.search(cleaned) is not None
    cleaned = _LABEL_RE.sub(b";", cleaned)
    text = cleaned.decode("ascii", errors="ignore")

    plotter = _Plotter()
    for match in _INSTRUCTION_RE.finditer(text):
        name = match.group(1).upper()
        values = [float(v) for v in _NUMBER_RE.findall(match.group(2))]
        _execute(plotter, name, values)

    drawing = plotter.drawing
    if has_labels:
        drawing.unsupported.add("LB")
    drawing.geometries_by_pen = {
        pen: geo
        for pen, geo in sorted(drawing.geometries_by_pen.items())
        if not geo.is_empty()
    }
    return drawing
