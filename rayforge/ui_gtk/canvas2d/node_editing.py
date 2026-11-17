"""
Helpers for editing the nodes of a workpiece's vector path: which nodes
and Bezier handles are shown, hit testing them, and drawing them.

Indices are command indices into the boundaries geometry; a node is the
end point of a command. In a closed contour the start and end node are
one node, shown once at its Move index. Positions are in boundary space
(0..1).
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import cairo
from raygeo.geo import Bezier, Geometry

NODE_HIT_PX = 7.0
NODE_SIZE_PX = 7.0
HANDLE_SIZE_PX = 6.0

Point2 = tuple[float, float]
ToScreen = Callable[[Point2], Point2]


@dataclass(frozen=True)
class Handle:
    """A Bezier handle: control point ``which`` (1 or 2) of ``index``."""

    index: int
    which: int
    point: Point2
    anchor: Point2


@dataclass
class NodeDrag:
    """An in-progress drag of a node or a handle."""

    index: int
    original: Geometry
    handle: int | None = None
    moved: bool = False


def _end(cmd) -> Point2:
    return (cmd.end[0], cmd.end[1])


def visible_nodes(
    geometry: Geometry, segments: Iterable[int], node: int | None
) -> list[int]:
    """Nodes of every contour holding a selected segment or node."""
    anchors = set(segments)
    if node is not None:
        anchors.add(node)
    infos = {
        geometry.contour_info(i)
        for i in anchors
        if 0 <= i < len(geometry.data)
    }
    nodes: list[int] = []
    for start, end, closed in sorted(infos):
        nodes.extend(range(start, end) if closed else range(start, end + 1))
    return nodes


def visible_handles(
    geometry: Geometry, segments: Iterable[int], node: int | None
) -> list[Handle]:
    """Handles of the selected Beziers and of those meeting at the node."""
    data = geometry.data
    indices = set(segments)
    if node is not None:
        indices.update((node, node + 1))
        if 0 <= node < len(data):
            start, end, closed = geometry.contour_info(node)
            if closed and node in (start, end):
                indices.update((start + 1, end))
    handles = []
    for index in sorted(indices):
        if not 0 < index < len(data):
            continue
        cmd = data[index]
        if not isinstance(cmd, Bezier):
            continue
        node_start = _end(data[index - 1])
        c1 = (cmd.control1[0], cmd.control1[1])
        c2 = (cmd.control2[0], cmd.control2[1])
        handles.append(Handle(index, 1, c1, node_start))
        handles.append(Handle(index, 2, c2, _end(cmd)))
    return handles


def nearest(
    candidates: Iterable[tuple[object, Point2]],
    local: Point2,
    tolerance: Point2,
) -> object | None:
    """The key of the closest candidate within an elliptic tolerance."""
    tx, ty = tolerance
    if tx <= 0 or ty <= 0:
        return None
    best, best_distance = None, 1.0
    for key, (x, y) in candidates:
        distance = ((x - local[0]) / tx) ** 2 + ((y - local[1]) / ty) ** 2
        if distance <= best_distance:
            best, best_distance = key, distance
    return best


def node_positions(data: Sequence, nodes: Iterable[int]):
    """(index, position) pairs for hit testing nodes."""
    return [(i, _end(data[i])) for i in nodes]


def handle_positions(handles: Iterable[Handle]):
    """(handle, position) pairs for hit testing handles."""
    return [(h, h.point) for h in handles]


def draw_handles(
    ctx: cairo.Context, to_screen: ToScreen, handles: Iterable[Handle]
) -> None:
    """Handle lines and round handle ends, in screen space."""
    ctx.save()
    ctx.set_line_width(1.0)
    for handle in handles:
        ax, ay = to_screen(handle.anchor)
        hx, hy = to_screen(handle.point)
        ctx.set_source_rgba(0.4, 0.6, 1.0, 0.8)
        ctx.move_to(ax, ay)
        ctx.line_to(hx, hy)
        ctx.stroke()
        ctx.arc(hx, hy, HANDLE_SIZE_PX / 2.0, 0, math.tau)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.2, 0.4, 0.9, 1.0)
        ctx.stroke()
    ctx.restore()


def draw_nodes(
    ctx: cairo.Context,
    to_screen: ToScreen,
    data: Sequence,
    nodes: Iterable[int],
    selected: int | None,
) -> None:
    """Square node markers in screen space; the selected one is filled."""
    half = NODE_SIZE_PX / 2.0
    ctx.save()
    ctx.set_line_width(1.0)
    for index in nodes:
        x, y = to_screen(_end(data[index]))
        ctx.rectangle(x - half, y - half, NODE_SIZE_PX, NODE_SIZE_PX)
        if index == selected:
            ctx.set_source_rgba(1.0, 0.3, 0.3, 1.0)
        else:
            ctx.set_source_rgba(1.0, 1.0, 1.0, 0.95)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.2, 0.2, 0.2, 1.0)
        ctx.stroke()
    ctx.restore()
