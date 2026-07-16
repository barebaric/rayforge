"""A non-interactive overlay element that draws shape outlines.

Given a list of world-space polygons (one per source item) and a list
of world-space transform deltas, the element strokes a translucent
outline of every polygon at every transformed position, plus a small
marker at the first corner so copies which only differ by rotation
(e.g. 0 and 180 degrees) remain visually distinct.

It can also draw a guide circle, used by the circular array to show
the virtual circle the copies sit on.

The element is a plain canvas element (added to and removed from the
canvas root by its owner) and carries no behaviour of its own beyond
rendering.
"""

import logging
import math
from typing import List, Optional, Tuple

import cairo

from ...canvas import CanvasElement

logger = logging.getLogger(__name__)


class OutlineElement(CanvasElement):
    """Draws translucent outlines of shapes at a set of transforms."""

    def __init__(self):
        super().__init__(
            0,
            0,
            1,
            1,
            selectable=False,
            visible=True,
            clip=False,
        )
        # Each entry in _shapes is a list of world-space (x, y) corners
        # forming the footprint of one source item.
        self._shapes: List[List[Tuple[float, float]]] = []
        # List of world-space delta Matrices; each shape is drawn once
        # per delta. An identity delta is skipped.
        self._deltas = []
        # Optional guide circle (world center, world radius).
        self._guide_circle: Optional[Tuple[Tuple[float, float], float]] = None

    def set_outlines(
        self,
        shapes,
        deltas,
        guide_circle: Optional[Tuple[Tuple[float, float], float]] = None,
    ) -> None:
        """Stores the per-item shape corners, transform deltas and an
        optional guide circle, then redraws.

        ``shapes`` is a list of polygon definitions; each polygon is
        itself a list of ``(x, y)`` corner tuples in world-space.
        For backward compatibility a single ``(min_x, min_y, max_x,
        max_y)`` bounding-box tuple is also accepted and converted to a
        single-axis-aligned rectangle.
        """
        self._shapes = self._normalise_shapes(shapes)
        self._deltas = list(deltas)
        self._guide_circle = guide_circle
        if self.canvas:
            self.canvas.queue_draw()

    @staticmethod
    def _normalise_shapes(shapes):
        """Accepts both the legacy bbox format (4‑tuple) and the new
        list-of-polygons format."""
        if not shapes:
            return []
        # bbox format: (min_x, min_y, max_x, max_y)
        if isinstance(shapes, tuple) and len(shapes) == 4:
            min_x, min_y, max_x, max_y = shapes
            return [
                [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                ]
            ]
        return list(shapes)

    def clear(self) -> None:
        """Clears all outlines."""
        self._deltas = []
        self._shapes = []
        self._guide_circle = None
        if self.canvas:
            self.canvas.queue_draw()

    def draw_overlay(self, ctx: cairo.Context) -> None:
        if not self.canvas:
            return
        view = self.canvas.view_transform

        if self._guide_circle is not None:
            self._draw_guide_circle(ctx, view)

        if not self._shapes or not self._deltas:
            return

        ctx.save()
        for delta in self._deltas:
            if delta.is_identity():
                continue
            for corners in self._shapes:
                screen = [
                    view.transform_point(delta.transform_point(p))
                    for p in corners
                ]

                # Translucent fill.
                ctx.set_source_rgba(0.45, 0.70, 1.0, 0.12)
                x0, y0 = screen[0]
                ctx.move_to(round(x0) + 0.5, round(y0) + 0.5)
                for sx, sy in screen[1:]:
                    ctx.line_to(round(sx) + 0.5, round(sy) + 0.5)
                ctx.close_path()
                ctx.fill_preserve()

                # Dashed outline.
                ctx.set_source_rgba(0.45, 0.70, 1.0, 0.9)
                ctx.set_line_width(1.0)
                ctx.set_dash([5.0, 3.0])
                ctx.stroke()

                # Orientation marker at the first corner.
                ctx.set_source_rgba(0.45, 0.70, 1.0, 0.95)
                ctx.set_dash([])
                ctx.arc(screen[0][0], screen[0][1], 2.5, 0.0, 2.0 * math.pi)
                ctx.fill()
        ctx.restore()

    def _draw_guide_circle(self, ctx: cairo.Context, view) -> None:
        guide = self._guide_circle
        if guide is None:
            return
        center, radius = guide
        if radius <= 0:
            return
        cx, cy = view.transform_point(center)
        px = view.transform_point((center[0] + 1.0, center[1]))
        py = view.transform_point((center[0], center[1] + 1.0))
        scale = (abs(px[0] - cx) + abs(py[1] - cy)) / 2.0
        r_px = radius * scale

        ctx.save()
        ctx.set_source_rgba(0.45, 0.70, 1.0, 0.55)
        ctx.set_line_width(1.0)
        ctx.set_dash([3.0, 3.0])
        ctx.arc(cx, cy, r_px, 0.0, 2.0 * math.pi)
        ctx.stroke()
        # Mark the centre.
        ctx.set_dash([])
        ctx.arc(cx, cy, 2.0, 0.0, 2.0 * math.pi)
        ctx.fill()
        ctx.restore()
