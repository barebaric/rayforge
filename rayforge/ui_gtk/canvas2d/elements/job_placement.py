import math

import cairo

from ...canvas import CanvasElement

_STROKE_COLOR = (0.1, 0.55, 0.95, 1.0)
_DASH_PX = 6.0
_LINE_PX = 2.0


class JobPlacementElement(CanvasElement):
    """
    A non-interactive CanvasElement that shows where a job placed via
    Start From ("User Origin" or "Current Position") will run: a dashed
    outline of the placed job and a marker on its job origin point.

    The element covers the placed job's bounding box; the anchor is
    given in the element's local coordinates (mm).
    """

    def __init__(self, **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=10.0,
            height=10.0,
            selectable=False,
            draggable=False,
            clip=False,
            **kwargs,
        )
        self._anchor = (0.0, 0.0)

    def set_placement(
        self,
        rect: tuple[float, float, float, float],
        anchor: tuple[float, float],
    ):
        """
        Places the outline on *rect* (min_x, min_y, max_x, max_y) and
        the marker on *anchor*, both in canvas coordinates.
        """
        min_x, min_y, max_x, max_y = rect
        self.set_size(max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6))
        self.set_pos(min_x, min_y)
        self._anchor = (anchor[0] - min_x, anchor[1] - min_y)

    @property
    def anchor(self) -> tuple[float, float]:
        """The anchor in the element's local coordinates."""
        return self._anchor

    def draw(self, ctx: cairo.Context):
        ctx.save()
        ctx.set_source_rgba(*_STROKE_COLOR)
        line, _unused = ctx.device_to_user_distance(_LINE_PX, 0)
        ctx.set_line_width(abs(line) or 0.1)
        dash, _unused = ctx.device_to_user_distance(_DASH_PX, 0)
        ctx.set_dash([abs(dash) or 1.0])
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.stroke()

        ctx.set_dash([])
        ax, ay = self._anchor
        radius = max(min(self.width, self.height) * 0.08, 1.0)
        ctx.arc(ax, ay, radius, 0, 2 * math.pi)
        ctx.stroke()
        ctx.move_to(ax - 2 * radius, ay)
        ctx.line_to(ax + 2 * radius, ay)
        ctx.move_to(ax, ay - 2 * radius)
        ctx.line_to(ax, ay + 2 * radius)
        ctx.stroke()
        ctx.restore()
