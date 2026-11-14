import math

import cairo

from ...canvas import CanvasElement


class WorkOriginElement(CanvasElement):
    """
    A non-interactive CanvasElement that draws a CNC-style work origin
    symbol (a quadrant with two axis arrows). Its position on the canvas
    represents the physical location of the active Work Coordinate System's
    zero point.
    """

    ARROWHEAD_PX = 6.0
    ARROWHEAD_ANGLE = math.pi / 4
    ARROWHEAD_MAX_FRACTION = 0.3

    def __init__(self, **kwargs):
        # The element's size is in world units (mm), so it scales with zoom.
        super().__init__(
            x=0,
            y=0,
            width=15.0,
            height=15.0,
            selectable=False,
            draggable=False,
            clip=False,  # Allow drawing outside bounds when scaled/flipped
            **kwargs,
        )
        self._x_dir = (1.0, 0.0)
        self._y_dir = (0.0, 1.0)

    def set_axis_direction(self, x_axis_right: bool, y_axis_down: bool):
        """
        Configure the arrow directions from the displayed axis orientation.

        The arrows show the work directions: which way to move from the
        origin corner into the bed. They depend only on the origin corner
        and never on axis negation (reverse_x/reverse_y).
        """
        self.set_axis_vectors(
            (-1.0, 0.0) if x_axis_right else (1.0, 0.0),
            (0.0, -1.0) if y_axis_down else (0.0, 1.0),
        )

    def set_axis_vectors(
        self,
        x_dir: tuple[float, float],
        y_dir: tuple[float, float],
    ):
        """
        Configure the arrow directions as unit vectors in canvas space.

        The vectors point away from the origin corner into the bed (the
        work directions), independent of axis negation.
        """
        if self._x_dir == x_dir and self._y_dir == y_dir:
            return

        self._x_dir = x_dir
        self._y_dir = y_dir

        # Trigger a redraw when orientation changes
        if self.canvas:
            self.canvas.queue_draw()

    def draw(self, ctx: cairo.Context):
        """
        Renders the origin symbol.
        """
        ctx.save()

        # Set drawing properties. Stroke width and arrowhead size are in
        # device space: world-unit dimensions would be thinner than a
        # pixel at most zoom levels, leaving the marker invisible after
        # anti-aliasing.
        ctx.set_source_rgba(0.2, 0.8, 0.2, 1.0)  # A distinct green color
        ctx.set_hairline(True)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        axis_len = self.width
        self._draw_axis_arrow(ctx, self._x_dir, axis_len)
        self._draw_axis_arrow(ctx, self._y_dir, axis_len)

        ctx.restore()

    def _draw_axis_arrow(self, ctx: cairo.Context, direction, length: float):
        """Draws one axis as a shaft with an arrowhead at its tip."""
        dir_x, dir_y = direction
        tip_x = dir_x * length
        tip_y = dir_y * length

        ctx.new_path()
        ctx.move_to(0.0, 0.0)
        ctx.line_to(tip_x, tip_y)
        ctx.stroke()

        head_len = self._arrowhead_length(ctx, direction, length)
        cos_a = math.cos(self.ARROWHEAD_ANGLE)
        sin_a = math.sin(self.ARROWHEAD_ANGLE)
        for sign in (1.0, -1.0):
            wing_x = -(dir_x * cos_a - sign * dir_y * sin_a)
            wing_y = -(sign * dir_x * sin_a + dir_y * cos_a)
            ctx.new_path()
            ctx.move_to(tip_x, tip_y)
            ctx.line_to(tip_x + wing_x * head_len, tip_y + wing_y * head_len)
            ctx.stroke()

    def _arrowhead_length(
        self, ctx: cairo.Context, direction, max_length: float
    ) -> float:
        """
        Returns the arrowhead length in local units that corresponds to a
        constant device-space size, capped to a fraction of the shaft.
        """
        dev_x, dev_y = ctx.get_matrix().transform_distance(*direction)
        scale = math.hypot(dev_x, dev_y)
        if scale <= 0.0:
            return 0.0
        return min(
            self.ARROWHEAD_PX / scale,
            max_length * self.ARROWHEAD_MAX_FRACTION,
        )
