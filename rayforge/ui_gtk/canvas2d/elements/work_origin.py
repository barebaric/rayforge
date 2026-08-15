import cairo

from ...canvas import CanvasElement


class WorkOriginElement(CanvasElement):
    """
    A non-interactive CanvasElement that draws a CNC-style work origin
    symbol (a quadrant with two axes arrows). Its position on the canvas
    represents the physical location of the active Work Coordinate System's
    zero point.
    """

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

        # Set drawing properties
        ctx.set_source_rgba(0.2, 0.8, 0.2, 0.9)  # A distinct green color
        ctx.set_line_width(0.2)  # Use a thin line width in world units (mm)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        # --- Draw X-Axis with Arrow ---
        axis_len = self.width
        x_dir_x, x_dir_y = self._x_dir
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(x_dir_x * axis_len, x_dir_y * axis_len)
        ctx.stroke()

        # --- Draw Y-Axis with Arrow ---
        y_dir_x, y_dir_y = self._y_dir
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(y_dir_x * axis_len, y_dir_y * axis_len)
        ctx.stroke()

        ctx.restore()
