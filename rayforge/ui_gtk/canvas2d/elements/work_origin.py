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
        self.x_axis_right = False
        self.y_axis_down = False

    def set_orientation(
        self,
        x_axis_right: bool,
        y_axis_down: bool,
    ):
        """
        Configures the direction of the arrows based on the machine config.
        """
        if (
            self.x_axis_right == x_axis_right
            and self.y_axis_down == y_axis_down
        ):
            return

        self.x_axis_right = x_axis_right
        self.y_axis_down = y_axis_down

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

        # Determine visual direction of Positive X arrow.
        # If Origin is Right (Machine 0 at Right): Values increase Left.
        # (Scale -1)
        scale_x = -1.0 if self.x_axis_right else 1.0

        # Determine visual direction of Positive Y arrow.
        scale_y = -1.0 if self.y_axis_down else 1.0

        ctx.scale(scale_x, scale_y)

        # --- Draw X-Axis with Arrow ---
        axis_len = self.width
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(axis_len, 0)
        ctx.stroke()

        # --- Draw Y-Axis with Arrow ---
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(0, axis_len)
        ctx.stroke()

        ctx.restore()
