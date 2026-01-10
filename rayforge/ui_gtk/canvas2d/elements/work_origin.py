import math
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
            width=25.0,
            height=25.0,
            selectable=False,
            draggable=False,
            clip=False,  # Allow drawing outside bounds when scaled/flipped
            **kwargs,
        )
        self.x_axis_right = False
        self.y_axis_down = False
        self.x_axis_negative = False
        self.y_axis_negative = False

    def set_orientation(
        self,
        x_axis_right: bool,
        y_axis_down: bool,
        x_axis_negative: bool,
        y_axis_negative: bool,
    ):
        """
        Configures the direction of the arrows based on the machine config.
        """
        if (
            self.x_axis_right == x_axis_right
            and self.y_axis_down == y_axis_down
            and self.x_axis_negative == x_axis_negative
            and self.y_axis_negative == y_axis_negative
        ):
            return

        self.x_axis_right = x_axis_right
        self.y_axis_down = y_axis_down
        self.x_axis_negative = x_axis_negative
        self.y_axis_negative = y_axis_negative

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
        ctx.set_line_width(0.5)  # Use a thin line width in world units (mm)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        # Determine visual direction of Positive X arrow.
        # If Origin is Right (Machine 0 at Right): Values increase Left.
        # (Scale -1)
        # Note: We ignore x_axis_negative here so visual layout is stable.
        scale_x = -1.0 if self.x_axis_right else 1.0

        # Determine visual direction of Positive Y arrow.
        scale_y = -1.0 if self.y_axis_down else 1.0

        ctx.scale(scale_x, scale_y)

        # --- Draw Arc ---
        # The arc is drawn in the first quadrant
        radius = self.width * 0.4
        ctx.new_path()
        ctx.arc(0, 0, radius, 0, math.pi / 2)
        ctx.stroke()

        # --- Draw X-Axis with Arrow ---
        axis_len = self.width
        arrow_size = self.width * 0.15
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(axis_len, 0)
        # Arrow head
        ctx.move_to(axis_len, 0)
        ctx.line_to(axis_len - arrow_size, arrow_size * 0.6)
        ctx.move_to(axis_len, 0)
        ctx.line_to(axis_len - arrow_size, -arrow_size * 0.6)
        ctx.stroke()

        # --- Draw Y-Axis with Arrow ---
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(0, axis_len)
        # Arrow head
        ctx.move_to(0, axis_len)
        ctx.line_to(arrow_size * 0.6, axis_len - arrow_size)
        ctx.move_to(0, axis_len)
        ctx.line_to(-arrow_size * 0.6, axis_len - arrow_size)
        ctx.stroke()

        ctx.restore()
