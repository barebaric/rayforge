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
            **kwargs,
        )

    def draw(self, ctx: cairo.Context):
        """
        Renders the origin symbol. The context is in the element's local
        Y-up space, with (0,0) at the bottom-left corner.
        """
        ctx.save()

        # Set drawing properties
        ctx.set_source_rgba(0.2, 0.8, 0.2, 0.9)  # A distinct green color
        ctx.set_line_width(0.5)  # Use a thin line width in world units (mm)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        # --- Draw Arc ---
        # The arc is drawn in the first quadrant, centered at the
        # element's (0,0)
        radius = self.width * 0.4
        ctx.new_path()
        ctx.arc(0, 0, radius, 0, math.pi / 2)
        ctx.stroke()

        # --- Draw X-Axis with Arrow ---
        axis_len = self.width
        arrow_size = self.width * 0.1
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(axis_len, 0)
        # Arrow head
        ctx.move_to(axis_len, 0)
        ctx.line_to(axis_len - arrow_size, arrow_size)
        ctx.move_to(axis_len, 0)
        ctx.line_to(axis_len - arrow_size, -arrow_size)
        ctx.stroke()

        # --- Draw Y-Axis with Arrow ---
        ctx.new_path()
        ctx.move_to(0, 0)
        ctx.line_to(0, axis_len)
        # Arrow head
        ctx.move_to(0, axis_len)
        ctx.line_to(arrow_size, axis_len - arrow_size)
        ctx.move_to(0, axis_len)
        ctx.line_to(-arrow_size, axis_len - arrow_size)
        ctx.stroke()

        ctx.restore()
