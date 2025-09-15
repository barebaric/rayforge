import cairo
import math
from typing import TYPE_CHECKING
from gi.repository import Gdk
from ..canvas.element import CanvasElement
from ...core.tab import Tab

if TYPE_CHECKING:
    from .workpiece import WorkPieceView


class TabHandleElement(CanvasElement):
    """
    A canvas element representing a single Tab, which can be visible
    during regular viewing or draggable during edit mode.
    """

    def __init__(self, tab_data: Tab, parent: "WorkPieceView"):
        super().__init__(
            x=0,
            y=0,
            width=1.0,  # A unit square, scaled by the transform
            height=1.0,
            data=tab_data,
            parent=parent,
            selectable=False,  # Not part of standard selection
            clip=False,  # Allow border to draw outside bounds
        )
        self.draggable = False  # Controlled by WorkPieceView
        self.opacity = 1.0  # Controlled by WorkPieceView

    def draw(self, ctx: cairo.Context):
        """Draws the tab handle as a themed, rounded rectangle."""
        if not self.canvas:
            return

        # The context is already transformed into this element's local 1x1
        # space.
        style_context = self.canvas.get_style_context()

        # Define fallback RGBA colors in case theme lookup fails
        fallback_bg = Gdk.RGBA(red=0.2, green=0.7, blue=0.3, alpha=0.8)
        fallback_border = Gdk.RGBA(red=0.1, green=0.5, blue=0.2, alpha=0.9)
        fallback_hover_bg = Gdk.RGBA(red=0.2, green=0.5, blue=0.9, alpha=0.9)
        fallback_hover_border = Gdk.RGBA(
            red=0.1, green=0.3, blue=0.7, alpha=1.0
        )

        if self.is_hovered:
            # Use the theme's main highlight/accent color for hover
            found, bg_color = style_context.lookup_color("accent_bg_color")
            if not found:
                bg_color = fallback_hover_bg

            # A dark foreground color makes a good border
            found, border_color = style_context.lookup_color("theme_fg_color")
            if not found:
                border_color = fallback_hover_border
        else:
            # Use the theme's success color for a visible, non-urgent handle
            found, bg_color = style_context.lookup_color("success_bg_color")
            if not found:
                bg_color = fallback_bg

            found, border_color = style_context.lookup_color("theme_fg_color")
            if not found:
                border_color = fallback_border

        # We draw inside a 1x1 unit square
        w, h = 1.0, 1.0
        radius = 0.2  # Relative radius

        ctx.new_sub_path()
        ctx.arc(radius, radius, radius, math.pi, 1.5 * math.pi)
        ctx.arc(w - radius, radius, radius, 1.5 * math.pi, 2 * math.pi)
        ctx.arc(w - radius, h - radius, radius, 0, 0.5 * math.pi)
        ctx.arc(radius, h - radius, radius, 0.5 * math.pi, math.pi)
        ctx.close_path()

        ctx.set_source_rgba(
            bg_color.red,
            bg_color.green,
            bg_color.blue,
            bg_color.alpha * self.opacity,
        )
        ctx.fill_preserve()

        # The line width needs to be ~1px in screen space. The CTM includes
        # the scale. We set line width in local user units (0-1).
        # A scale of S means 1 user unit = S pixels. So 1px = 1/S user units.
        world_transform = self.get_world_transform()
        scale_x, scale_y = world_transform.get_abs_scale()
        avg_scale = (scale_x + scale_y) / 2.0
        line_width = 1.0 / avg_scale if avg_scale > 0 else 1.0

        ctx.set_line_width(line_width)
        ctx.set_source_rgba(
            border_color.red,
            border_color.green,
            border_color.blue,
            border_color.alpha * self.opacity,
        )
        ctx.stroke()
