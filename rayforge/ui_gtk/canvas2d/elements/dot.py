import logging
import math

import cairo

from ...canvas import CanvasElement

logger = logging.getLogger(__name__)


class DotElement(CanvasElement):
    """
    Draws a simple colored dot, either filled or as a hollow ring. The
    dot has a constant size in its local coordinate space.
    """

    def __init__(
        self,
        x,
        y,
        diameter: float = 5.0,
        color: tuple[float, float, float] = (0.9, 0, 0),
        filled: bool = True,
        **kwargs,
    ):
        """
        Initializes a DotElement.

        The dimensions (x, y, diameter) are in the parent's coordinate
        system. For WorkSurface, this is typically millimeters.

        Args:
            x: The x-coordinate relative to the parent.
            y: The y-coordinate relative to the parent.
            diameter: The diameter of the dot.
            color: The RGB color of the dot (each 0..1).
            filled: True draws a filled disc, False a hollow ring.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        # Laser dots are always circles, so width and height should be
        # equal.
        super().__init__(
            x,
            y,
            diameter,
            diameter,
            visible=True,
            selectable=False,
            **kwargs,
        )
        self._color = color
        self._filled = filled

    def set_filled(self, filled: bool):
        """Switches between a filled disc and a hollow ring."""
        self._filled = filled

    def draw(self, ctx: cairo.Context):
        """Renders the dot onto the provided cairo context."""
        # Let the parent draw its background if any.
        super().draw(ctx)

        # Prepare the context for our drawing.
        ctx.set_source_rgb(*self._color)

        # Draw the circle centered within the element's local bounds.
        center_x = self.width / 2
        center_y = self.height / 2
        if self._filled:
            radius = self.width / 2
            ctx.arc(center_x, center_y, radius, 0.0, 2 * math.pi)
            ctx.fill()
        else:
            # Hollow ring: keep the stroke inside the element bounds.
            line_width = self.width * 0.2
            radius = self.width / 2 - line_width / 2
            ctx.set_line_width(line_width)
            ctx.arc(center_x, center_y, radius, 0.0, 2 * math.pi)
            ctx.stroke()
