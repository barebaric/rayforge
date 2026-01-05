import math
import logging
import cairo
from ...canvas import CanvasElement


logger = logging.getLogger(__name__)


class DotElement(CanvasElement):
    """
    Draws a simple red dot. The dot has a constant size in its local
    coordinate space.
    """

    def __init__(self, x, y, diameter: float = 5.0, **kwargs):
        """
        Initializes a DotElement.

        The dimensions (x, y, diameter) are in the parent's coordinate
        system. For WorkSurface, this is typically millimeters.

        Args:
            x: The x-coordinate relative to the parent.
            y: The y-coordinate relative to the parent.
            diameter: The diameter of the dot.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        # Laser dot is always a circle, so width and height should be equal.
        super().__init__(
            x,
            y,
            diameter,
            diameter,
            visible=True,
            selectable=False,
            **kwargs,
        )

    def draw(self, ctx: cairo.Context):
        """Renders the dot onto the provided cairo context."""
        # Let the parent draw its background if any.
        super().draw(ctx)

        # Prepare the context for our drawing.
        ctx.set_source_rgb(0.9, 0, 0)

        # Draw the circle centered within the element's local bounds.
        center_x = self.width / 2
        center_y = self.height / 2
        radius = self.width / 2
        ctx.arc(center_x, center_y, radius, 0.0, 2 * math.pi)
        ctx.fill()
