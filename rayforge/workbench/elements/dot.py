import math
import logging
import cairo
from ..canvas import CanvasElement


logger = logging.getLogger(__name__)


class DotElement(CanvasElement):
    """
    Draws a simple red dot.
    """
    def __init__(self, x, y, diameter: float = 5.0, **kwargs):
        """
        Initializes a DotElement with pixel dimensions.

        Args:
            x: The x-coordinate (pixel) relative to the parent.
            y: The y-coordinate (pixel) relative to the parent.
            radius: The radius (pixel).
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        # Laser dot is always a circle, so width and height should be equal.
        # We store the radius in mm for rendering purposes.
        super().__init__(x,
                         y,
                         diameter,
                         diameter,
                         visible=True,
                         selectable=False,
                         **kwargs)

    def draw(self, ctx: cairo.Context):
        assert self.canvas, "Canvas must be set before drawing"

        """Renders the dot to the element's surface."""
        # Clear the surface with the background color.
        super().draw(ctx)

        # Prepare the context.
        ctx.set_hairline(True)
        ctx.set_source_rgb(.9, 0, 0)

        # Draw the circle centered within the element's pixel bounds
        center_x = self.width / 2
        center_y = self.height / 2
        ctx.arc(center_x, center_y, self.width/2, 0., 2*math.pi)
        ctx.fill()
