import math
import logging
import cairo
from .surfaceelem import SurfaceElement


logger = logging.getLogger(__name__)


class DotElement(SurfaceElement):
    """
    Draws a simple red dot.
    """
    def __init__(self, x, y, width, height, **kwargs):
        """
        Initializes a DotElement with pixel dimensions.

        Args:
            x: The x-coordinate (pixel) relative to the parent.
            y: The y-coordinate (pixel) relative to the parent.
            width: The width (pixel).
            height: The height (pixel).
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        # Laser dot is always a circle, so width and height should be equal.
        # We store the radius in mm for rendering purposes.
        self.radius_mm = 1.0  # Default radius in mm
        super().__init__(x,
                         y,
                         width,
                         height,
                         visible=True,
                         selectable=False,
                         **kwargs)

    def render(
        self,
        clip: tuple[float, float, float, float] | None = None,
        force: bool = False,
    ):
        """Renders the dot to the element's surface."""
        if not self.dirty and not force:
            return
        if not self.canvas or not self.parent or self.surface is None:
            return

        # Clear the surface.
        clip = clip or self.rect()
        self.clear_surface(clip)

        # Prepare the context.
        ctx = cairo.Context(self.surface)
        ctx.set_hairline(True)
        ctx.set_source_rgb(.9, 0, 0)

        # Calculate radius in pixels based on the stored mm radius
        radius_px = self.radius_mm * self.canvas.pixels_per_mm_x

        # Draw the circle centered within the element's pixel bounds
        center_x = self.width / 2
        center_y = self.height / 2
        ctx.arc(center_x, center_y, radius_px, 0., 2*math.pi)
        ctx.fill()
