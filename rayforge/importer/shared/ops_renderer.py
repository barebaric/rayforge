import cairo
from typing import Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece

from ...pipeline.encoder.cairoencoder import CairoEncoder
from ..base_renderer import Renderer

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384


class OpsRenderer(Renderer):
    """
    A stateless, shared renderer for any WorkPiece that contains vector
    data in its `source_ops` attribute. It uses the CairoEncoder to draw
    the ops.
    """

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        For vector ops, the natural size is the bounding box of the geometry.
        """
        if not workpiece.source_ops or workpiece.source_ops.is_empty():
            return None

        min_x, min_y, max_x, max_y = workpiece.source_ops.rect()
        width = max_x - min_x
        height = max_y - min_y
        return width, height

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        if not workpiece.source_ops or workpiece.source_ops.is_empty():
            return None

        render_width, render_height = width, height
        if render_width <= 0 or render_height <= 0:
            return None

        # Downscale if requested size exceeds Cairo's limit
        if (
            render_width > CAIRO_MAX_DIMENSION
            or render_height > CAIRO_MAX_DIMENSION
        ):
            scale_factor = 1.0
            if render_width > CAIRO_MAX_DIMENSION:
                scale_factor = CAIRO_MAX_DIMENSION / render_width
            if render_height > CAIRO_MAX_DIMENSION:
                scale_factor = min(
                    scale_factor, CAIRO_MAX_DIMENSION / render_height
                )
            render_width = max(1, int(render_width * scale_factor))
            render_height = max(1, int(render_height * scale_factor))

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, render_width, render_height
        )
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(0, 0, 0, 0)  # Transparent background
        ctx.paint()
        ctx.set_source_rgb(0, 0, 0)  # Black lines

        # Calculate scaling to fit the workpiece's local ops into the surface
        ops_min_x, ops_min_y, ops_max_x, ops_max_y = (
            workpiece.source_ops.rect()
        )
        ops_width = ops_max_x - ops_min_x
        ops_height = ops_max_y - ops_min_y

        if ops_width <= 1e-9 or ops_height <= 1e-9:
            return surface  # Return transparent surface if ops have no size

        scale_x = render_width / ops_width
        scale_y = render_height / ops_height

        # Translate the ops so their top-left corner is at the origin
        ctx.translate(-ops_min_x * scale_x, -ops_min_y * scale_y)

        encoder = CairoEncoder()
        encoder.encode(
            ops=workpiece.source_ops,
            ctx=ctx,
            scale=(scale_x, scale_y),
            cut_color=(0, 0, 0),
        )

        return surface


# A shared, stateless singleton instance of the renderer.
OPS_RENDERER = OpsRenderer()
