import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from ..core.workpiece import WorkPiece

from ..core.ops import Ops
from ..pipeline.encoder.cairoencoder import CairoEncoder
from ..shared.util.colors import ColorSet
from .base_renderer import Renderer

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384


class OpsRenderer(Renderer):
    """
    A stateless, shared renderer for any WorkPiece that contains vector
    data in its `vectors` attribute. It uses the CairoEncoder to draw
    the geometry.
    """

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        For vector geometry, the natural size is the physical size of
        the workpiece as determined by the importer. This is stored in the
        workpiece's matrix.
        """
        if not workpiece.vectors or workpiece.vectors.is_empty():
            return None

        # The .size property reads the physical scale from the matrix.
        # This fulfills the contract of returning the untransformed size in mm.
        return workpiece.size

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        if not workpiece.vectors or workpiece.vectors.is_empty():
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

        # Calculate scaling to fit the workpiece's local geometry into
        # the surface
        geo_min_x, geo_min_y, geo_max_x, geo_max_y = workpiece.vectors.rect()
        geo_width = geo_max_x - geo_min_x
        geo_height = geo_max_y - geo_min_y

        if geo_width <= 1e-9 or geo_height <= 1e-9:
            return surface  # Return transparent surface if no size

        scale_x = render_width / geo_width
        scale_y = render_height / geo_height

        # Translate the geometry so its top-left corner is at the origin
        ctx.translate(-geo_min_x * scale_x, -geo_min_y * scale_y)

        # The CairoEncoder expects an Ops object, so we convert our pure
        # geometry into a temporary Ops object for rendering.
        render_ops = Ops.from_geometry(workpiece.vectors)

        encoder = CairoEncoder()

        # Create a simple ColorSet with black cut color
        cut_lut = np.zeros((256, 4))
        cut_lut[:, 3] = 1.0  # Full alpha

        colors = ColorSet(
            {
                "cut": cut_lut,
                "engrave": cut_lut,  # Use same for engrave
                "travel": (0, 0, 0, 0.0),  # transparent
                "zero_power": (0, 0, 0, 1.0),  # black
            }
        )

        encoder.encode(
            ops=render_ops,
            ctx=ctx,
            scale=(scale_x, scale_y),
            colors=colors,
        )

        return surface


# A shared, stateless singleton instance of the renderer.
OPS_RENDERER = OpsRenderer()
