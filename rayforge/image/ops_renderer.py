import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from ..core.ops import Ops
from ..pipeline.encoder.cairoencoder import CairoEncoder
from ..shared.util.colors import ColorSet
from .base_renderer import Renderer
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384


class OpsRenderer(Renderer):
    """
    Renders vector geometry (Ops/Geometry) to an image.
    """

    def _render_to_cairo_surface(
        self, boundaries: Any, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Internal helper for DXF renderer reuse."""
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
        geo_min_x, geo_min_y, geo_max_x, geo_max_y = boundaries.rect()
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
        render_ops = Ops.from_geometry(boundaries)

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

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        boundaries = kwargs.get("boundaries")
        if not boundaries or boundaries.is_empty():
            return None

        surface = self._render_to_cairo_surface(boundaries, width, height)
        if not surface:
            return None

        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        b, g, r, a = (
            vips_image[0],
            vips_image[1],
            vips_image[2],
            vips_image[3],
        )
        return r.bandjoin([g, b, a]).copy(interpretation="srgb")


OPS_RENDERER = OpsRenderer()
