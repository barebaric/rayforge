import cairo
from typing import Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    pass

from ..core.geo import Geometry
from .base_renderer import Renderer
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

# Cairo has a hard limit on surface dimensions, often 32767.
# We use a slightly more conservative value to be safe.
CAIRO_MAX_DIMENSION = 16384

logger = logging.getLogger(__name__)


class OpsRenderer(Renderer):
    """
    Renders vector geometry (Geometry) to an image.
    """

    def _render_to_cairo_surface(
        self, boundaries: Geometry, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Internal helper for renderer reuse."""
        render_width, render_height = width, height
        if render_width <= 0 or render_height <= 0:
            logger.warning(
                f"OpsRenderer received invalid dimensions: {width}x{height}. "
                "Cannot render."
            )
            return None

        logger.debug(
            f"OpsRenderer: Rendering to Cairo surface of "
            f"{render_width}x{render_height} px."
        )

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
            logger.warning(
                "Requested render size exceeds Cairo limit. "
                f"Downscaling to {render_width}x{render_height}."
            )

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, render_width, render_height
        )
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(0, 0, 0, 0)  # Transparent background
        ctx.paint()

        # Calculate scaling to fit the workpiece's local geometry into
        # the surface
        geo_min_x, geo_min_y, geo_max_x, geo_max_y = boundaries.rect()
        geo_width = geo_max_x - geo_min_x
        geo_height = geo_max_y - geo_min_y

        if geo_width <= 1e-9 or geo_height <= 1e-9:
            logger.warning(
                "Geometry has zero size. Returning transparent surface."
            )
            return surface  # Return transparent surface if no size

        scale_x = render_width / geo_width
        scale_y = render_height / geo_height

        # Render directly from Geometry to support Beziers and avoid
        # intermediate linearization in Ops.
        ctx.save()

        # Transform Logic:
        # Map Geometry box (min_x, min_y, max_x, max_y) to Surface
        # (0, 0, w, h).
        # Surface (0,0) is Top-Left.
        # Geometry is Cartesian (Y-Up).
        # PixelX = (GeoX - min_x) * scale_x
        # PixelY = (max_y - GeoY) * scale_y
        #
        # Translate(-min_x*scale_x, max_y*scale_y) -> Scale(scale_x, -scale_y)
        ctx.translate(-geo_min_x * scale_x, geo_max_y * scale_y)
        ctx.scale(scale_x, -scale_y)

        # Set style
        ctx.set_source_rgb(0, 0, 0)  # Black lines

        # Try to use hairlines for crisp rendering independent of scale
        try:
            ctx.set_hairline(True)
        except AttributeError:
            # Fallback for older cairo
            lw = 1.0 / max(scale_x, scale_y)
            ctx.set_line_width(lw)
            logger.debug(f"Using fallback line width: {lw}")

        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

        boundaries.to_cairo(ctx)
        ctx.stroke()
        logger.debug("Stroked geometry path to Cairo context.")

        ctx.restore()

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
            logger.warning(
                "OpsRenderer: No boundaries provided or boundaries are empty."
            )
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
