from typing import Optional, TYPE_CHECKING
import logging
from ..base_renderer import Renderer
from ..ops_renderer import OPS_RENDERER
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DxfRenderer(Renderer):
    """
    A renderer for DXF workpieces. Uses OpsRenderer for vector outlines
    and overlays solid fills if present.
    """

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
                "DxfRenderer: No boundaries provided, cannot render."
            )
            return None

        # 1. Render vector outlines using OpsRenderer
        logger.debug("DxfRenderer: Rendering vector outlines...")
        surface = OPS_RENDERER._render_to_cairo_surface(
            boundaries, width, height
        )
        if not surface:
            logger.error("DxfRenderer: Failed to render vector outlines.")
            return None

        # 2. Draw solids if present
        source_metadata = kwargs.get("source_metadata")

        if source_metadata:
            solids = source_metadata.get("solids", [])
            if solids:
                import cairo

                logger.debug(
                    f"DxfRenderer: Rendering {len(solids)} solid fills..."
                )
                ctx = cairo.Context(surface)

                # Get the transformation info from the vector rendering step
                geo_min_x, geo_min_y, geo_max_x, geo_max_y = boundaries.rect()
                geo_width = geo_max_x - geo_min_x
                geo_height = geo_max_y - geo_min_y

                if geo_width > 1e-9 and geo_height > 1e-9:
                    scale_x = width / geo_width
                    scale_y = height / geo_height

                    # Apply the same Y-flipping transform as OpsRenderer
                    ctx.save()
                    ctx.translate(-geo_min_x * scale_x, geo_max_y * scale_y)
                    ctx.scale(scale_x, -scale_y)

                    ctx.set_source_rgb(0, 0, 0)  # Black fill
                    for i, solid_points in enumerate(solids):
                        if len(solid_points) < 3:
                            logger.warning(
                                f"Skipping degenerate solid #{i} "
                                f"with < 3 points."
                            )
                            continue

                        p_start = solid_points[0]
                        ctx.move_to(p_start[0], p_start[1])
                        for x, y in solid_points[1:]:
                            ctx.line_to(x, y)
                        ctx.close_path()
                        ctx.fill()
                    ctx.restore()
                else:
                    logger.warning(
                        "Cannot render solids because geometry has zero size."
                    )

        # 3. Convert Cairo surface to PyVips Image
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
        return r.bandjoin([g, b, a])


DXF_RENDERER = DxfRenderer()
