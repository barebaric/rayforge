from typing import Optional, TYPE_CHECKING
from ..base_renderer import Renderer
from ..ops_renderer import OPS_RENDERER
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass


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
            return None

        # 1. Render vector outlines using OpsRenderer
        surface = OPS_RENDERER._render_to_cairo_surface(
            boundaries, width, height
        )
        if not surface:
            return None

        # 2. Draw solids if present
        source_metadata = kwargs.get("source_metadata")
        workpiece_matrix = kwargs.get("workpiece_matrix")

        if source_metadata and workpiece_matrix:
            solids = source_metadata.get("solids", [])
            if solids:
                import cairo

                ctx = cairo.Context(surface)
                ops_min_x, ops_min_y, ops_max_x, ops_max_y = boundaries.rect()
                ops_width = ops_max_x - ops_min_x
                ops_height = ops_max_y - ops_min_y

                if ops_width > 1e-9 and ops_height > 1e-9:
                    scale_x = width / ops_width
                    scale_y = height / ops_height
                    norm_tx, norm_ty = workpiece_matrix.get_translation()

                    ctx.scale(scale_x, scale_y)
                    ctx.translate(-norm_tx, -norm_ty)

                    ctx.set_source_rgb(0, 0, 0)
                    for solid_points in solids:
                        if len(solid_points) < 3:
                            continue
                        ctx.move_to(solid_points[0][0], solid_points[0][1])
                        for x, y in solid_points[1:]:
                            ctx.line_to(x, y)
                        ctx.close_path()
                        ctx.fill()

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
