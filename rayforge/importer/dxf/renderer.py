import cairo
import json
from typing import Optional, TYPE_CHECKING, Tuple
from ..shared.ops_renderer import OPS_RENDERER
from ..base_renderer import Renderer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class DxfRenderer(Renderer):
    """
    A renderer for DXF workpieces that can contain both vector operations
    (source_ops) for toolpaths and special data (workpiece.data) for filled
    shapes like SOLID entities.
    """

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        The natural size is determined by the bounding box of the toolpaths
        (source_ops), as this represents the machinable area.
        """
        return OPS_RENDERER.get_natural_size(workpiece)

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        # First, render the outlines from source_ops using the standard
        # OPS_RENDERER. This gives us a surface with the correct dimensions
        # and transformations for the outlines.
        surface = OPS_RENDERER.render_to_pixels(workpiece, width, height)
        if not surface or not workpiece.source_ops:
            return None

        # Now, check for special renderer data (for filled solids)
        if workpiece.data:
            try:
                # Deserialize the solid data
                payload = json.loads(workpiece.data.decode("utf-8"))
                solids = payload.get("solids", [])

                if solids:
                    ctx = cairo.Context(surface)
                    # We need to apply the same transformation that
                    # OPS_RENDERER used to draw the outlines, so the fills
                    # align perfectly.
                    ops_min_x, ops_min_y, ops_max_x, ops_max_y = (
                        workpiece.source_ops.rect()
                    )
                    ops_width = ops_max_x - ops_min_x
                    ops_height = ops_max_y - ops_min_y

                    if ops_width > 1e-9 and ops_height > 1e-9:
                        scale_x = width / ops_width
                        scale_y = height / ops_height
                        ctx.translate(
                            -ops_min_x * scale_x, -ops_min_y * scale_y
                        )

                        # Draw each solid as a filled black polygon
                        ctx.set_source_rgb(0, 0, 0)
                        for solid_points in solids:
                            if len(solid_points) < 3:
                                continue
                            ctx.move_to(
                                solid_points[0][0] * scale_x,
                                solid_points[0][1] * scale_y,
                            )
                            for x, y in solid_points[1:]:
                                ctx.line_to(x * scale_x, y * scale_y)
                            ctx.close_path()
                            ctx.fill()
            except (json.JSONDecodeError, KeyError, TypeError):
                # Ignore malformed data
                pass

        return surface


DXF_RENDERER = DxfRenderer()
