import cairo
from typing import Optional, TYPE_CHECKING, Tuple
from ..shared.ops_renderer import OPS_RENDERER
from ..base_renderer import Renderer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class DxfRenderer(Renderer):
    """
    A renderer for DXF workpieces that can contain both vector operations
    (geometry) for toolpaths and special data (workpiece.data) for filled
    shapes like SOLID entities.
    """

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        The natural size is determined by the bounding box of the toolpaths
        (geometry), as this represents the machinable area.
        """

        return OPS_RENDERER.get_natural_size(workpiece)

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        # First, render the outlines from geometry using the standard
        # OPS_RENDERER. This gives us a surface with the correct dimensions
        # and transformations for the outlines.
        surface = OPS_RENDERER.render_to_pixels(workpiece, width, height)
        if not surface or not workpiece.vectors:
            return None

        # Now, check the ImportSource metadata for filled solids.
        source = workpiece.source
        if source and source.metadata:
            solids = source.metadata.get("solids", [])
            if solids:
                ctx = cairo.Context(surface)
                # We need to apply the same transformation that
                # OPS_RENDERER used to draw the outlines, so the fills
                # align perfectly.
                ops_min_x, ops_min_y, ops_max_x, ops_max_y = (
                    workpiece.vectors.rect()
                )
                ops_width = ops_max_x - ops_min_x
                ops_height = ops_max_y - ops_min_y

                if ops_width > 1e-9 and ops_height > 1e-9:
                    scale_x = width / ops_width
                    scale_y = height / ops_height

                    # The importer stores the normalization offset (the
                    # top-left corner of the un-normalized geometry) in
                    # the workpiece's translation.
                    # We need this to align the un-normalized
                    # solid points with the normalized vector outlines.
                    norm_tx, norm_ty = workpiece.matrix.get_translation()

                    # Apply the full transform: scale, then translate.
                    # This maps the original, un-normalized coordinates
                    # of the solids to the correct pixel locations.
                    ctx.scale(scale_x, scale_y)
                    ctx.translate(-norm_tx, -norm_ty)

                    # Draw each solid as a filled black polygon
                    ctx.set_source_rgb(0, 0, 0)
                    for solid_points in solids:
                        if len(solid_points) < 3:
                            continue
                        # Draw with the original, un-normalized coordinates.
                        # The context's transform will handle the math.
                        ctx.move_to(
                            solid_points[0][0],
                            solid_points[0][1],
                        )
                        for x, y in solid_points[1:]:
                            ctx.line_to(x, y)
                        ctx.close_path()
                        ctx.fill()
        return surface


DXF_RENDERER = DxfRenderer()
