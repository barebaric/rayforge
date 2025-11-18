import cairo
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any

from ..ops_renderer import OPS_RENDERER
from ..base_renderer import Renderer
from ...core.geo import Geometry
from ...core.matrix import Matrix

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...core.source_asset_segment import SourceAssetSegment
    import pyvips


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

    def _render_to_pixels_internal(
        self,
        surface: cairo.ImageSurface,
        *,
        boundaries: Geometry,
        source_metadata: Dict[str, Any],
        workpiece_matrix: Matrix,
        width: int,
        height: int,
    ) -> cairo.ImageSurface:
        """Internal rendering logic, decoupled from the WorkPiece object."""
        solids = source_metadata.get("solids", [])
        if not solids:
            return surface

        ctx = cairo.Context(surface)
        # We need to apply the same transformation that
        # OPS_RENDERER used to draw the outlines, so the fills
        # align perfectly.
        ops_min_x, ops_min_y, ops_max_x, ops_max_y = boundaries.rect()
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
            norm_tx, norm_ty = workpiece_matrix.get_translation()

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

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        # First, render the outlines from geometry using the standard
        # OPS_RENDERER. This gives us a surface with the correct dimensions
        # and transformations for the outlines.
        surface = OPS_RENDERER.render_to_pixels(workpiece, width, height)
        if not surface or not workpiece.boundaries:
            return None

        # Now, check the ImportSource metadata for filled solids.
        source = workpiece.source
        if source and source.metadata:
            return self._render_to_pixels_internal(
                surface,
                boundaries=workpiece.boundaries,
                source_metadata=source.metadata,
                workpiece_matrix=workpiece.matrix,
                width=width,
                height=height,
            )
        return surface

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        return OPS_RENDERER.get_natural_size_from_data(
            render_data=render_data,
            source_segment=source_segment,
            source_metadata=source_metadata,
            boundaries=boundaries,
            current_size=current_size,
        )

    def render_from_data(
        self,
        *,
        render_data: Optional[bytes],
        original_data: Optional[bytes] = None,
        source_segment: Optional["SourceAssetSegment"] = None,
        source_px_dims: Optional[Tuple[int, int]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        boundaries: Optional["Geometry"] = None,
        workpiece_matrix: Optional["Matrix"] = None,
        width: int,
        height: int,
    ) -> Optional["pyvips.Image"]:
        if not boundaries or boundaries.is_empty():
            return None

        surface = OPS_RENDERER._render_to_pixels_internal(
            boundaries=boundaries, width=width, height=height
        )
        if not surface:
            return None

        if source_metadata and workpiece_matrix:
            surface = self._render_to_pixels_internal(
                surface,
                boundaries=boundaries,
                source_metadata=source_metadata,
                workpiece_matrix=workpiece_matrix,
                width=width,
                height=height,
            )

        import pyvips

        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        b, g, r, a = vips_image[0], vips_image[1], vips_image[2], vips_image[3]
        return r.bandjoin([g, b, a])


DXF_RENDERER = DxfRenderer()
