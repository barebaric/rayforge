import cairo
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any

if TYPE_CHECKING:
    from ..core.workpiece import WorkPiece
    from ..core.geo import Geometry
    from ..core.source_asset_segment import SourceAssetSegment
    from ..core.matrix import Matrix
    import pyvips

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

    def _get_natural_size_internal(
        self,
        *,
        boundaries: Optional["Geometry"],
        source_metadata: Optional[Dict[str, Any]],
        current_size: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if not boundaries or boundaries.is_empty():
            return None

        # Prioritize the static, pre-calculated size from the importer.
        if source_metadata:
            natural_size = source_metadata.get("natural_size")
            if natural_size:
                return natural_size

        # Fallback: Use current size
        return current_size

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        For vector geometry, the "natural size" is the intrinsic physical
        size of the content from the original file, in millimeters.

        This implementation prioritizes the 'natural_size' value stored in
        the ImportSource metadata by the importer. This ensures the size is
        static and correct. If not found, it falls back to the workpiece's
        current dynamic size.
        """
        source = workpiece.source
        return self._get_natural_size_internal(
            boundaries=workpiece.boundaries,
            source_metadata=source.metadata if source else None,
            current_size=workpiece.size,
        )

    def _render_to_pixels_internal(
        self, *, boundaries: "Geometry", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
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

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        if not workpiece.boundaries or workpiece.boundaries.is_empty():
            return None
        return self._render_to_pixels_internal(
            boundaries=workpiece.boundaries, width=width, height=height
        )

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        return self._get_natural_size_internal(
            boundaries=boundaries,
            source_metadata=source_metadata,
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

        surface = self._render_to_pixels_internal(
            boundaries=boundaries, width=width, height=height
        )
        if not surface:
            return None

        import pyvips

        h, w = surface.get_height(), surface.get_width()
        vips_image = pyvips.Image.new_from_memory(
            surface.get_data(), w, h, 4, "uchar"
        )
        b, g, r, a = vips_image[0], vips_image[1], vips_image[2], vips_image[3]
        return r.bandjoin([g, b, a])


# A shared, stateless singleton instance of the renderer.
OPS_RENDERER = OpsRenderer()
