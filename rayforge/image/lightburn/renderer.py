import logging
import warnings
from typing import TYPE_CHECKING, Optional, Tuple
from xml.etree import ElementTree as ET

from ..base_renderer import Renderer, RenderSpecification
from ..ops_renderer import OPS_RENDERER
from ..svg.svg_fallback import (
    SVG_LOAD_AVAILABLE,
    cairo_surface_to_vips,
    render_svg_to_cairo,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.workpiece import RenderContext
    from ...image.structures import ImportResult

logger = logging.getLogger(__name__)


class LightBurnRenderer(Renderer):
    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        kwargs = {
            "boundaries": source_context.boundaries,
        }
        return RenderSpecification(
            width=target_size[0],
            height=target_size[1],
            data=source_context.data,
            kwargs=kwargs,
            apply_mask=False,
        )

    def render_preview_image(
        self,
        import_result: "ImportResult",
        target_width: int,
        target_height: int,
    ) -> Optional[pyvips.Image]:
        if not import_result.payload:
            return None

        source = import_result.payload.source
        data_to_render = source.base_render_data or source.original_data
        if not data_to_render:
            return None

        return self.render_base_image(
            data=data_to_render, width=target_width, height=target_height
        )

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        if data and data.startswith(b"<svg"):
            return self._render_svg_image(data, width, height)

        boundaries = kwargs.get("boundaries")
        if not boundaries or boundaries.is_empty():
            logger.warning("LightBurnRenderer: No boundaries or SVG data.")
            return None

        surface = OPS_RENDERER._render_to_cairo_surface(
            boundaries, width, height
        )
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
        return r.bandjoin([g, b, a])

    def _render_svg_image(
        self, svg_data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        try:
            root = ET.fromstring(svg_data)
        except ET.ParseError:
            return None
        root.set("width", f"{width}px")
        root.set("height", f"{height}px")
        root.set("preserveAspectRatio", "xMidYMid meet")
        svg_bytes = ET.tostring(root)
        try:
            if SVG_LOAD_AVAILABLE:
                return pyvips.Image.svgload_buffer(svg_bytes)
            surface = render_svg_to_cairo(svg_bytes, width, height)
            if surface:
                return cairo_surface_to_vips(surface)
        except (pyvips.Error, Exception):
            return None
        return None


LIGHTBURN_RENDERER = LightBurnRenderer()
