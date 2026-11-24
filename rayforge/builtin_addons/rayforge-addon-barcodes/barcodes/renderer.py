import logging
import warnings
from typing import TYPE_CHECKING, Optional

from raygeo.formats.svg import geometry_to_svg_path

from rayforge.image.base_renderer import Renderer, RenderSpecification
from rayforge.image.svg.svg_fallback import (
    SVG_LOAD_AVAILABLE,
    cairo_surface_to_vips,
    render_svg_to_cairo,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from rayforge.core.source_asset_segment import SourceAssetSegment
    from rayforge.core.workpiece import RenderContext
    from rayforge.image.structures import FillRenderData

logger = logging.getLogger(__name__)


class BarcodeRenderer(Renderer):
    """
    Renders a barcode's filled geometry as an in-memory SVG.

    Barcodes are drawn from solid fills; when a provider returns no fill
    (for example for the invalid-content placeholder), the stroke
    geometry is outlined instead.
    """

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        kwargs = {
            "boundaries": source_context.boundaries,
            "fills": source_context.fills,
        }
        return RenderSpecification(
            width=target_size[0],
            height=target_size[1],
            data=source_context.data,
            kwargs=kwargs,
            apply_mask=False,
        )

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> pyvips.Image | None:
        boundaries = kwargs.get("boundaries")
        fills = kwargs.get("fills")

        if not boundaries and not fills:
            return None

        svg_parts = [
            (
                f'<svg width="{width}" height="{height}" '
                'xmlns="http://www.w3.org/2000/svg">'
            )
        ]

        if fills:
            for fill in fills:
                path = geometry_to_svg_path(fill.geometry, width, height)
                if path:
                    svg_parts.append(self._fill_to_svg(fill, path))
        elif boundaries:
            path = geometry_to_svg_path(boundaries, width, height)
            if path:
                svg_parts.append(
                    f'<path d="{path}" fill="none" stroke="black" '
                    'stroke-width="1" />'
                )

        svg_parts.append("</svg>")
        svg_bytes = "".join(svg_parts).encode("utf-8")

        try:
            if SVG_LOAD_AVAILABLE:
                return pyvips.Image.svgload_buffer(svg_bytes)
            surface = render_svg_to_cairo(svg_bytes, width, height)
            if not surface:
                logger.error("Failed to render barcode SVG with Cairo.")
                return None
            return cairo_surface_to_vips(surface)
        except pyvips.Error as exc:
            logger.error(f"Failed to render barcode SVG with Vips: {exc}")
            return None

    def _fill_to_svg(self, fill: "FillRenderData", path: str) -> str:
        red = int(fill.color[0] * 255)
        green = int(fill.color[1] * 255)
        blue = int(fill.color[2] * 255)
        opacity = fill.color[3]
        return (
            f'<path d="{path}" fill="rgb({red},{green},{blue})" '
            f'fill-opacity="{opacity}" stroke="none" />'
        )


BARCODE_RENDERER = BarcodeRenderer()
