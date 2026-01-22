import warnings
import logging
from typing import Optional, TYPE_CHECKING, List, Tuple
from xml.etree import ElementTree as ET
from ..base_renderer import Renderer, RenderSpecification
from .svgutil import filter_svg_layers
from ...core.vectorization_spec import TraceSpec


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.workpiece import RenderContext


class SvgRenderer(Renderer):
    """Renders SVG data."""

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        """
        Calculates the render specification for an SVG source. This method
        populates the kwargs with `viewbox` or `visible_layer_ids` as needed.
        """
        kwargs = {}
        target_width, target_height = target_size

        # A segment is required for any special SVG handling.
        if segment:
            # Handle layer visibility
            if segment.layer_id:
                kwargs["visible_layer_ids"] = [segment.layer_id]

            # Handle viewbox cropping for direct vector imports
            if segment.crop_window_px:
                # The upscale-then-crop logic of rasters does not apply to
                # vectors. Instead, we pass a `viewbox` to the renderer, but
                # ONLY if it's a direct vector import (PassthroughSpec). If
                # it's a TraceSpec, we must render the full SVG as a bitmap,
                # so we don't pass a viewbox.
                is_vector = not isinstance(
                    segment.vectorization_spec, TraceSpec
                )
                if is_vector:
                    kwargs["viewbox"] = segment.crop_window_px

        return RenderSpecification(
            width=target_width,
            height=target_height,
            data=source_context.data,
            kwargs=kwargs,
            # Vector renders from SVG are pre-masked by their nature;
            # applying a secondary mask based on potentially open-path
            # geometry is incorrect and would hide the content.
            apply_mask=False,
        )

    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        visible_layer_ids: Optional[List[str]] = None,
        viewbox: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders raw SVG data to a pyvips Image by setting its pixel dimensions.
        Expects data to be pre-trimmed for content.
        Can optionally filter by layer IDs if 'visible_layer_ids' is passed.
        Can optionally override the viewBox if 'viewbox' is passed
        (x, y, w, h).
        """
        if not data:
            return None

        render_data = data
        if visible_layer_ids:
            render_data = filter_svg_layers(data, visible_layer_ids)

        if not render_data:
            return None

        try:
            # Modify SVG dimensions for the loader to render at target size
            root = ET.fromstring(render_data)
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            # Allow overriding the viewBox (used for rendering split/cropped
            # vector segments)
            if viewbox:
                vb_x, vb_y, vb_w, vb_h = viewbox
                root.set("viewBox", f"{vb_x} {vb_y} {vb_w} {vb_h}")

            # This causes the content to stretch to fill the width/height
            # instead of scaling proportionally. This is REQUIRED for tracing
            # non-uniformly scaled objects correctly.
            root.set("style", "overflow: visible")

            svg_bytes = ET.tostring(root)
            image = pyvips.Image.svgload_buffer(svg_bytes)
            # logger.debug(
            #    f"SvgRenderer.render_base_image: requested width={width}, "
            #    f"height={height}, actual image width={image.width}, "
            #    f"height={image.height}"
            # )
            return image
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None


SVG_RENDERER = SvgRenderer()
