import warnings
from typing import Optional, TYPE_CHECKING, List, Tuple
from xml.etree import ElementTree as ET
from ..base_renderer import Renderer
from .svgutil import filter_svg_layers

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    pass


class SvgRenderer(Renderer):
    """Renders SVG data."""

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

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None


SVG_RENDERER = SvgRenderer()
