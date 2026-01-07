import warnings
from typing import Optional, TYPE_CHECKING, List
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
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders raw SVG data to a pyvips Image by setting its pixel dimensions.
        Expects data to be pre-trimmed for content.
        Can optionally filter by layer IDs if 'visible_layer_ids' is passed.
        """
        if not data:
            return None

        render_data = data
        visible_layer_ids: Optional[List[str]] = kwargs.get(
            "visible_layer_ids"
        )
        if visible_layer_ids:
            render_data = filter_svg_layers(data, visible_layer_ids)

        if not render_data:
            return None

        try:
            # Modify SVG dimensions for the loader to render at target size
            root = ET.fromstring(render_data)
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            # REMOVED: root.set("preserveAspectRatio", "none")
            # This was causing the content to stretch to fill the width/height
            # instead of scaling proportionally. Default behavior is correct.
            root.set("style", "overflow: visible")

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None


SVG_RENDERER = SvgRenderer()
