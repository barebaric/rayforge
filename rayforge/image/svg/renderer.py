import warnings
from typing import Optional, TYPE_CHECKING
from xml.etree import ElementTree as ET
from ..base_renderer import Renderer

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
        """
        if not data:
            return None
        try:
            # Modify SVG dimensions for the loader to render at target size
            root = ET.fromstring(data)
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")
            # Add overflow:visible to render content outside the viewBox.
            # Some designs rely on bezier control points outside the viewbox,
            # which would otherwise be clipped by default.
            root.set("style", "overflow: visible")

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None


SVG_RENDERER = SvgRenderer()
