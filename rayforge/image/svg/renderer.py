import warnings
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any
from xml.etree import ElementTree as ET
from ..base_renderer import Renderer
from .svgutil import get_natural_size as get_size_from_data

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ...core.geo import Geometry
    from ...core.source_asset_segment import SourceAssetSegment


class SvgRenderer(Renderer):
    """Renders SVG data."""

    def get_natural_size_from_data(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
        boundaries: Optional["Geometry"] = None,
        current_size: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        # For traced/cropped items, the authoritative size is in the config.
        if source_segment:
            w = source_segment.cropped_width_mm
            h = source_segment.cropped_height_mm
            if w is not None and h is not None:
                return w, h

        # For passthrough items, use the trimmed SVG dimensions from metadata.
        if source_metadata:
            w = source_metadata.get("trimmed_width_mm")
            h = source_metadata.get("trimmed_height_mm")
            if w is not None and h is not None:
                return w, h

        # Fallback for transient workpieces (like in the import dialog)
        if render_data:
            return get_size_from_data(render_data)

        return None

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

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None


SVG_RENDERER = SvgRenderer()
