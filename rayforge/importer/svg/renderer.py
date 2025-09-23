import cairo
import warnings
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer


class SvgRenderer(Renderer):
    """Renders SVG data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the trimmed natural size from pre-calculated metadata.
        """
        source = workpiece.source
        if not source or not source.metadata:
            return None

        metadata = source.metadata
        w = metadata.get("trimmed_width_mm")
        h = metadata.get("trimmed_height_mm")

        if w is None or h is None:
            return None

        return w, h

    def _render_vips_from_data(
        self, data: bytes, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        Renders raw SVG data to a pyvips Image by setting its pixel dimensions.
        Expects data to be pre-trimmed for content.
        """
        if not data:
            return None
        try:
            root = ET.fromstring(data)
            root.set("width", f"{width}px")
            root.set("height", f"{height}px")
            root.set("preserveAspectRatio", "none")

            return pyvips.Image.svgload_buffer(ET.tostring(root))
        except (pyvips.Error, ET.ParseError, ValueError, TypeError):
            return None

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        # Use the workpiece's data property, which correctly handles both
        # attached (via .source) and transient/subprocess instances.
        data_to_render = workpiece.data
        if not data_to_render:
            return None

        return self._render_vips_from_data(data_to_render, width, height)

    def render_to_pixels_from_data(
        self, data: bytes, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """Renders raw SVG data directly to a Cairo surface."""
        if not data:
            return None
        final_image = self._render_vips_from_data(data, width, height)
        if not final_image:
            return None

        if final_image.bands < 4:
            final_image = final_image.bandjoin(255)

        # Vips RGBA -> Cairo BGRA
        b, g, r, a = (
            final_image[2],
            final_image[1],
            final_image[0],
            final_image[3],
        )
        bgra_image = b.bandjoin([g, r, a])
        mem_buffer = bgra_image.write_to_memory()

        return cairo.ImageSurface.create_for_data(
            mem_buffer,
            cairo.FORMAT_ARGB32,
            final_image.width,
            final_image.height,
            final_image.width * 4,
        )

    def render_to_pixels(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        final_image = self.get_or_create_vips_image(workpiece, width, height)
        if not final_image:
            return None

        if final_image.bands < 4:
            final_image = final_image.bandjoin(255)

        # Vips RGBA -> Cairo BGRA
        b, g, r, a = (
            final_image[2],
            final_image[1],
            final_image[0],
            final_image[3],
        )
        bgra_image = b.bandjoin([g, r, a])
        mem_buffer = bgra_image.write_to_memory()

        return cairo.ImageSurface.create_for_data(
            mem_buffer,
            cairo.FORMAT_ARGB32,
            final_image.width,
            final_image.height,
            final_image.width * 4,
        )


SVG_RENDERER = SvgRenderer()
