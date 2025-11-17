import cairo
import warnings
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util
from .svgutil import get_natural_size as get_size_from_data


class SvgRenderer(Renderer):
    """Renders SVG data from a WorkPiece."""

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural size from metadata or by parsing the data.
        """
        # For traced/cropped items, the authoritative size is in the config.
        if config := workpiece.source_segment:
            w = config.cropped_width_mm
            h = config.cropped_height_mm
            if w is not None and h is not None:
                return w, h

        # For passthrough items, use the trimmed SVG dimensions from metadata.
        source = workpiece.source
        if source and source.metadata:
            w = source.metadata.get("trimmed_width_mm")
            h = source.metadata.get("trimmed_height_mm")
            if w is not None and h is not None:
                return w, h

        # Fallback for transient workpieces (like in the import dialog)
        if workpiece.data:
            return get_size_from_data(workpiece.data)

        return None

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
        config = workpiece.source_segment
        source = workpiece.source
        # Path 1: Traced SVG. Render the original data, then crop.
        if (
            config
            and config.crop_window_px
            and source
            and source.width_px
            and source.height_px
        ):
            full_render_data = workpiece.original_data
            if not full_render_data:
                return None

            full_image = self._render_vips_from_data(
                full_render_data,
                source.width_px,
                source.height_px,
            )
            if not full_image:
                return None

            x, y, w, h = map(int, config.crop_window_px)
            image_to_process = image_util.safe_crop(full_image, x, y, w, h)
            if image_to_process is None:
                return pyvips.Image.black(width, height, bands=4)

        # Path 2: Passthrough SVG. Render the workpiece's data directly.
        else:
            data_to_render = workpiece.data
            if not data_to_render:
                return None
            image_to_process = self._render_vips_from_data(
                data_to_render, width, height
            )

        if not image_to_process:
            return None

        # Apply mask (which is now normalized)
        if config:
            mask_geo = config.segment_mask_geometry
            masked_image = image_util.apply_mask_to_vips_image(
                image_to_process, mask_geo
            )
            if masked_image:
                image_to_process = masked_image

        if image_to_process.width == 0 or image_to_process.height == 0:
            return image_to_process

        # Resize the result to the final requested dimensions if needed.
        if (
            image_to_process.width != width
            or image_to_process.height != height
        ):
            h_scale = width / image_to_process.width
            v_scale = height / image_to_process.height
            return image_to_process.resize(h_scale, vscale=v_scale)

        return image_to_process

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
