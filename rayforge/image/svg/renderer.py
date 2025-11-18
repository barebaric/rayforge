import cairo
import warnings
import math
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any
from xml.etree import ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.workpiece import WorkPiece
from ..base_renderer import Renderer
from .. import image_util
from .svgutil import get_natural_size as get_size_from_data
from ..tracing import VTRACER_PIXEL_LIMIT

if TYPE_CHECKING:
    from ...core.source_asset_segment import SourceAssetSegment
    from ...core.geo import Geometry
    from ...core.matrix import Matrix


class SvgRenderer(Renderer):
    """Renders SVG data from a WorkPiece."""

    def _get_natural_size_internal(
        self,
        *,
        render_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_metadata: Optional[Dict[str, Any]],
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

    def get_natural_size(
        self, workpiece: "WorkPiece"
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the natural size from metadata or by parsing the data.
        """
        source = workpiece.source
        return self._get_natural_size_internal(
            render_data=workpiece.data,
            source_segment=workpiece.source_segment,
            source_metadata=source.metadata if source else None,
        )

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

    def _render_to_vips_internal(
        self,
        *,
        render_data: Optional[bytes],
        original_data: Optional[bytes],
        source_segment: Optional["SourceAssetSegment"],
        source_px_dims: Optional[Tuple[int, int]],
        width: int,
        height: int,
    ) -> Optional[pyvips.Image]:
        config = source_segment
        image_to_process: Optional[pyvips.Image] = None

        # Path 1: Traced SVG. Render the original data, then crop.
        if config and config.crop_window_px:
            # Get the source dimensions, which are critical for this path.
            source_w_px = source_px_dims[0] if source_px_dims else None
            source_h_px = source_px_dims[1] if source_px_dims else None

            # Fallback for when dimensions are not passed (e.g., subprocess)
            if not (source_w_px and source_h_px) and original_data:
                size_mm = get_size_from_data(original_data)
                if size_mm and size_mm[0] and size_mm[1]:
                    w_mm, h_mm = size_mm
                    aspect = w_mm / h_mm if h_mm > 0 else 1.0
                    TARGET_DIM = math.sqrt(VTRACER_PIXEL_LIMIT)
                    if aspect >= 1.0:
                        w_px = int(TARGET_DIM)
                        h_px = int(TARGET_DIM / aspect)
                    else:
                        h_px = int(TARGET_DIM)
                        w_px = int(TARGET_DIM * aspect)
                    source_w_px, source_h_px = max(1, w_px), max(1, h_px)

            if not (source_w_px and source_h_px) or not original_data:
                return pyvips.Image.black(width, height, bands=4)

            full_image = self._render_vips_from_data(
                original_data, source_w_px, source_h_px
            )
            if not full_image:
                return None

            x, y, w, h = map(int, config.crop_window_px)
            image_to_process = image_util.safe_crop(full_image, x, y, w, h)
            if image_to_process is None:
                return pyvips.Image.black(width, height, bands=4)

        # Path 2: Passthrough SVG. Render the workpiece's data directly.
        else:
            data_to_render = render_data
            if not data_to_render:
                return None
            image_to_process = self._render_vips_from_data(
                data_to_render, width, height
            )

        # Guard against failure in either of the paths above.
        if not image_to_process:
            return None

        # Apply mask (which is now normalized)
        if config:
            mask_geo = config.segment_mask_geometry
            # Reassign image_to_process, as masking can fail and return None.
            image_to_process = image_util.apply_mask_to_vips_image(
                image_to_process, mask_geo
            )

        # Add a new guard to safely handle the Optional return from masking.
        if not image_to_process:
            return None

        if image_to_process.width == 0 or image_to_process.height == 0:
            return image_to_process

        # Resize the result to the final requested dimensions if needed.
        if (
            image_to_process.width != width
            or image_to_process.height != height
        ):
            h_scale = (
                width / image_to_process.width
                if image_to_process.width > 0
                else 1
            )
            v_scale = (
                height / image_to_process.height
                if image_to_process.height > 0
                else 1
            )
            return image_to_process.resize(h_scale, vscale=v_scale)

        return image_to_process

    def _render_to_vips_image(
        self, workpiece: "WorkPiece", width: int, height: int
    ) -> Optional[pyvips.Image]:
        source = workpiece.source
        dims = None
        if (
            source
            and source.width_px is not None
            and source.height_px is not None
        ):
            dims = (source.width_px, source.height_px)

        return self._render_to_vips_internal(
            render_data=workpiece.data,
            original_data=workpiece.original_data,
            source_segment=workpiece.source_segment,
            source_px_dims=dims,
            width=width,
            height=height,
        )

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
            render_data=render_data,
            source_segment=source_segment,
            source_metadata=source_metadata,
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
    ) -> Optional[pyvips.Image]:
        return self._render_to_vips_internal(
            render_data=render_data,
            original_data=original_data,
            source_segment=source_segment,
            source_px_dims=source_px_dims,
            width=width,
            height=height,
        )


SVG_RENDERER = SvgRenderer()
