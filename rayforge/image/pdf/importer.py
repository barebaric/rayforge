import io
import logging
from typing import Optional, Tuple, cast
import cv2
import numpy as np
from pypdf import PdfReader
import math
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from .. import image_util
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface, VTRACER_PIXEL_LIMIT
from ..util import to_mm
from .renderer import PDF_RENDERER

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    """
    Imports vector and raster data from PDF files by tracing their content.

    This importer renders the PDF to a high-resolution bitmap, finds the
    content bounds, crops the PDF to that area, and then traces the result
    to generate vector geometry.
    """

    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)
    _TRACE_PPM = 24.0  # ~600 DPI for tracing
    _MAX_RENDER_DIM = 16384

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Retrieve document items from the PDF file.

        The PDF content will always be auto-cropped, traced, and imported as
        vectors using the provided vectorization specification.

        Args:
            vectorization_spec: Configuration for vector tracing. If None,
              default trace settings will be used.

        Returns:
            An ImportPayload containing the source and a WorkPiece, or None if
            processing fails.
        """
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PDF_RENDERER,
        )

        original_size_mm = self._get_pdf_size(source)
        if not original_size_mm:
            # If size can't be determined, return with just the source.
            return ImportPayload(source=source, items=[])

        # Populate the intrinsic dimensions of the source asset
        w_mm, h_mm = original_size_mm
        source.width_mm = w_mm
        source.height_mm = h_mm

        # We always render the PDF to a bitmap to find the content bounds
        # and to trace.
        render_w_px, render_h_px = self._calculate_render_resolution(
            w_mm, h_mm
        )
        # DPI calculation is handled internally by the renderer based on
        # target dimensions.
        dpi = float(render_w_px / w_mm) * 25.4 if w_mm > 0 else 300.0

        vips_image = PDF_RENDERER.render_base_image(
            source.original_data, width=render_w_px, height=render_h_px
        )
        if not vips_image:
            logger.error("Failed to render PDF to an image for processing.")
            return None

        # Store the rendered image so the preview dialog can use it. This
        # ensures the dialog crops the exact same image the crop_window was
        # calculated from.
        source.base_render_data = vips_image.pngsave_buffer()

        # Update the source asset with the rendered image dimensions
        source.width_px = vips_image.width
        source.height_px = vips_image.height

        # Set resolution metadata on the newly rendered image
        px_per_mm = dpi / 25.4
        vips_image = vips_image.copy(xres=px_per_mm, yres=px_per_mm)

        # Convert to a Cairo surface for tracing or analysis
        norm_image = image_util.normalize_to_rgba(vips_image)
        if not norm_image:
            return None
        surface = image_util.vips_rgba_to_cairo_surface(norm_image)

        spec = vectorization_spec
        if not isinstance(spec, TraceSpec):
            logger.warning(
                "PdfImporter did not receive a TraceSpec, using defaults."
            )
            spec = TraceSpec()

        geometries = trace_surface(surface, spec)
        items = image_util.create_single_workpiece_from_trace(
            geometries,
            source,
            vips_image,
            spec,
            self.source_file.stem,
        )

        return ImportPayload(source=source, items=items)

    def _get_pdf_size(
        self, source: SourceAsset
    ) -> Optional[Tuple[float, float]]:
        """
        Retrieve the natural size of the PDF's first page in millimeters.

        Args:
            source: The SourceAsset containing the PDF data.

        Returns:
            A tuple of (width, height) in millimeters, or None if the size
            cannot be determined.
        """
        try:
            # Use original_data to always get the size of the pristine file.
            reader = PdfReader(io.BytesIO(source.original_data))
            media_box = reader.pages[0].mediabox
            width_pt = float(media_box.width)
            height_pt = float(media_box.height)
            size_mm = (to_mm(width_pt, "pt"), to_mm(height_pt, "pt"))
            return size_mm if size_mm[0] > 0 and size_mm[1] > 0 else None
        except Exception as e:
            logger.error(f"Failed to read PDF size: {e}")
        return None

    def _find_content_bounding_box_px(
        self, vips_image
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Finds the bounding box of non-white content in an image using OpenCV.

        Args:
            vips_image: A pyvips image to analyze.

        Returns:
            A tuple (x, y, width, height) of the bounding box in pixels,
            or None.
        """
        normalized_image = image_util.normalize_to_rgba(vips_image)
        if not normalized_image:
            return None

        np_rgba = np.ndarray(
            buffer=normalized_image.write_to_memory(),
            dtype=np.uint8,
            shape=[normalized_image.height, normalized_image.width, 4],
        )

        # Convert to grayscale and create a binary mask of the content.
        gray = cv2.cvtColor(np_rgba, cv2.COLOR_RGBA2GRAY)
        # Threshold to find all non-white pixels (value < 255).
        _, binary_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Combine all contours to find the overall bounding box.
        all_points = np.vstack(contours)
        return cast(Tuple[int, int, int, int], cv2.boundingRect(all_points))

    def _calculate_render_resolution(
        self, w_mm: float, h_mm: float
    ) -> Tuple[int, int]:
        """
        Calculates optimal rendering dimensions in pixels based on a fixed
        pixels-per-millimeter setting.

        This ensures a consistent, high level of detail for tracing,
        regardless of the PDF's physical dimensions, while respecting a
        maximum dimension limit to manage memory.

        Args:
            w_mm: Width in millimeters.
            h_mm: Height in millimeters.

        Returns:
            A tuple of (width, height) in pixels.
        """
        if w_mm <= 0 or h_mm <= 0:
            aspect = 210 / 297  # A4 Aspect ratio
        else:
            aspect = w_mm / h_mm

        TARGET_DIM = math.sqrt(VTRACER_PIXEL_LIMIT)
        if aspect >= 1.0:  # Landscape or square
            w_px = int(TARGET_DIM)
            h_px = int(TARGET_DIM / aspect)
        else:  # Portrait
            h_px = int(TARGET_DIM)
            w_px = int(TARGET_DIM * aspect)

        return max(1, w_px), max(1, h_px)
