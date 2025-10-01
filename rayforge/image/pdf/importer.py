import io
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation

from ...core.geo import Geometry
from ...core.import_source import ImportSource
from ...core.matrix import Matrix
from ...core.vectorization_config import TraceConfig
from ...core.workpiece import WorkPiece
from .. import image_util
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from ..util import to_mm
from .renderer import PDF_RENDERER

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    """
    Imports vector and raster data from PDF files.

    This importer can operate in two modes:
    1.  As-is Import: The PDF is imported as a background image, preserving its
        original dimensions. This happens when no vectorization is requested.
    2.  Vectorization: If a TraceConfig is provided, the importer performs an
        auto-cropping and tracing operation. It first finds the content bounds,
        crops the PDF to that area, and then traces the result to generate
        vector geometry.
    """

    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    # --- Constants for Rendering and Conversion ---
    _TARGET_MEGAPIXELS = 8.0
    _MAX_RENDER_DIM = 8192
    _POINTS_PER_INCH = 72.0
    _MM_PER_INCH = 25.4
    _PT_PER_MM = _POINTS_PER_INCH / _MM_PER_INCH

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        """
        Retrieve document items from the PDF file.

        If a vector_config is provided, the PDF content will be auto-cropped,
        traced, and imported as vectors. Otherwise, the PDF is imported as a
        background image with its original dimensions.

        Args:
            vector_config: Configuration for vector tracing. If None, the PDF
              is not traced.

        Returns:
            An ImportPayload containing the source and a WorkPiece, or None if
            processing fails.
        """
        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PDF_RENDERER,
        )
        wp = WorkPiece(name=self.source_file.stem)
        wp.import_source_uid = source.uid

        original_size_mm = self._get_pdf_size(source)
        if not original_size_mm:
            # If size can't be determined, return a workpiece without a size.
            return ImportPayload(source=source, items=[wp])

        if not vector_config:
            # No vectorization, just set the size and return.
            wp.set_size(original_size_mm[0], original_size_mm[1])
            return ImportPayload(source=source, items=[wp])

        # Perform the full crop-and-trace operation.
        trace_result = self._autocrop_and_trace(
            source, original_size_mm, vector_config
        )

        if trace_result:
            final_geo_mm, final_size_mm = trace_result
            final_geo_mm.close_gaps()
            self._populate_workpiece_with_vectors(
                wp, final_geo_mm, final_size_mm
            )
        else:
            # Fallback to original size if tracing fails.
            logger.warning(
                "PDF tracing failed. Falling back to original dimensions."
            )
            wp.set_size(original_size_mm[0], original_size_mm[1])

        return ImportPayload(source=source, items=[wp])

    def _get_pdf_size(
        self, source: ImportSource
    ) -> Optional[Tuple[float, float]]:
        """
        Retrieve the natural size of the PDF's first page in millimeters.

        Args:
            source: The ImportSource containing the PDF data.

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

            if size_mm[0] > 0 and size_mm[1] > 0:
                return size_mm
        except Exception as e:
            logger.error(f"Failed to read PDF size: {e}")
        return None

    def _populate_workpiece_with_vectors(
        self,
        wp: WorkPiece,
        geometry_mm: Geometry,
        size_mm: Tuple[float, float],
    ) -> None:
        """
        Normalizes geometry and sets the workpiece size and vectors.

        The WorkPiece expects its vector geometry to be normalized to a 1x1
        unit box. This function performs that normalization and then sets the
        workpiece's final physical size, which correctly configures its
        internal matrix to scale the vectors back up.

        Args:
            wp: The WorkPiece to populate.
            geometry_mm: The traced geometry, scaled to its final physical (mm)
              size.
            size_mm: The final physical (width, height) in mm.
        """
        width, height = size_mm
        if width > 0 and height > 0:
            # Create a matrix to scale the physically-sized geometry down
            # to a 1x1 box.
            norm_matrix = Matrix.scale(1.0 / width, 1.0 / height)
            geometry_mm.transform(norm_matrix.to_4x4_numpy())

        # Assign the now-normalized vectors to the workpiece.
        wp.vectors = geometry_mm

        # Set the workpiece size. This scales the normalized vectors back up.
        wp.set_size(width, height)

    def _autocrop_and_trace(
        self,
        source: ImportSource,
        original_size_mm: Tuple[float, float],
        vector_config: TraceConfig,
    ) -> Optional[Tuple[Geometry, Tuple[float, float]]]:
        """
        Orchestrates the autocrop, trace, and scale process.

        Args:
            source: The ImportSource containing the PDF.
            original_size_mm: The original dimensions of the PDF.
            vector_config: Configuration for vector tracing.

        Returns:
            A tuple containing the final scaled Geometry and the final size in
            millimeters, or None on failure.
        """
        # Stage 1: Create a new, tightly cropped PDF in memory.
        crop_result = self._autocrop_pdf(source, original_size_mm)
        if not crop_result:
            logger.warning("PDF auto-cropping failed.")
            return None

        cropped_pdf_data, final_size_mm = crop_result
        # Update the source's main data with the cropped version for rendering.
        source.data = cropped_pdf_data

        # Stage 2: Trace the new, cropped PDF to get pixel-based geometry.
        trace_result = self._trace_pdf_data(cropped_pdf_data, final_size_mm)
        if not trace_result:
            logger.warning("Failed to trace cropped PDF data.")
            return None

        pixel_geometry, render_size_px = trace_result
        if not pixel_geometry.commands:
            logger.warning("Tracing resulted in empty geometry.")
            return None

        # Stage 3: Scale the pixel-based geometry to its final physical
        # size in mm.
        final_w_mm, final_h_mm = final_size_mm
        render_w_px, render_h_px = render_size_px

        scale_x = final_w_mm / render_w_px if render_w_px > 0 else 1.0
        scale_y = final_h_mm / render_h_px if render_h_px > 0 else 1.0

        scaling_matrix = Matrix.scale(scale_x, scale_y)
        pixel_geometry.transform(scaling_matrix.to_4x4_numpy())

        return pixel_geometry, final_size_mm

    def _autocrop_pdf(
        self, source: ImportSource, original_size_mm: Tuple[float, float]
    ) -> Optional[Tuple[bytes, Tuple[float, float]]]:
        """
        Crops a PDF to its content's bounding box.

        Renders the PDF, uses OpenCV to find the content bounds, and then uses
        pypdf to create a new, cropped PDF in memory.

        Args:
            source: The PDF import source.
            original_size_mm: The original (width, height) of the PDF in mm.

        Returns:
            A tuple of (cropped_pdf_bytes, (new_width_mm, new_height_mm)),
            or None.
        """
        # Render the original PDF to an image for content analysis.
        w_mm, h_mm = original_size_mm
        w_px, h_px = self._calculate_render_resolution(w_mm, h_mm)

        vips_image = PDF_RENDERER.render_data_to_vips_image(
            source.original_data, w_px, h_px
        )
        if not vips_image:
            return None

        # Find the bounding box of the content in pixels.
        crop_box_px = self._find_content_bounding_box_px(vips_image)
        if not crop_box_px:
            logger.warning("No content found in PDF for cropping.")
            return None

        # Convert pixel crop box to PDF's coordinate system (points, Y-up).
        min_x_px, min_y_px, box_w_px, box_h_px = crop_box_px
        mm_per_px_x = w_mm / w_px
        mm_per_px_y = h_mm / h_px

        # PDF origin (0,0) is at the bottom-left. Pixel origin is top-left.
        left_pt = min_x_px * mm_per_px_x * self._PT_PER_MM
        bottom_pt = (
            (h_px - (min_y_px + box_h_px)) * mm_per_px_y * self._PT_PER_MM
        )
        crop_width_pt = box_w_px * mm_per_px_x * self._PT_PER_MM
        crop_height_pt = box_h_px * mm_per_px_y * self._PT_PER_MM

        if crop_width_pt <= 0 or crop_height_pt <= 0:
            return None

        # Use pypdf to create the new, cropped PDF in memory.
        reader = PdfReader(io.BytesIO(source.original_data))
        page = reader.pages[0]

        # Apply a transformation to move the content's top-left corner
        # to (0,0).
        op = Transformation().translate(tx=-left_pt, ty=-bottom_pt)
        page.add_transformation(op)

        # Set the new page size (media box) to the size of the content.
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (crop_width_pt, crop_height_pt)

        writer = PdfWriter()
        writer.add_page(page)
        output_stream = io.BytesIO()
        writer.write(output_stream)

        final_size_mm = (
            to_mm(crop_width_pt, "pt"),
            to_mm(crop_height_pt, "pt"),
        )
        return output_stream.getvalue(), final_size_mm

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
        return cv2.boundingRect(all_points)

    def _trace_pdf_data(
        self, pdf_data: bytes, size_mm: Tuple[float, float]
    ) -> Optional[Tuple[Geometry, Tuple[int, int]]]:
        """
        Renders PDF data to an image and traces it to produce vector geometry.

        Args:
            pdf_data: The raw bytes of the PDF to trace.
            size_mm: The (width, height) of the PDF in millimeters.

        Returns:
            A tuple containing the traced Geometry in pixel coordinates and the
            (width, height) of the rendered image in pixels, None on failure.
        """
        w_mm, h_mm = size_mm
        w_px, h_px = self._calculate_render_resolution(w_mm, h_mm)

        vips_image = PDF_RENDERER.render_data_to_vips_image(
            pdf_data, w_px, h_px
        )
        if not vips_image:
            return None

        norm_image = image_util.normalize_to_rgba(vips_image)
        if not norm_image:
            return None

        surface = image_util.vips_rgba_to_cairo_surface(norm_image)
        if not surface:
            return None

        geometries = trace_surface(surface)
        if not geometries:
            return None

        combined_geo = self._combine_geometries(geometries)
        return combined_geo, (w_px, h_px)

    def _calculate_render_resolution(
        self, w_mm: float, h_mm: float
    ) -> Tuple[int, int]:
        """
        Calculates optimal rendering dimensions in pixels.

        Aims for a target megapixel count to ensure sufficient detail for
        tracing, while respecting a maximum dimension limit to manage memory.

        Args:
            w_mm: Width in millimeters.
            h_mm: Height in millimeters.

        Returns:
            A tuple of (width, height) in pixels.
        """
        if w_mm <= 0 or h_mm <= 0:
            return 2048, 2048  # Default resolution for invalid input

        target_pixels = self._TARGET_MEGAPIXELS * 1024 * 1024
        aspect_ratio = h_mm / w_mm

        w_px = int((target_pixels / aspect_ratio) ** 0.5)
        h_px = int(w_px * aspect_ratio)

        # Clamp dimensions to the maximum allowed size.
        if w_px > self._MAX_RENDER_DIM:
            w_px = self._MAX_RENDER_DIM
            h_px = int(w_px * aspect_ratio)
        if h_px > self._MAX_RENDER_DIM:
            h_px = self._MAX_RENDER_DIM
            w_px = int(h_px / aspect_ratio)

        return max(w_px, 1), max(h_px, 1)

    def _combine_geometries(self, geometries: List[Geometry]) -> Geometry:
        """
        Merges a list of Geometry objects into a single one.

        Args:
            geometries: A list of Geometry objects.

        Returns:
            A single Geometry object containing all commands from the
            input list.
        """
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)
        return combined_geo
