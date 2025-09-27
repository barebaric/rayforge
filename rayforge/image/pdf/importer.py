import logging
import io
from pypdf import PdfReader, PdfWriter, Transformation
from typing import Optional, List

from ...core.geo import Geometry
from ...core.import_source import ImportSource
from ...core.matrix import Matrix
from ...core.vectorization_config import TraceConfig
from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from ..util import to_mm
from .. import image_util
from .renderer import PDF_RENDERER

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        """Retrieve document items from a PDF file, optionally tracing vectors.

        Args:
            vector_config: Configuration for vector tracing, if any.

        Returns:
            An ImportPayload containing the source and a WorkPiece with
            rendered or traced content, or None if processing fails.
        """
        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PDF_RENDERER,
        )
        wp = WorkPiece(name=self.source_file.stem)
        wp.import_source_uid = source.uid

        size_mm = self._get_pdf_size(source)
        if not size_mm:
            return ImportPayload(source=source, items=[wp])

        if not vector_config:
            wp.set_size(size_mm[0], size_mm[1])
            return ImportPayload(source=source, items=[wp])

        # Destructive trace & crop operation
        trace_result = self._trace_and_crop_pdf(source, size_mm, vector_config)

        if trace_result:
            # This now returns the final pre-scaled geometry and its final size
            final_geo_mm, final_size_mm = trace_result

            # 1. We have the final, correctly sized geometry (e.g., 80mm wide).
            #    We also know its final physical size (80mm x 60mm).

            # 2. Create a normalization matrix to scale the pre-scaled geometry
            #    down to a 1x1 unit box.
            width, height = final_size_mm
            if width > 0 and height > 0:
                norm_matrix = Matrix.scale(1.0 / width, 1.0 / height)
                final_geo_mm.transform(norm_matrix.to_4x4_numpy())

            # 3. Assign the now-normalized vectors to the workpiece.
            wp.vectors = final_geo_mm

            # 4. Set the workpiece size. This correctly sets the matrix
            #    to scale the normalized vectors back up to their physical
            #    size.
            wp.set_size(width, height)
        else:
            # Fallback if tracing fails, use the original PDF size
            wp.set_size(size_mm[0], size_mm[1])

        return ImportPayload(source=source, items=[wp])

    def _get_pdf_size(
        self, source: ImportSource
    ) -> Optional[tuple[float, float]]:
        """Retrieve the natural size of the PDF in millimeters.

        Args:
            source: The ImportSource containing the PDF data.

        Returns:
            Tuple of (width, height) in millimeters, or None if size cannot be
            determined.
        """
        try:
            reader = PdfReader(io.BytesIO(source.data))
            media_box = reader.pages[0].mediabox
            size_pt = (float(media_box.width), float(media_box.height))
            size_mm = (to_mm(size_pt[0], "pt"), to_mm(size_pt[1], "pt"))

            if size_mm[0] and size_mm[1]:
                return size_mm
        except Exception:
            # If size can't be determined (e.g., invalid PDF), return the
            # workpiece without a size. The UI can handle this.
            pass
        return None

    def _trace_and_crop_pdf(
        self,
        source: ImportSource,
        size_mm: tuple[float, float],
        vector_config: TraceConfig,
    ) -> Optional[tuple[Geometry, tuple[float, float]]]:
        """Trace the PDF content, crop it, and update the ImportSource.

        Args:
            source: The ImportSource containing data to modify.
            size_mm: Tuple of (width, height) in millimeters.
            vector_config: Configuration for vector tracing.

        Returns:
            A tuple containing the transformed, pre-scaled Geometry in mm and
            the final cropped size in mm, or None.
        """
        w_mm, h_mm = size_mm
        w_px, h_px = self._calculate_render_resolution(w_mm, h_mm)

        vips_image = PDF_RENDERER.render_data_to_vips_image(
            source.data, w_px, h_px
        )
        if not vips_image:
            return None

        normalized_image = image_util.normalize_to_rgba(vips_image)
        if not normalized_image:
            logger.warning("Failed to normalize vips image for tracing.")
            return None

        surface = image_util.vips_rgba_to_cairo_surface(normalized_image)
        if not surface:
            logger.warning("Failed to convert vips image to cairo surface.")
            return None

        geometries = trace_surface(surface)
        if not geometries:
            return None

        combined_geo = self._combine_geometries(geometries)
        if not combined_geo.commands:
            return None

        # Crop the PDF data in memory and update the source
        return self._crop_and_transform_geo(
            source, combined_geo, w_mm, h_mm, w_px, h_px
        )

    def _calculate_render_resolution(
        self, w_mm: float, h_mm: float
    ) -> tuple[int, int]:
        """
        Calculate rendering resolution for tracing.

        Args:
            w_mm: Width of the PDF in millimeters.
            h_mm: Height of the PDF in millimeters.

        Returns:
            Tuple of (width, height) in pixels.
        """
        TARGET_MEGAPIXELS = 8.0
        MAX_DIM = 8192

        if w_mm <= 0 or h_mm <= 0:
            return 2048, 2048

        target_pixels = TARGET_MEGAPIXELS * 1024 * 1024
        aspect_ratio = h_mm / w_mm if w_mm > 0 else 1.0

        if aspect_ratio == 0:
            aspect_ratio = 1.0

        w_px = int((target_pixels / aspect_ratio) ** 0.5)
        h_px = int(w_px * aspect_ratio)

        if w_px > MAX_DIM:
            w_px = MAX_DIM
            h_px = int(w_px * aspect_ratio)
        if h_px > MAX_DIM:
            h_px = MAX_DIM
            w_px = int(h_px / aspect_ratio) if aspect_ratio > 0 else MAX_DIM

        return max(w_px, 1), max(h_px, 1)

    def _combine_geometries(self, geometries: List[Geometry]) -> Geometry:
        """Combine multiple geometries into a single Geometry object."""
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)
        return combined_geo

    def _crop_and_transform_geo(
        self,
        source: ImportSource,
        combined_geo: Geometry,  # This is now in PIXEL coordinates
        w_mm: float,
        h_mm: float,
        w_px: int,  # Pass in the pixel dimensions
        h_px: int,  # Pass in the pixel dimensions
    ) -> Optional[tuple[Geometry, tuple[float, float]]]:
        """Crop the PDF in the source and transform geometry to match."""
        # Get the bounding box of the geometry in PIXEL coordinates
        min_x_px, min_y_px, max_x_px, max_y_px = combined_geo.rect()
        if not (max_x_px > min_x_px and max_y_px > min_y_px):
            return None

        # Convert the pixel-based bounding box to millimeter units
        # before cropping
        mm_per_px_x = w_mm / w_px
        mm_per_px_y = h_mm / h_px

        min_x_mm = min_x_px * mm_per_px_x
        min_y_mm = min_y_px * mm_per_px_y
        max_x_mm = max_x_px * mm_per_px_x
        max_y_mm = max_y_px * mm_per_px_y

        # Now use the correct millimeter values for the PDF crop calculations
        pt_per_mm = 72 / 25.4
        left_pt = min_x_mm * pt_per_mm
        bottom_pt = min_y_mm * pt_per_mm
        right_pt = max_x_mm * pt_per_mm
        top_pt = max_y_mm * pt_per_mm

        crop_width_pt = right_pt - left_pt
        crop_height_pt = top_pt - bottom_pt

        reader = PdfReader(io.BytesIO(source.original_data))
        if not reader.pages:
            return None
        page = reader.pages[0]

        op = Transformation().translate(tx=-left_pt, ty=-bottom_pt)
        page.add_transformation(op, expand=False)
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (crop_width_pt, crop_height_pt)

        writer = PdfWriter()
        writer.add_page(page)
        bio = io.BytesIO()
        writer.write(bio)
        source.data = bio.getvalue()

        # The geometry must now be scaled from pixels to millimeters...
        scaling_matrix = Matrix.scale(mm_per_px_x, mm_per_px_y)
        translation_matrix = Matrix.translation(-min_x_mm, -min_y_mm)

        # The final transform scales the pixel geometry and moves it
        # to the origin
        final_transform = translation_matrix @ scaling_matrix
        combined_geo.transform(final_transform.to_4x4_numpy())

        final_size_mm = (
            to_mm(crop_width_pt, "pt"),
            to_mm(crop_height_pt, "pt"),
        )

        return combined_geo, final_size_mm
