from typing import Optional, List
import logging
import io
import numpy as np
from pypdf import PdfReader, PdfWriter, Transformation
from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...core.matrix import Matrix
from ..tracing import trace_surface
from ..util import to_mm
from ...core.import_source import ImportSource
from ..base_importer import Importer, ImportPayload
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

        size_mm = self._get_pdf_size(wp, source)

        if not vector_config or not size_mm:
            return ImportPayload(source=source, items=[wp])

        # Destructive trace & crop operation
        combined_geo = self._trace_and_crop_pdf(
            wp, source, size_mm, vector_config
        )
        if combined_geo:
            wp.vectors = combined_geo
            # After cropping, the source data has changed, so we must
            # clear the cache before getting the new size.
            wp.clear_render_cache()
            self._get_pdf_size(wp, source)

        return ImportPayload(source=source, items=[wp])

    def _get_pdf_size(
        self, wp: WorkPiece, source: ImportSource
    ) -> Optional[tuple[float, float]]:
        """Retrieve the natural size of the PDF in millimeters and set it.

        Args:
            wp: WorkPiece to set the size on.
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
                wp.set_size(size_mm[0], size_mm[1])
                return size_mm
        except Exception:
            # If size can't be determined (e.g., invalid PDF), return the
            # workpiece without a size. The UI can handle this.
            pass
        return None

    def _trace_and_crop_pdf(
        self,
        wp: WorkPiece,
        source: ImportSource,
        size_mm: tuple[float, float],
        vector_config: TraceConfig,
    ) -> Optional[Geometry]:
        """Trace the PDF content, crop it, and update the ImportSource.

        Args:
            wp: WorkPiece being built.
            source: The ImportSource containing data to modify.
            size_mm: Tuple of (width, height) in millimeters.
            vector_config: Configuration for vector tracing.

        Returns:
            The transformed Geometry from the traced content, or None.
        """
        w_mm, h_mm = size_mm
        w_px, h_px = self._calculate_render_resolution(w_mm, h_mm)

        vips_image = PDF_RENDERER.render_data_to_vips_image(
            source.data, w_px, h_px
        )
        if not vips_image:
            return None

        # Normalize the image.
        normalized_image = image_util.normalize_to_rgba(vips_image)
        if not normalized_image:
            logger.warning("Failed to normalize vips image for tracing.")
            return None

        surface = image_util.vips_rgba_to_cairo_surface(normalized_image)
        if not surface:
            logger.warning("Failed to convert vips image to cairo surface.")
            return None

        pixels_per_mm = (w_px / w_mm, h_px / h_mm) if w_mm and h_mm else (1, 1)
        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return None

        combined_geo = self._combine_geometries(geometries)
        if not combined_geo.commands:
            return None

        # Crop the PDF data in memory and update the source
        return self._crop_and_transform_geo(source, combined_geo, w_mm, h_mm)

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
        """Combine multiple geometries into a single Geometry object.

        Args:
            geometries: List of Geometry objects from tracing.

        Returns:
            Combined Geometry object containing all commands.
        """
        combined_geo = Geometry()
        for geo in geometries:
            combined_geo.commands.extend(geo.commands)
        return combined_geo

    def _crop_and_transform_geo(
        self,
        source: ImportSource,
        combined_geo: Geometry,
        w_mm: float,
        h_mm: float,
    ) -> Optional[Geometry]:
        """Crop the PDF in the source and transform geometry to match.

        Args:
            source: The ImportSource whose working_data will be updated.
            combined_geo: Combined Geometry object from tracing.
            w_mm: Original width of the PDF in millimeters.
            h_mm: Original height of the PDF in millimeters.

        Returns:
            The transformed geometry, or None if cropping fails.
        """
        min_x, min_y, max_x, max_y = combined_geo.rect()
        if not (max_x > min_x and max_y > min_y):
            return None

        pt_per_mm = 72 / 25.4
        left_pt = min_x * pt_per_mm
        bottom_pt = min_y * pt_per_mm
        right_pt = max_x * pt_per_mm
        top_pt = max_y * pt_per_mm
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
        page.cropbox.lower_left = (0, 0)
        page.cropbox.upper_right = (crop_width_pt, crop_height_pt)

        writer = PdfWriter()
        writer.add_page(page)
        bio = io.BytesIO()
        writer.write(bio)
        source.data = bio.getvalue()

        translation_matrix = Matrix.translation(-min_x, -min_y)
        m_3x3 = translation_matrix.to_numpy()
        m_4x4 = np.eye(4)
        m_4x4[:3, :3] = m_3x3
        m_4x4[:2, 3] = translation_matrix.get_translation()
        combined_geo.transform(m_4x4)

        return combined_geo
