from typing import Optional, List
from ...core.item import DocItem
from ...core.workpiece import WorkPiece
from ...core.vectorization_config import TraceConfig
from ...core.geo import Geometry
from ...core.matrix import Matrix
from ...shared.util.tracing import trace_surface
from ..base_importer import Importer
from .renderer import PDF_RENDERER
import logging
import numpy as np
import io
from pypdf import PdfReader, PdfWriter, Transformation

logger = logging.getLogger(__name__)


class PdfImporter(Importer):
    label = "PDF files"
    mime_types = ("application/pdf",)
    extensions = (".pdf",)

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List[DocItem]]:
        """Retrieve document items from a PDF file, optionally tracing vectors.

        Args:
            vector_config: Configuration for vector tracing, if any.

        Returns:
            List containing a WorkPiece with rendered or traced content, or
            None if processing fails.
        """
        wp = self._create_workpiece()
        size_mm = self._get_pdf_size(wp)

        if not vector_config or not size_mm:
            return [wp]

        return self._trace_and_crop_pdf(wp, size_mm, vector_config)

    def _create_workpiece(self) -> WorkPiece:
        """Create a WorkPiece from the raw PDF data.

        Returns:
            WorkPiece initialized with source file, renderer, and raw data.
        """
        return WorkPiece(
            source_file=self.source_file,
            renderer=PDF_RENDERER,
            data=self.raw_data,
        )

    def _get_pdf_size(self, wp: WorkPiece) -> Optional[tuple[float, float]]:
        """Retrieve the natural size of the PDF in millimeters.

        Args:
            wp: WorkPiece containing the PDF data.

        Returns:
            Tuple of (width, height) in millimeters, or None if size cannot be
            determined.
        """
        try:
            size_mm = PDF_RENDERER.get_natural_size(wp)
            if size_mm and size_mm[0] and size_mm[1]:
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
        size_mm: tuple[float, float],
        vector_config: TraceConfig,
    ) -> List[DocItem]:
        """Trace the PDF content and crop it based on traced geometry.

        Args:
            wp: WorkPiece containing the PDF data.
            size_mm: Tuple of (width, height) in millimeters.
            vector_config: Configuration for vector tracing.

        Returns:
            List containing the processed WorkPiece with traced vectors and
            cropped content.
        """
        w_mm, h_mm = size_mm
        w_px, h_px = self._calculate_render_resolution(w_mm, h_mm)
        surface = PDF_RENDERER.render_to_pixels(wp, w_px, h_px)
        if not surface:
            return [wp]

        pixels_per_mm = (w_px / w_mm, h_px / h_mm) if w_mm and h_mm else (1, 1)
        geometries = trace_surface(surface, pixels_per_mm)
        if not geometries:
            return [wp]

        combined_geo = self._combine_geometries(geometries)
        return self._crop_and_transform_pdf(wp, combined_geo, w_mm, h_mm)

    def _calculate_render_resolution(
        self, w_mm: float, h_mm: float
    ) -> tuple[int, int]:
        """Calculate rendering resolution for tracing.

        Args:
            w_mm: Width of the PDF in millimeters.
            h_mm: Height of the PDF in millimeters.

        Returns:
            Tuple of (width, height) in pixels.
        """
        w_px = 2048
        h_px = int(w_px * (h_mm / w_mm)) if w_mm > 0 else 2048
        return w_px, h_px

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

    def _crop_and_transform_pdf(
        self, wp: WorkPiece, combined_geo: Geometry, w_mm: float, h_mm: float
    ) -> List[DocItem]:
        """Crop the PDF and transform its geometry to match new coordinates.

        Args:
            wp: WorkPiece containing the PDF data.
            combined_geo: Combined Geometry object from tracing.
            w_mm: Original width of the PDF in millimeters.
            h_mm: Original height of the PDF in millimeters.

        Returns:
            List containing the updated WorkPiece with cropped PDF and
            transformed geometry.
        """
        if not combined_geo.commands:
            return [wp]

        min_x, min_y, max_x, max_y = combined_geo.rect()
        if not (max_x > min_x and max_y > min_y):
            return [wp]

        content_width = max_x - min_x
        content_height = max_y - min_y
        pt_per_mm = 72 / 25.4
        page_height_pt = h_mm * pt_per_mm
        left_pt = min_x * pt_per_mm
        bottom_pt = page_height_pt - max_y * pt_per_mm
        right_pt = max_x * pt_per_mm
        top_pt = page_height_pt - min_y * pt_per_mm
        crop_width_pt = right_pt - left_pt
        crop_height_pt = top_pt - bottom_pt

        reader = PdfReader(io.BytesIO(self.raw_data))
        if not reader.pages:
            return [wp]
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
        wp.data = bio.getvalue()

        translation_matrix = Matrix.translation(-min_x, -(h_mm - max_y))
        m_3x3 = translation_matrix.to_numpy()
        m_4x4 = np.eye(4)
        m_4x4[:3, :3] = m_3x3
        m_4x4[:2, 3] = translation_matrix.get_translation()
        combined_geo.transform(m_4x4)

        wp.set_size(content_width, content_height)
        wp.vectors = combined_geo
        return [wp]
