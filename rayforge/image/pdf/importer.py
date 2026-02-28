import io
import logging
from typing import Optional, Tuple
from pathlib import Path
from pypdf import PdfReader
from pypdf.errors import PdfReadError
import warnings
from gettext import gettext as _

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from .. import util
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
    ImportManifest,
)
from ..tracing import trace_surface
from ..util import to_mm
from .renderer import PDF_RENDERER
from ..engine import NormalizationEngine

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
    features = {ImporterFeature.BITMAP_TRACING}
    _TRACE_PPM = 24.0  # ~600 DPI for tracing
    _MAX_RENDER_DIM = 16384

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._image: Optional[pyvips.Image] = None

    def scan(self) -> ImportManifest:
        """
        Scans the PDF to find the dimensions of its first page.
        """
        try:
            reader = PdfReader(io.BytesIO(self.raw_data))
            if not reader.pages:
                self.add_error(_("PDF file contains no pages."))
                return ImportManifest(
                    title=self.source_file.name, errors=self._errors
                )
            media_box = reader.pages[0].mediabox
            width_pt = float(media_box.width)
            height_pt = float(media_box.height)
            size_mm = (to_mm(width_pt, "pt"), to_mm(height_pt, "pt"))
            title = reader.metadata.title if reader.metadata else None
            return ImportManifest(
                title=title or self.source_file.name,
                natural_size_mm=size_mm,
                warnings=self._warnings,
                errors=self._errors,
            )
        except PdfReadError as e:
            logger.warning(f"PDF scan failed for {self.source_file.name}: {e}")
            self.add_error(_(f"Could not read PDF: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during PDF scan for "
                f"{self.source_file.name}: {e}",
                exc_info=True,
            )
            self.add_error(_(f"Unexpected error while scanning PDF: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """Creates a SourceAsset for the PDF import."""
        assert self._image is not None, "parse() must have been called first"

        # Populate dimensions
        _, _, w_px, h_px = parse_result.document_bounds
        width_mm = w_px * parse_result.native_unit_to_mm
        height_mm = h_px * parse_result.native_unit_to_mm

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=PDF_RENDERER,
            width_px=int(w_px),
            height_px=int(h_px),
            width_mm=width_mm,
            height_mm=height_mm,
        )

        # Store the rendered image so the preview dialog can use it.
        source.base_render_data = self._image.pngsave_buffer()
        return source

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """Phase 3: Generate vector geometry by tracing the bitmap."""
        assert self._image is not None, "parse() must be called first"
        if not isinstance(spec, TraceSpec):
            raise TypeError("PdfImporter only supports TraceSpec")

        norm_image = util.normalize_to_rgba(self._image)
        if not norm_image:
            logger.error("Failed to normalize PDF image for tracing.")
            self.add_error(_("Failed to process PDF image data."))
            return VectorizationResult(
                geometries_by_layer={}, source_parse_result=parse_result
            )

        surface = util.vips_rgba_to_cairo_surface(norm_image)
        geometries = trace_surface(surface, spec)
        merged_geo = Geometry()
        for geo in geometries:
            merged_geo.extend(geo)

        return VectorizationResult(
            geometries_by_layer={None: merged_geo},
            source_parse_result=parse_result,
        )

    def parse(self) -> Optional[ParsingResult]:
        """Phase 2: Render PDF to a high-res image and extract facts."""
        try:
            reader = PdfReader(io.BytesIO(self.raw_data))
            media_box = reader.pages[0].mediabox
            width_pt = float(media_box.width)
            height_pt = float(media_box.height)
            size_mm = (to_mm(width_pt, "pt"), to_mm(height_pt, "pt"))
        except Exception as e:
            logger.error(f"Failed to read PDF size: {e}")
            self.add_error(_(f"Failed to read PDF page dimensions: {e}"))
            self._image = None
            return None

        w_mm, h_mm = size_mm
        if w_mm <= 0 or h_mm <= 0:
            self._image = None
            self.add_error(_("PDF page has zero dimensions"))
            return None

        # Calculate render resolution
        render_w_px, render_h_px = self._calculate_render_resolution(
            w_mm, h_mm
        )
        # DPI calculation
        dpi = float(render_w_px / w_mm) * 25.4

        vips_image = PDF_RENDERER.render_base_image(
            self.raw_data, width=render_w_px, height=render_h_px
        )
        if not vips_image:
            logger.error("Failed to render PDF to an image for processing")
            self.add_error(_("Failed to rasterize PDF"))
            self._image = None
            return None

        # Set resolution metadata
        px_per_mm = dpi / 25.4
        self._image = vips_image.copy(xres=px_per_mm, yres=px_per_mm)

        document_bounds = (0.0, 0.0, float(render_w_px), float(render_h_px))
        native_unit_to_mm = 1.0 / px_per_mm

        x, y, w, h = document_bounds
        world_frame = (
            x * native_unit_to_mm,
            0.0,
            w * native_unit_to_mm,
            h * native_unit_to_mm,
        )

        # Create temporary result to calculate background transform
        temp_result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=native_unit_to_mm,
            is_y_down=True,
            layers=[],
            world_frame_of_reference=world_frame,
            background_world_transform=None,  # type: ignore
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            document_bounds, temp_result
        )

        return ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=native_unit_to_mm,
            is_y_down=True,
            layers=[
                LayerGeometry(
                    layer_id="__default__",
                    name="__default__",
                    content_bounds=document_bounds,
                )
            ],
            world_frame_of_reference=world_frame,
            background_world_transform=bg_item.world_matrix,
        )

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
            return 1, 1

        # Calculate ideal dimensions based on a target resolution (~600 DPI)
        ideal_w = w_mm * self._TRACE_PPM
        ideal_h = h_mm * self._TRACE_PPM

        # If the ideal size is too large, scale it down proportionally
        scale = 1.0
        if ideal_w > self._MAX_RENDER_DIM:
            scale = self._MAX_RENDER_DIM / ideal_w
        if ideal_h > self._MAX_RENDER_DIM:
            scale = min(scale, self._MAX_RENDER_DIM / ideal_h)

        final_w = int(ideal_w * scale)
        final_h = int(ideal_h * scale)

        return max(1, final_w), max(1, final_h)
