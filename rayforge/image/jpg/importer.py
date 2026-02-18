import warnings
from typing import Optional
import logging
from pathlib import Path

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
from .renderer import JPG_RENDERER
from ..engine import NormalizationEngine

logger = logging.getLogger(__name__)


class JpgImporter(Importer):
    label = "JPEG files"
    mime_types = ("image/jpeg",)
    extensions = (".jpg", ".jpeg")
    features = {ImporterFeature.BITMAP_TRACING}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._image: Optional[pyvips.Image] = None

    def scan(self) -> ImportManifest:
        """
        Scans the JPEG to extract physical dimensions from its metadata.
        """
        try:
            image = pyvips.Image.jpegload_buffer(
                self.raw_data, access=pyvips.Access.SEQUENTIAL
            )
            size_mm = util.get_physical_size_mm(image)
            return ImportManifest(
                title=self.source_file.name,
                natural_size_mm=size_mm,
                warnings=self._warnings,
                errors=self._errors,
            )
        except pyvips.Error as e:
            logger.warning(
                f"JPEG scan failed for {self.source_file.name}: {e}"
            )
            self.add_error(_(f"Failed to scan JPEG file: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """
        Creates a SourceAsset for JPEG import.
        """
        metadata = util.extract_vips_metadata(self._image)
        metadata["image_format"] = "JPEG"
        _, _, w_px, h_px = parse_result.document_bounds
        width_mm = w_px * parse_result.native_unit_to_mm
        height_mm = h_px * parse_result.native_unit_to_mm

        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=JPG_RENDERER,
            metadata=metadata,
            width_px=int(w_px),
            height_px=int(h_px),
            width_mm=width_mm,
            height_mm=height_mm,
        )

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """Phase 3: Generate vector geometry by tracing the bitmap."""
        assert self._image is not None, "parse() must be called first"
        if not isinstance(spec, TraceSpec):
            raise TypeError("JpgImporter only supports TraceSpec")

        normalized_image = util.normalize_to_rgba(self._image)
        if not normalized_image:
            logger.error("Failed to normalize image to RGBA format.")
            self.add_error(_("Failed to process image data."))
            return VectorizationResult(
                geometries_by_layer={}, source_parse_result=parse_result
            )

        surface = util.vips_rgba_to_cairo_surface(normalized_image)
        geometries = trace_surface(surface, spec)
        merged_geo = Geometry()
        for geo in geometries:
            merged_geo.extend(geo)

        return VectorizationResult(
            geometries_by_layer={None: merged_geo},
            source_parse_result=parse_result,
        )

    def parse(self) -> Optional[ParsingResult]:
        """Phase 2: Parse the JPG into a vips image and extract facts."""
        try:
            image = pyvips.Image.jpegload_buffer(
                self.raw_data, access=pyvips.Access.RANDOM
            )
        except pyvips.Error as e:
            logger.error(
                f"pyvips failed to load JPEG buffer: {e}", exc_info=True
            )
            self.add_error(_(f"Image load failed: {e}"))
            self._image = None
            return None

        self._image = image

        # Extract geometric facts
        width_px = float(image.width)
        height_px = float(image.height)
        document_bounds = (0.0, 0.0, width_px, height_px)

        # VIPS stores resolution as pixels per millimeter in xres/yres
        if image.xres > 0:
            native_unit_to_mm = 1.0 / image.xres
        else:
            # Fallback to a standard screen DPI if metadata is missing
            default_dpi = 96.0
            native_unit_to_mm = 25.4 / default_dpi

        # World frame is Y-Up, so y-origin is 0.
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
