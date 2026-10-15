import logging
import warnings
from gettext import gettext as _
from pathlib import Path
from typing import ClassVar

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from raygeo.geo import Geometry

from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import TraceSpec, VectorizationSpec
from .. import util
from ..base_importer import Importer, ImporterFeature
from ..engine import NormalizationEngine
from ..structures import (
    ImportManifest,
    LayerGeometry,
    ParsingResult,
    VectorizationResult,
)
from ..tracing import trace_surface
from .renderer import BITMAP_RENDERER, load_first_page

logger = logging.getLogger(__name__)


class _BitmapImporter(Importer):
    """
    Imports any bitmap format that libvips can load by content, such as
    GIF, TIFF and WebP. Multi-page files (animated GIFs, multi-page
    TIFFs) import their first page. The bitmap is traced into vectors.
    """

    image_format: ClassVar[str] = ""
    features: ClassVar[set[ImporterFeature]] = {ImporterFeature.BITMAP_TRACING}

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        self._image: pyvips.Image | None = None

    def _load(self) -> pyvips.Image | None:
        try:
            return load_first_page(self.raw_data)
        except pyvips.Error as e:
            logger.warning(
                f"{self.image_format} load failed for "
                f"{self.source_file.name}: {e}"
            )
            self.add_error(
                _("Could not read the {format} image: {error}").format(
                    format=self.image_format, error=e
                )
            )
            return None

    def scan(self) -> ImportManifest:
        image = self._load()
        if image is None:
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )
        return ImportManifest(
            title=self.source_file.name,
            natural_size_mm=util.get_physical_size_mm(image),
            warnings=self._warnings,
            errors=self._errors,
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        metadata = util.extract_vips_metadata(self._image)
        metadata["image_format"] = self.image_format
        _ignored1, _ignored2, w_px, h_px = parse_result.document_bounds
        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=BITMAP_RENDERER,
            metadata=metadata,
            thumbnail_data=self._render_thumbnail_from_vips(self._image),
            width_px=int(w_px),
            height_px=int(h_px),
            width_mm=w_px * parse_result.native_unit_to_mm,
            height_mm=h_px * parse_result.native_unit_to_mm,
        )

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """Phase 3: Generate vector geometry by tracing the bitmap."""
        assert self._image is not None, "parse() must be called first"
        if not isinstance(spec, TraceSpec):
            raise TypeError(f"{type(self).__name__} only supports TraceSpec")

        normalized_image = util.normalize_to_rgba(self._image)
        if not normalized_image:
            self.add_error(_("Failed to process image data."))
            return VectorizationResult(
                geometries_by_layer={}, source_parse_result=parse_result
            )

        surface = util.vips_rgba_to_cairo_surface(normalized_image)
        merged_geo = Geometry()
        for geo in trace_surface(surface, spec):
            merged_geo.extend(geo)

        return VectorizationResult(
            geometries_by_layer={None: merged_geo},
            source_parse_result=parse_result,
        )

    def parse(self) -> ParsingResult | None:
        """Phase 2: Load the first page and extract geometric facts."""
        self._image = self._load()
        if self._image is None:
            return None

        image = self._image
        document_bounds = (0.0, 0.0, float(image.width), float(image.height))
        native_unit_to_mm, _mm_per_px_y = util.get_mm_per_pixel(image)

        x, _y, w, h = document_bounds
        world_frame = (
            x * native_unit_to_mm,
            0.0,
            w * native_unit_to_mm,
            h * native_unit_to_mm,
        )
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


class GifImporter(_BitmapImporter):
    label = "GIF files"
    mime_types = ("image/gif",)
    extensions = (".gif",)
    image_format = "GIF"


class TiffImporter(_BitmapImporter):
    label = "TIFF files"
    mime_types = ("image/tiff",)
    extensions = (".tif", ".tiff")
    image_format = "TIFF"


class WebpImporter(_BitmapImporter):
    label = "WebP files"
    mime_types = ("image/webp",)
    extensions = (".webp",)
    image_format = "WebP"
