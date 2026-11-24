import logging
from gettext import gettext as _
from pathlib import Path
from typing import ClassVar

from raygeo.formats.coreldraw import CdrDocument, CdrError, parse
from raygeo.geo import Geometry

from ...core.source_asset import SourceAsset
from ..base_importer import ImporterFeature
from ..dxf.importer import DxfImporter
from ..engine import NormalizationEngine
from ..structures import (
    ImportManifest,
    LayerGeometry,
    LayerInfo,
    ParsingResult,
)
from .renderer import CDR_RENDERER

logger = logging.getLogger(__name__)


class CorelDrawImporter(DxfImporter):
    """
    Imports CorelDRAW drawings as vector geometry.

    The whole file is read by raygeo, which parses the native RIFF and
    ZIP containers. Named layers come in as selectable layers, text is
    already outlined, and anything the reader did not convert (bitmaps,
    unknown objects) is reported as a warning. Only the first drawing
    page is imported; later pages are reported. Layer filtering and
    splitting are inherited from the DXF importer, which works on the
    same Y-up geometry.
    """

    label = "CorelDRAW files"
    mime_types = (
        "application/vnd.corel-draw",
        "image/x-coreldraw",
        "image/x-cdr",
        "application/x-cdr",
    )
    extensions = (".cdr",)
    native_mime_type = "application/vnd.corel-draw"
    features: ClassVar[set[ImporterFeature]] = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
    }

    def __init__(self, data: bytes, source_file: Path | None = None):
        super().__init__(data, source_file)
        self._document: CdrDocument | None = None
        self._read = False

    def _read_document(self) -> CdrDocument | None:
        """Reads the file once, caching the result between the phases."""
        if self._read:
            return self._document
        self._read = True
        try:
            document = parse(self.raw_data)
        except CdrError as e:
            logger.warning(f"CorelDRAW read failed: {e}")
            self.add_error(
                _("CorelDRAW file is corrupt or invalid: {}").format(e)
            )
            return None
        except Exception as e:
            logger.exception("Unexpected error while reading CorelDRAW")
            self.add_error(
                _("Unexpected error while reading CorelDRAW: {}").format(e)
            )
            return None

        for issue in document.issues:
            self.add_warning(_("CorelDRAW import: {}").format(issue))
        if document.page_count > 1:
            self.add_warning(
                _(
                    "The document has {} pages; only the first page is "
                    "imported."
                ).format(document.page_count)
            )

        geometries: dict[str | None, Geometry] = {
            name: geometry.copy()
            for name, geometry in document.layers
            if not geometry.is_empty()
        }
        if not geometries:
            self.add_error(
                _("{} contains no vector geometry.").format(
                    self.source_file.name
                )
            )
        self._geometries_by_layer = geometries
        self._document = document
        return document

    def scan(self) -> ImportManifest:
        document = self._read_document()
        if document is None:
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

        scale = document.units_to_mm
        _, _, width, height = self._page_bounds(document)
        return ImportManifest(
            title=self.source_file.name,
            layers=self._layer_infos(),
            natural_size_mm=(width * scale, height * scale),
            warnings=self._warnings,
            errors=self._errors,
        )

    def _page_bounds(
        self, document: CdrDocument
    ) -> tuple[float, float, float, float]:
        """The page box, or the drawing's own box when the page size is
        missing from the file."""
        if document.page_width > 0 and document.page_height > 0:
            return (0.0, 0.0, document.page_width, document.page_height)
        merged = self._merged_geometry()
        if merged is None:
            return (0.0, 0.0, 0.0, 0.0)
        x1, y1, x2, y2 = merged.rect()
        return (x1, y1, x2 - x1, y2 - y1)

    def _layer_infos(self) -> list[LayerInfo]:
        return [
            LayerInfo(
                id=name,
                name=name,
                feature_count=len(geometry.split_drawn_contours()),
            )
            for name, geometry in self._geometries_by_layer.items()
            if name is not None
        ]

    def parse(self) -> ParsingResult | None:
        document = self._read_document()
        if document is None:
            return None
        if not self._geometries_by_layer:
            return None

        scale = document.units_to_mm
        doc_bounds = self._page_bounds(document)

        temp_result = ParsingResult(
            document_bounds=doc_bounds,
            native_unit_to_mm=scale,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=(0, 0, 0, 0),
            background_world_transform=None,  # type: ignore
        )
        x, y, w, h = doc_bounds
        world_frame = (x * scale, y * scale, w * scale, h * scale)
        bg_layout_item = NormalizationEngine.calculate_layout_item(
            doc_bounds, temp_result
        )

        result = ParsingResult(
            document_bounds=doc_bounds,
            native_unit_to_mm=scale,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=world_frame,
            background_world_transform=bg_layout_item.world_matrix,
        )
        for layer_name, geometry in self._geometries_by_layer.items():
            if layer_name is None:
                continue
            gx1, gy1, gx2, gy2 = geometry.rect()
            result.layers.append(
                LayerGeometry(
                    layer_id=layer_name,
                    name=layer_name,
                    content_bounds=(gx1, gy1, gx2 - gx1, gy2 - gy1),
                )
            )
        return result

    def _native_unit_to_mm(self) -> float:
        if self._document is None:
            return 1.0
        return self._document.units_to_mm or 1.0

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        _, _, w, h = parse_result.document_bounds
        scale = parse_result.native_unit_to_mm
        return SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=CDR_RENDERER,
            metadata={"is_vector": True},
            thumbnail_data=self._render_thumbnail(),
            width_mm=w * scale,
            height_mm=h * scale,
        )
