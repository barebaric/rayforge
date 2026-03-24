from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Iterator, List, Optional, TYPE_CHECKING
from gettext import gettext as _
from rayforge.core.item import DocItem
from rayforge.core.matrix import Matrix
from rayforge.core.source_asset import SourceAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.image.base_importer import (
    Importer,
    ImporterFeature,
)
from rayforge.image.structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
    ImportPayload,
    ImportManifest,
)
from rayforge.image.engine import NormalizationEngine
from rayforge import const
from ..core import Sketch

if TYPE_CHECKING:
    from rayforge.core.vectorization_spec import VectorizationSpec

logger = logging.getLogger(__name__)


class SketchImporter(Importer):
    """
    Parses a .rfs file (serialized Sketch data) and prepares it for
    integration into a document.
    """

    label = _("{app_name} Sketch").format(app_name=const.APP_NAME)
    extensions = (".rfs",)
    mime_types = (const.MIME_TYPE_SKETCH,)
    features = {ImporterFeature.DIRECT_VECTOR}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.parsed_sketch: Optional[Sketch] = None

    def scan(self) -> ImportManifest:
        """
        Scans the sketch JSON to extract its name.
        """
        try:
            sketch_dict = json.loads(self.raw_data.decode("utf-8"))
            name = sketch_dict.get("name") or self.source_file.stem
            return ImportManifest(
                title=name, warnings=self._warnings, errors=self._errors
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Sketch scan failed for {self.source_file.name}: {e}"
            )
            self.add_error(_(f"Sketch file is invalid JSON: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

    def _post_process_payload(self, payload: ImportPayload) -> ImportPayload:
        """
        Overrides the base importer hook to add sketch-specific data.
        This links the generated WorkPieces back to the Sketch definition
        and includes the Sketch itself in the payload for the document to
        register.
        """
        if not self.parsed_sketch:
            return payload

        def find_workpieces(items: List[DocItem]) -> Iterator[WorkPiece]:
            """Recursively find all WorkPiece objects in a list of items."""
            for item in items:
                if isinstance(item, WorkPiece):
                    yield item
                elif item.children:
                    yield from find_workpieces(item.children)

        for wp in find_workpieces(payload.items):
            wp.geometry_provider_uid = self.parsed_sketch.uid
            wp.name = self.parsed_sketch.name

        payload.assets = [self.parsed_sketch]
        return payload

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """
        Creates a SourceAsset for Sketch import.
        """
        from .renderer import SKETCH_RENDERER

        _, _, width, height = parse_result.document_bounds

        return SourceAsset(
            source_file=self.source_file
            if self.source_file
            else Path("sketch.rfs"),
            original_data=self.raw_data,
            renderer=SKETCH_RENDERER,
            metadata={"is_vector": True},
            width_mm=width,
            height_mm=height,
        )

    def parse(self) -> Optional[ParsingResult]:
        """Phase 2: Parse JSON into Sketch model and solve it for bounds."""
        try:
            sketch_dict = json.loads(self.raw_data.decode("utf-8"))
            self.parsed_sketch = Sketch.from_dict(sketch_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse sketch data: {e}")
            self.add_error(_(f"Failed to load sketch structure: {e}"))
            return None

        final_name = self.parsed_sketch.name
        if not final_name and self.source_file:
            final_name = self.source_file.stem
        if not final_name:
            final_name = "Untitled"
        self.parsed_sketch.name = final_name

        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()

        if geometry.is_empty():
            min_x, min_y, width, height = 0.0, 0.0, 1.0, 1.0
        else:
            min_x, min_y, max_x, max_y = geometry.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

        document_bounds = (min_x, min_y, width, height)

        temp_result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=document_bounds,
            background_world_transform=Matrix(),
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            document_bounds, temp_result
        )

        layer_id = "__default__"

        return ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,
            layers=[
                LayerGeometry(
                    layer_id=layer_id,
                    name=layer_id,
                    content_bounds=document_bounds,
                )
            ],
            world_frame_of_reference=document_bounds,
            background_world_transform=bg_item.world_matrix,
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        """Phase 3: Extract geometry from the solved sketch."""
        if not self.parsed_sketch:
            return VectorizationResult({}, parse_result)

        geometry = self.parsed_sketch.to_geometry()
        geometry.close_gaps()
        geometry.upgrade_to_scalable()

        fill_render_data = self.parsed_sketch.get_fill_render_data()
        for fd in fill_render_data:
            fd.geometry.upgrade_to_scalable()

        layer_id = parse_result.layers[0].layer_id
        return VectorizationResult(
            geometries_by_layer={layer_id: geometry},
            fills_by_layer={layer_id: fill_render_data},
            source_parse_result=parse_result,
        )
