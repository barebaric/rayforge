from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from ...core.sketcher.sketch import Sketch
from ...core.source_asset import SourceAsset
from ...core.workpiece import WorkPiece
from ...core.vectorization_spec import PassthroughSpec
from ..assembler import ItemAssembler
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
)
from ..engine import NormalizationEngine
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
from .renderer import SKETCH_RENDERER

if TYPE_CHECKING:
    from ...core.vectorization_spec import VectorizationSpec

logger = logging.getLogger(__name__)


class SketchImporter(Importer):
    """
    Parses a .rfs file (serialized Sketch data) and prepares it for
    integration into a document.
    """

    label = "Rayforge Sketch"
    extensions = (".rfs",)
    mime_types = ("application/vnd.rayforge-sketch",)
    features = {ImporterFeature.DIRECT_VECTOR}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.renderer = SKETCH_RENDERER
        self.parsed_sketch: Optional[Sketch] = None

    def scan(self) -> ImportManifest:
        """
        Scans the sketch JSON to extract its name.
        """
        try:
            sketch_dict = json.loads(self.raw_data.decode("utf-8"))
            name = sketch_dict.get("name") or self.source_file.stem
            return ImportManifest(title=name)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                f"Sketch scan failed for {self.source_file.name}: {e}"
            )
            return ImportManifest(
                title=self.source_file.name,
                warnings=["Could not parse Sketch file. It may be corrupt."],
            )

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Deserializes the raw sketch data and converts it into a WorkPiece.
        """
        # Phase 2: Parse
        parse_result = self.parse()
        if not parse_result or not self.parsed_sketch:
            return None

        # Create SourceAsset
        source_asset = self.create_source_asset(parse_result)

        spec = vectorization_spec or PassthroughSpec()

        # Phase 3: Vectorize
        vec_result = self.vectorize(parse_result, spec)

        # Phase 4: Layout
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)
        if not plan:
            return ImportPayload(source=source_asset, items=[])

        # Phase 5: Assembly
        assembler = ItemAssembler()
        # Use the sketch name for the item
        final_name = self.parsed_sketch.name or "Untitled"
        items = assembler.create_items(
            source_asset=source_asset,
            layout_plan=plan,
            spec=spec,
            source_name=final_name,
            geometries=vec_result.geometries_by_layer,
        )

        # Post-Processing: Link WorkPieces to the Sketch object
        # The assembler creates standard WorkPieces linked to the source
        # segment. For Sketch items, we also need to link them to the
        # Sketch asset itself.
        for item in items:
            if isinstance(item, WorkPiece):
                item.sketch_uid = self.parsed_sketch.uid
                # Pre-populate caches to prevent immediate re-solve.
                # The geometry from _vectorize is in native units (mm), but the
                # cache expects normalized 0-1 geometry. The assembler created
                # the WorkPiece with the correct matrix.
                # We can re-use the logic from WorkPiece.from_sketch or simply
                # let the WorkPiece solve itself on first render. Since
                # WorkPiece.from_sketch logic is complex, letting it self-heal
                # is safer than duplicating normalization logic here.
                # However, to match previous behavior, we can try to set it if
                # simple.
                pass

        return ImportPayload(
            source=source_asset,
            items=items,
            sketches=[self.parsed_sketch],
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """
        Creates a SourceAsset for Sketch import.
        """
        _, _, width, height = parse_result.page_bounds

        return SourceAsset(
            source_file=self.source_file
            if self.source_file
            else Path("sketch.rfs"),
            original_data=self.raw_data,
            renderer=self.renderer,
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
            return None

        # Determine final name logic here to keep state consistent
        final_name = self.parsed_sketch.name
        if not final_name and self.source_file:
            final_name = self.source_file.stem
        if not final_name:
            final_name = "Untitled"
        self.parsed_sketch.name = final_name

        # Solve to get geometric properties
        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()
        # Note: Sketch geometry is Y-Up (mathematical).
        # We need to standardize on Y-Down for the pipeline or flag it.
        # The pipeline assumes native_is_y_down=False means Y-Up.

        if geometry.is_empty():
            min_x, min_y, width, height = 0.0, 0.0, 1.0, 1.0
        else:
            min_x, min_y, max_x, max_y = geometry.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

        # For Sketch, native units are mm.
        page_bounds = (min_x, min_y, width, height)
        layer_id = "__default__"

        return ParsingResult(
            page_bounds=page_bounds,
            native_unit_to_mm=1.0,
            is_y_down=False,  # Sketches are Y-Up
            layers=[
                LayerGeometry(
                    layer_id=layer_id,
                    name=layer_id,
                    content_bounds=page_bounds,
                )
            ],
        )

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        """Phase 3: Extract geometry from the solved sketch."""
        if not self.parsed_sketch:
            # Should not happen if parse succeeded
            return VectorizationResult({}, parse_result)

        geometry = self.parsed_sketch.to_geometry()
        geometry.close_gaps()
        geometry.upgrade_to_scalable()

        # We treat the sketch as a single layer
        layer_id = parse_result.layers[0].layer_id
        return VectorizationResult(
            geometries_by_layer={layer_id: geometry},
            source_parse_result=parse_result,
        )
