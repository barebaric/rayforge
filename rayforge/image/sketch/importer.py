from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, cast

from ..base_importer import Importer, ImportPayload
from ...core.sketcher import Sketch
from ...core.workpiece import WorkPiece
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import PassthroughSpec

if TYPE_CHECKING:
    from ...core.vectorization_spec import VectorizationSpec
    from ...image.base_renderer import Renderer

logger = logging.getLogger(__name__)


class SketchImporter(Importer):
    """
    Parses a .rfs file (serialized Sketch data) and prepares it for
    integration into a document.
    """

    label = "Rayforge Sketch"
    extensions = (".rfs",)
    mime_types = ("application/vnd.rayforge-sketch",)
    is_bitmap = False

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.parsed_sketch: Optional[Sketch] = None

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Deserializes the raw sketch data and converts it into a WorkPiece.
        """
        try:
            # 1. Parse JSON into Sketch model
            sketch_dict = json.loads(self.raw_data.decode("utf-8"))
            self.parsed_sketch = Sketch.from_dict(sketch_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse sketch data: {e}")
            return None

        # 2. Convert Sketch to Geometry
        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()

        if geometry.is_empty():
            logger.warning("Imported sketch has no geometry.")

        # 3. Create the SourceAsset container first
        renderer = cast("Renderer", self)
        source_asset = SourceAsset(
            source_file=self.source_file
            if self.source_file
            else Path("sketch.rfs"),
            original_data=self.raw_data,
            renderer=renderer,
        )

        # 4. Create SourceAssetSegment linked to the asset
        segment = SourceAssetSegment(
            source_asset_uid=source_asset.uid,
            segment_mask_geometry=geometry,
            vectorization_spec=PassthroughSpec(),
        )

        # Store dimensions
        min_x, min_y, max_x, max_y = geometry.rect()
        width = max_x - min_x
        height = max_y - min_y

        segment.width_mm = width
        segment.height_mm = height
        source_asset.width_mm = width
        source_asset.height_mm = height

        # 5. Create WorkPiece
        name = "Sketch"
        if self.source_file:
            name = self.source_file.stem

        workpiece = WorkPiece(name=name, source_segment=segment)

        return ImportPayload(source=source_asset, items=[workpiece])
