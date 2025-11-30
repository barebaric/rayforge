from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from ...core.matrix import Matrix
from ...core.sketcher.sketch import Sketch
from ...core.source_asset import SourceAsset
from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
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
    is_bitmap = False

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.renderer = SKETCH_RENDERER
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

        # 2. Convert Sketch to Geometry (for the WorkPiece's cache/bounds)
        # Note: We solve it here just to get the initial bounds/mask.
        # The WorkPiece will re-solve its own instance later.
        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()
        geometry.close_gaps()

        min_x, min_y, max_x, max_y = geometry.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        # 1. Create the SourceAsset container to hold the original file bytes.
        # This is important for saving, exporting, and round-trip editing.
        source_asset = SourceAsset(
            source_file=self.source_file
            if self.source_file
            else Path("sketch.rfs"),
            original_data=self.raw_data,
            renderer=self.renderer,
            metadata={"is_vector": True},
            width_mm=width,
            height_mm=height,
        )

        # 2. Create the WorkPiece without a SourceAssetSegment.
        name = "Sketch"
        if self.source_file:
            name = self.source_file.stem

        workpiece = WorkPiece(name=name, source_segment=None)

        # 3. Set its dimensions and transformation directly.
        workpiece.natural_width_mm = width
        workpiece.natural_height_mm = height
        workpiece.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
            width, height
        )

        # 4. Link the WorkPiece to the Sketch Template.
        workpiece.sketch_uid = self.parsed_sketch.uid

        return ImportPayload(
            source=source_asset,
            items=[workpiece],
            sketches=[self.parsed_sketch],
        )
