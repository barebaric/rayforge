from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from ...core.matrix import Matrix
from ...core.sketcher import Sketch
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import PassthroughSpec
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

        # 2. Convert Sketch to Geometry
        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()
        if geometry.is_empty():
            logger.warning("Imported sketch has no geometry.")

        # 3. Create the SourceAsset container first
        source_asset = SourceAsset(
            source_file=self.source_file
            if self.source_file
            else Path("sketch.rfs"),
            original_data=self.raw_data,
            renderer=self.renderer,
            metadata={"is_vector": True},
        )

        # 4. Create SourceAssetSegment linked to the asset
        min_x, min_y, max_x, max_y = geometry.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        # Normalize the Y-UP geometry to a 0-1 box
        normalized_geo = geometry.copy()
        norm_matrix = Matrix.scale(
            1.0 / width, 1.0 / height
        ) @ Matrix.translation(-min_x, -min_y)
        normalized_geo.transform(norm_matrix.to_4x4_numpy())

        # Flip the Y-axis of the normalized geometry for storage (Y-DOWN)
        y_down_geo = normalized_geo.copy()
        flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
        y_down_geo.transform(flip_matrix.to_4x4_numpy())

        segment = SourceAssetSegment(
            source_asset_uid=source_asset.uid,
            segment_mask_geometry=y_down_geo,
            vectorization_spec=PassthroughSpec(),
            width_mm=width,
            height_mm=height,
        )

        source_asset.width_mm = width
        source_asset.height_mm = height

        # 5. Create WorkPiece
        name = "Sketch"
        if self.source_file:
            name = self.source_file.stem

        workpiece = WorkPiece(name=name, source_segment=segment)
        workpiece.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
            width, height
        )

        return ImportPayload(source=source_asset, items=[workpiece])
