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

        # 2. Determine final name with priority logic.
        # Sketch.from_dict provides the serialized name, or "" if missing.
        final_name = self.parsed_sketch.name
        if not final_name and self.source_file:
            # Fallback to filename if no serialized name.
            final_name = self.source_file.stem

        # If still no name, apply a system default.
        if not final_name:
            final_name = "Untitled"

        # Apply the final name to the sketch object.
        self.parsed_sketch.name = final_name

        # 3. Solve the sketch and get all geometry (strokes and fills)
        self.parsed_sketch.solve()
        geometry = self.parsed_sketch.to_geometry()
        fill_geometries = self.parsed_sketch.get_fill_geometries()
        geometry.close_gaps()

        min_x, min_y, max_x, max_y = geometry.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        # 4. Create the SourceAsset container.
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

        # 5. Create the WorkPiece, giving it the same final name.
        workpiece = WorkPiece(name=final_name, source_segment=None)

        # 6. Set dimensions and transformation.
        workpiece.natural_width_mm = width
        workpiece.natural_height_mm = height
        workpiece.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
            width, height
        )

        # 7. Link to Sketch Template and SourceAsset.
        workpiece.sketch_uid = self.parsed_sketch.uid
        workpiece.source_asset_uid = source_asset.uid

        # 8. Pre-populate the workpiece's caches with normalized geometry.
        # This prevents the first render from being empty.
        norm_matrix = Matrix.scale(
            1.0 / width, 1.0 / height
        ) @ Matrix.translation(-min_x, -min_y)
        geometry.transform(norm_matrix.to_4x4_numpy())
        for fill_geo in fill_geometries:
            fill_geo.transform(norm_matrix.to_4x4_numpy())

        workpiece._boundaries_cache = geometry
        workpiece._fills_cache = fill_geometries

        return ImportPayload(
            source=source_asset,
            items=[workpiece],
            sketches=[self.parsed_sketch],
        )
