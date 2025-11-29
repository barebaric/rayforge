import json
import logging
from typing import Dict, Any, TYPE_CHECKING, Optional

from rayforge.undo import Command
from rayforge.core.sketcher import Sketch
from rayforge.core.geo import Geometry
from rayforge.core.matrix import Matrix

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.core.source_asset import SourceAsset
    from rayforge.core.source_asset_segment import SourceAssetSegment

logger = logging.getLogger(__name__)


class UpdateSketchSourceCommand(Command):
    """
    A command that updates the raw JSON data of a SourceAsset with a new
    sketch. It also recalculates and updates the geometry and dimensions of
    the asset and any associated segments, making it a self-contained update.
    """

    def __init__(
        self,
        doc: "Doc",
        source_asset: "SourceAsset",
        new_sketch_dict: Dict[str, Any],
        name: str = "Edit Sketch",
    ):
        super().__init__(name)
        self.doc = doc
        self.source_asset = source_asset
        self.new_data = json.dumps(new_sketch_dict).encode("utf-8")
        self.old_data = source_asset.original_data

        # --- Store old state for undo ---
        self.old_width_mm = source_asset.width_mm
        self.old_height_mm = source_asset.height_mm
        self.old_geometry: Optional[Geometry] = None
        first_segment = self._find_first_segment()
        if first_segment:
            self.old_geometry = first_segment.segment_mask_geometry.copy()

        # --- Calculate new state from the provided sketch dict ---
        new_sketch = Sketch.from_dict(new_sketch_dict)
        raw_geometry = new_sketch.to_geometry()

        if raw_geometry.is_empty():
            self.new_width_mm = 50.0  # Default for empty sketch
            self.new_height_mm = 50.0
            self.new_geometry = raw_geometry
        else:
            min_x, min_y, max_x, max_y = raw_geometry.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

            self.new_width_mm = width
            self.new_height_mm = height

            # Normalize the Y-UP geometry to a 0-1 box
            normalized_geo = raw_geometry.copy()
            norm_matrix = Matrix.scale(
                1.0 / width, 1.0 / height
            ) @ Matrix.translation(-min_x, -min_y)
            normalized_geo.transform(norm_matrix.to_4x4_numpy())

            # Flip the Y-axis of the normalized geometry for storage (Y-DOWN)
            self.new_geometry = normalized_geo.copy()
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            self.new_geometry.transform(flip_matrix.to_4x4_numpy())

    def _find_first_segment(self) -> Optional["SourceAssetSegment"]:
        """Finds the first SourceAssetSegment in the doc using our asset."""
        for wp in self.doc.all_workpieces:
            if (
                wp.source_segment
                and wp.source_segment.source_asset_uid == self.source_asset.uid
            ):
                return wp.source_segment
        return None

    def _update_asset_and_segments(
        self,
        data: bytes,
        geometry: Optional[Geometry],
        width_mm: float,
        height_mm: float,
    ):
        """Helper to apply a full state to the asset and its segments."""
        self.source_asset.original_data = data
        self.source_asset.width_mm = width_mm
        self.source_asset.height_mm = height_mm

        # Update all workpieces that use this source asset
        for workpiece in self.doc.all_workpieces:
            if (
                workpiece.source_segment
                and workpiece.source_segment.source_asset_uid
                == self.source_asset.uid
            ):
                segment = workpiece.source_segment
                if geometry is not None:
                    segment.segment_mask_geometry = geometry.copy()
                segment.width_mm = width_mm
                segment.height_mm = height_mm

                # Synchronize the WorkPiece matrix with the new content size.
                # This ensures the visual scaling matches the new source
                # dimensions.
                workpiece.set_size(width_mm, height_mm)

                workpiece.clear_render_cache()

        self.doc.updated.send(self.doc)

    def execute(self):
        self._do_execute()

    def undo(self):
        self._do_undo()

    def _do_execute(self):
        logger.debug(
            f"Executing UpdateSketchSourceCommand for asset "
            f"{self.source_asset.uid}"
        )
        self._update_asset_and_segments(
            self.new_data,
            self.new_geometry,
            self.new_width_mm,
            self.new_height_mm,
        )

    def _do_undo(self):
        logger.debug(
            f"Undoing UpdateSketchSourceCommand for asset "
            f"{self.source_asset.uid}"
        )
        self._update_asset_and_segments(
            self.old_data,
            self.old_geometry,
            self.old_width_mm,
            self.old_height_mm,
        )
