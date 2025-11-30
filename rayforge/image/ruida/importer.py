import logging
from typing import List, Optional
from ...core.item import DocItem
from ...core.geo import Geometry
from ...core.vectorization_spec import VectorizationSpec
from ..base_importer import Importer, ImportPayload
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import PassthroughSpec
from .renderer import RUIDA_RENDERER
from .parser import RuidaParser
from .job import RuidaJob
from ...core.workpiece import WorkPiece
from ...core.matrix import Matrix

logger = logging.getLogger(__name__)


class RuidaImporter(Importer):
    label = "Ruida files"
    mime_types = ("application/x-rd-file", "application/octet-stream")
    extensions = (".rd",)

    def _get_job(self) -> RuidaJob:
        """Parses the Ruida data into a job object."""
        parser = RuidaParser(self.raw_data)
        return parser.parse()

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportPayload]:
        # Ruida files are always vector, so vectorization_spec is ignored.
        job = self._get_job()
        geometry = self._get_geometry(job)
        geometry.close_gaps()

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=RUIDA_RENDERER,
        )

        if not geometry or geometry.is_empty():
            # Still return a source for an empty file, but no items.
            return ImportPayload(source=source, items=[])

        # Calculate and store the true natural size from the job's extents.
        min_x, min_y, max_x, max_y = job.get_extents()
        width_mm = max_x - min_x
        height_mm = max_y - min_y
        if width_mm > 0 and height_mm > 0:
            source.width_mm = width_mm
            source.height_mm = height_mm
            source.metadata["natural_size"] = (width_mm, height_mm)

        # Create a single workpiece for the entire geometry (no splitting)
        width = max(width_mm, 1e-9)
        height = max(height_mm, 1e-9)

        # The geometry from the Ruida is Y-up. We must convert it to a
        # normalized Y-down geometry for storage in the segment.
        segment_mask_geo = geometry.copy()

        # 1. Translate to origin (0,0 is bottom-left).
        translate_matrix = Matrix.translation(-min_x, -min_y)
        segment_mask_geo.transform(translate_matrix.to_4x4_numpy())

        # 2. Normalize to a 1x1 box (still Y-up).
        if width > 0 and height > 0:
            norm_matrix = Matrix.scale(1.0 / width, 1.0 / height)
            segment_mask_geo.transform(norm_matrix.to_4x4_numpy())

        # 3. Flip the Y-axis to convert to Y-down for storage.
        flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
        segment_mask_geo.transform(flip_matrix.to_4x4_numpy())

        passthrough_spec = PassthroughSpec()
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=segment_mask_geo,
            vectorization_spec=passthrough_spec,
        )
        wp = WorkPiece(
            name=self.source_file.stem,
            source_segment=gen_config,
        )
        wp.natural_width_mm = width
        wp.natural_height_mm = height
        wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
            width, height
        )

        items: List[DocItem] = [wp]
        return ImportPayload(source=source, items=items)

    def _get_geometry(self, job: RuidaJob) -> Geometry:
        """
        Returns the parsed vector geometry. The coordinate system is
        canonical (Y-up, origin at bottom-left of content).
        """
        geo = Geometry()
        if not job.commands:
            return geo

        _min_x, min_y, _max_x, max_y = job.get_extents()
        y_flip_val = max_y + min_y

        for cmd in job.commands:
            # Check the command type first, then safely access params.
            if cmd.command_type in ("Move_Abs", "Cut_Abs"):
                # Ensure params are valid before unpacking.
                if not cmd.params or len(cmd.params) != 2:
                    logger.warning(
                        f"Skipping Ruida command with invalid params: {cmd}"
                    )
                    continue

                x, y = cmd.params
                flipped_y = y_flip_val - y
                if cmd.command_type == "Move_Abs":
                    geo.move_to(x, flipped_y)
                elif cmd.command_type == "Cut_Abs":
                    geo.line_to(x, flipped_y)
        return geo
