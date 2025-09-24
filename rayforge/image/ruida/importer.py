import logging
from typing import List, Optional
import numpy as np
from ...core.item import DocItem
from ...core.geo import Geometry
from ...core.vectorization_config import TraceConfig
from ..base_importer import Importer, ImportPayload
from ...core.import_source import ImportSource
from .renderer import RUIDA_RENDERER
from .parser import RuidaParser
from .job import RuidaJob
from ...core.workpiece import WorkPiece
from ...core.matrix import Matrix
from ...core.group import Group

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
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[ImportPayload]:
        # Ruida files are always vector, so vector_config is ignored.
        job = self._get_job()
        geometry = self._get_geometry(job)

        if not geometry or geometry.is_empty():
            return None

        source = ImportSource(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=RUIDA_RENDERER,
        )

        component_geometries = geometry.split_into_components()

        workpieces = []
        for component_geo in component_geometries:
            min_x, min_y, max_x, max_y = component_geo.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

            # Normalize the component geometry to have its origin at (0,0)
            normalized_geo = component_geo.copy()
            translate_matrix = np.array(
                [
                    [1, 0, 0, -min_x],
                    [0, 1, 0, -min_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            normalized_geo.transform(translate_matrix)

            # Create a workpiece for this component
            wp = WorkPiece(name=self.source_file.stem, vectors=normalized_geo)
            wp.import_source_uid = source.uid
            wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
                width, height
            )

            workpieces.append(wp)

        items: List[DocItem]
        if len(workpieces) > 1:
            # 1. Calculate collective bounding box of new workpieces.
            all_corners = []
            for wp in workpieces:
                unit_corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
                # At this stage, world transform is just the local matrix
                world_transform = wp.matrix
                all_corners.extend(
                    [world_transform.transform_point(c) for c in unit_corners]
                )

            if not all_corners:
                items = workpieces  # Fallback
            else:
                min_x = min(p[0] for p in all_corners)
                min_y = min(p[1] for p in all_corners)
                max_x = max(p[0] for p in all_corners)
                max_y = max(p[1] for p in all_corners)
                bbox_x, bbox_y = min_x, min_y
                bbox_w = max(max_x - min_x, 1e-9)
                bbox_h = max(max_y - min_y, 1e-9)

                # 2. Create group and set its matrix to match the bbox.
                group = Group(name=self.source_file.stem)
                group.matrix = Matrix.translation(
                    bbox_x, bbox_y
                ) @ Matrix.scale(bbox_w, bbox_h)

                # 3. Update workpiece matrices to be relative to the group.
                try:
                    group_inv_matrix = group.matrix.invert()
                    for wp in workpieces:
                        wp.matrix = group_inv_matrix @ wp.matrix
                    # 4. Add children and return the configured group.
                    group.set_children(workpieces)
                    items = [group]
                except np.linalg.LinAlgError:
                    items = workpieces  # Fallback

        elif workpieces:
            items = workpieces
        else:
            items = []

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
