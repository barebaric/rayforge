import logging
from typing import List, Optional, Tuple
import numpy as np
from ...core.item import DocItem
from ...core.geo import Geometry
from ...core.vectorization_config import TraceConfig
from ..base_importer import Importer
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

    def __init__(self, data: bytes, source_file=None):
        super().__init__(data, source_file)
        self._job_cache: Optional[RuidaJob] = None
        self._geometry_cache: Optional[Geometry] = None
        self._extents_cache: Optional[Tuple[float, float, float, float]] = None

    def _get_job(self) -> RuidaJob:
        """Parses the Ruida data and caches the resulting job."""
        if self._job_cache is None:
            parser = RuidaParser(self.raw_data)
            self._job_cache = parser.parse()
        return self._job_cache

    def _get_extents(self) -> Tuple[float, float, float, float]:
        """Gets the extents of the job, using a cache."""
        if self._extents_cache is None:
            job = self._get_job()
            self._extents_cache = job.get_extents()
        return self._extents_cache

    def get_doc_items(
        self, vector_config: Optional["TraceConfig"] = None
    ) -> Optional[List["DocItem"]]:
        # Ruida files are always vector, so vector_config is ignored.
        geometry = self._get_geometry()
        if not geometry or geometry.is_empty():
            return []

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
            wp = WorkPiece(
                source_file=self.source_file,
                renderer=RUIDA_RENDERER,
                vectors=normalized_geo,
            )
            wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
                width, height
            )
            workpieces.append(wp)

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
                return workpieces  # Fallback

            min_x = min(p[0] for p in all_corners)
            min_y = min(p[1] for p in all_corners)
            max_x = max(p[0] for p in all_corners)
            max_y = max(p[1] for p in all_corners)
            bbox_x, bbox_y = min_x, min_y
            bbox_w = max(max_x - min_x, 1e-9)
            bbox_h = max(max_y - min_y, 1e-9)

            # 2. Create group and set its matrix to match the bbox.
            group = Group(name=self.source_file.stem)
            group.matrix = Matrix.translation(bbox_x, bbox_y) @ Matrix.scale(
                bbox_w, bbox_h
            )

            # 3. Update workpiece matrices to be relative to the group.
            try:
                group_inv_matrix = group.matrix.invert()
                for wp in workpieces:
                    wp.matrix = group_inv_matrix @ wp.matrix
                # 4. Add children and return the configured group.
                group.set_children(workpieces)
                return [group]
            except np.linalg.LinAlgError:
                return workpieces  # Fallback

        elif workpieces:
            return workpieces

        return []

    def _get_geometry(self) -> Geometry:
        """
        Returns the parsed vector geometry. The coordinate system is
        canonical (Y-up, origin at bottom-left of content).
        """
        if self._geometry_cache is not None:
            return self._geometry_cache

        job = self._get_job()
        geo = Geometry()
        if not job.commands:
            self._geometry_cache = geo
            return self._geometry_cache

        _min_x, min_y, _max_x, max_y = self._get_extents()
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
        self._geometry_cache = geo
        return self._geometry_cache
