import logging
from typing import List, Optional, Tuple
from ...core.item import DocItem
from ...core.geometry import Geometry
from ...core.vectorization_config import TraceConfig
from ..base_importer import Importer
from .renderer import RUIDA_RENDERER
from .parser import RuidaParser
from .job import RuidaJob

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
        from ...core.workpiece import WorkPiece
        from ...core.matrix import Matrix

        # Ruida files are always vector, so vector_config is ignored.
        geometry = self._get_geometry()
        if not geometry or geometry.is_empty():
            return []

        # Ruida data is already in mm, with origin at bottom-left of job
        min_x, min_y, max_x, max_y = geometry.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        # The parsed geometry is already normalized relative to job extents
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=RUIDA_RENDERER,
            vectors=geometry,
        )
        wp.matrix = Matrix.translation(0, 0) @ Matrix.scale(width, height)
        # Position can be adjusted later by user

        return [wp]

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
