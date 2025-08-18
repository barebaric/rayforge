import logging
from typing import List, Optional, Tuple
from ...core.ops import Ops
from ...core.item import DocItem
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
        self._ops_cache: Optional[Ops] = None
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

    def get_doc_items(self) -> Optional[List["DocItem"]]:
        from ...core.workpiece import WorkPiece
        from ...core.matrix import Matrix

        ops = self._get_vector_ops()
        if not ops or ops.is_empty():
            return []

        # Ruida ops are already in mm, with origin at bottom-left of job
        min_x, min_y, max_x, max_y = ops.rect()
        width = max(max_x - min_x, 1e-9)
        height = max(max_y - min_y, 1e-9)

        # The parsed ops are already normalized relative to job extents
        wp = WorkPiece(
            source_file=self.source_file,
            renderer=RUIDA_RENDERER,
            source_ops=ops,
        )
        wp.matrix = Matrix.translation(0, 0) @ Matrix.scale(width, height)
        # Position can be adjusted later by user

        return [wp]

    def _get_vector_ops(self) -> Optional[Ops]:
        """
        Returns the parsed vector operations. The coordinate system is
        canonical (Y-up, origin at bottom-left of content).
        """
        if self._ops_cache is not None:
            return self._ops_cache

        job = self._get_job()
        if not job.commands:
            self._ops_cache = Ops()
            return self._ops_cache

        _min_x, min_y, _max_x, max_y = self._get_extents()
        y_flip_val = max_y + min_y

        ops = Ops()
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
                    ops.move_to(x, flipped_y)
                elif cmd.command_type == "Cut_Abs":
                    ops.line_to(x, flipped_y)
        self._ops_cache = ops
        return self._ops_cache
