from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Tuple, Optional
from copy import deepcopy
from ...core.ops import Ops, ScanLinePowerCommand
from ..artifact import WorkPieceArtifact
from ..artifact.key import ArtifactKey
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.matrix import Matrix
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager
    from ..dag.scheduler import DagScheduler

logger = logging.getLogger(__name__)


class WorkPiecePipelineStage(PipelineStage):
    """
    Provides access to workpiece artifacts and handles invalidation.
    Task launching is delegated to the DagScheduler.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
        scheduler: "DagScheduler",
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._scheduler = scheduler

    def _sizes_are_close(
        self,
        size1: Optional[Tuple[float, float]],
        size2: Optional[Tuple[float, float]],
    ) -> bool:
        """Compares two size tuples with a safe tolerance for float errors."""
        if size1 is None or size2 is None:
            return False
        return math.isclose(size1[0], size2[0], abs_tol=1e-6) and math.isclose(
            size1[1], size2[1], abs_tol=1e-6
        )

    def invalidate_for_step(self, step_uid: str):
        """Invalidates all workpiece artifacts associated with a step."""
        logger.debug(f"Invalidating workpiece artifacts for step '{step_uid}'")
        step_key = ArtifactKey.for_step(step_uid)
        keys_to_clean = self._artifact_manager._get_dependents(step_key)
        logger.debug(
            f"WorkPiecePipelineStage: keys_to_clean for step '{step_uid}': "
            f"{keys_to_clean}"
        )
        for key in keys_to_clean:
            self._cleanup_entry(key)
        self._artifact_manager.invalidate_for_step(step_key)
        self._scheduler.mark_node_dirty(step_key)

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """Invalidates all artifacts for a workpiece across all steps."""
        logger.debug(f"Invalidating artifacts for workpiece '{workpiece_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_manager.get_all_workpiece_keys()
            if k.id == workpiece_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def _cleanup_entry(self, key: ArtifactKey):
        """
        Removes a workpiece cache entry, releases its resources, and
        requests cancellation of its task.
        """
        logger.debug(f"WorkPiecePipelineStage: Cleaning up entry {key}.")
        task = self._task_manager.get_task(key)
        if task and task.is_running():
            logger.debug(f"Task {key} is running, canceling it")
            self._task_manager.cancel_task(key)
        self._artifact_manager.invalidate_for_workpiece(key)

    def get_artifact(
        self,
        step_uid: str,
        workpiece_uid: str,
        workpiece_size: Tuple[float, float],
        generation_id: int,
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete, validated artifact from the cache."""
        handle = self._artifact_manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(workpiece_uid, step_uid),
            generation_id,
        )
        if handle is None:
            return None

        if not handle.is_scalable:
            if not self._sizes_are_close(
                handle.generation_size, workpiece_size
            ):
                return None

        artifact = self._artifact_manager.get_artifact(handle)
        return artifact if isinstance(artifact, WorkPieceArtifact) else None

    def get_scaled_ops(
        self,
        step_uid: str,
        workpiece_uid: str,
        world_transform: "Matrix",
        generation_id: int,
    ) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache and scales them to the
        final world size.
        """
        final_size = world_transform.get_abs_scale()
        if any(s <= 0 for s in final_size):
            return None

        artifact = self.get_artifact(
            step_uid, workpiece_uid, final_size, generation_id
        )
        if artifact is None:
            return None

        ops = deepcopy(artifact.ops)
        if artifact.is_scalable:
            self._scale_ops_to_final_size(ops, artifact, final_size)

        scanline_count = sum(
            1 for cmd in ops.commands if isinstance(cmd, ScanLinePowerCommand)
        )
        logger.debug(
            f"Returning final ops for workpiece '{workpiece_uid}' with "
            f"{scanline_count} ScanLinePowerCommands."
        )
        return ops

    def _scale_ops_to_final_size(
        self,
        ops: Ops,
        artifact: WorkPieceArtifact,
        final_size_mm: Tuple[float, float],
    ):
        """
        Scales an Ops object from its source coordinate system to the
        provided final physical size in millimeters.
        """
        if not artifact.source_dimensions:
            logger.warning(
                "Cannot scale ops: artifact is missing source size."
            )
            return

        source_width, source_height = artifact.source_dimensions
        final_width_mm, final_height_mm = final_size_mm
        scale_x = final_width_mm / source_width if source_width > 1e-9 else 1.0
        scale_y = (
            final_height_mm / source_height if source_height > 1e-9 else 1.0
        )

        if not (math.isclose(scale_x, 1.0) and math.isclose(scale_y, 1.0)):
            logger.debug(f"Scaling ops by ({scale_x}, {scale_y})")
            ops.scale(scale_x, scale_y)
