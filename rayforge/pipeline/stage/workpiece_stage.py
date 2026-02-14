from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from copy import deepcopy
from ...core.ops import Ops, ScanLinePowerCommand
from ...shared.tasker.task import Task
from ...shared.util.size import sizes_are_close
from ..artifact import WorkPieceArtifact
from ..artifact.key import ArtifactKey
from ..artifact.manager import make_composite_key
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.matrix import Matrix
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact import BaseArtifactHandle
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

    def validate_for_launch(
        self, key: ArtifactKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Validates workpiece size before launching.
        Returns True if launch should proceed, False otherwise.
        """
        if any(s <= 0 for s in workpiece.size):
            logger.warning(
                f"Skipping launch for {key}: invalid size {workpiece.size}"
            )
            self._cleanup_entry(key)
            return False
        return True

    def prepare_task_settings(
        self, step: "Step"
    ) -> Optional[Tuple[Dict, Optional[Dict]]]:
        """
        Prepares settings with machine parameters and selects laser.
        Returns (settings_dict, laser_dict) or None if preparation fails.
        """
        if not self._machine:
            logger.error("Cannot generate ops: No machine is configured.")
            return None

        settings = step.get_settings()
        settings["machine_supports_arcs"] = self._machine.supports_arcs
        settings["arc_tolerance"] = self._machine.arc_tolerance

        try:
            selected_laser = step.get_selected_laser(self._machine)
        except ValueError as e:
            logger.error(f"Cannot select laser for step '{step.name}': {e}")
            return None

        return settings, selected_laser.to_dict()

    def prepare_workpiece_dict(self, workpiece: "WorkPiece") -> Dict:
        """
        Prepares the fully-hydrated, serializable WorkPiece dictionary.
        """
        world_workpiece = workpiece.in_world()
        return world_workpiece.to_dict()

    def validate_task_completion(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task_generation_id: int,
    ) -> bool:
        """
        Validates task completion by checking the generation ID against the
        ledger. Returns True if task should be processed, False otherwise.
        """
        composite_key = make_composite_key(ledger_key, task_generation_id)
        entry = self._artifact_manager._get_ledger_entry(composite_key)
        if entry is None:
            logger.debug(
                f"[{key}] Ledger entry missing. Ignoring task completion."
            )
            return False

        if entry.generation_id != task_generation_id:
            logger.debug(
                f"[{key}] Stale generation ID {task_generation_id}. "
                f"Current: {entry.generation_id}. Ignoring."
            )
            return False

        return True

    def check_result_stale(
        self, key: ArtifactKey, workpiece: "WorkPiece", generation_id: int
    ) -> bool:
        """
        Checks if result is stale due to size change during generation.
        Returns True if stale, False otherwise.
        """
        handle = self._artifact_manager.get_workpiece_handle(
            key, generation_id
        )

        if handle and not handle.is_scalable:
            if not sizes_are_close(handle.generation_size, workpiece.size):
                logger.info(
                    f"[{key}] Result for {key} is stale due to size "
                    "change during generation. Regenerating."
                )
                return True

        return False

    def handle_artifact_created(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        handle: Optional["BaseArtifactHandle"],
        generation_id: int,
        step_uid: str,
    ):
        """
        Processes artifact_created event.
        """
        if handle is not None:
            self._artifact_manager.cache_handle(
                ledger_key, handle, generation_id
            )
        else:
            self._artifact_manager.complete_generation(
                ledger_key, generation_id, handle=None
            )

        node = self._scheduler.graph.find_node(ledger_key)
        if node is not None:
            node.state = NodeState.VALID

        self._scheduler.workpiece_artifact_adopted.send(
            self._scheduler, step_uid=step_uid, workpiece_uid=key.id
        )

    def handle_visual_chunk_ready(
        self, key: ArtifactKey, handle, generation_id: int, step_uid: str
    ):
        """
        Processes visual_chunk_ready event.
        """
        self._scheduler.visual_chunk_available.send(
            self._scheduler,
            key=key,
            chunk_handle=handle,
            generation_id=generation_id,
            step_uid=step_uid,
        )

    def handle_canceled_task(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
    ):
        """
        Handles canceled task status.
        """
        handle = self._artifact_manager.get_workpiece_handle(
            key, task_generation_id
        )
        if handle is not None:
            logger.debug(
                f"[{key}] Task was canceled but artifact already committed, "
                "sending finished signal with handle."
            )
            self._scheduler.generation_finished.send(
                self._scheduler,
                step=step,
                workpiece=workpiece,
                handle=handle,
                generation_id=task_generation_id,
                task_status="canceled",
            )
            return

        logger.debug(
            f"[{key}] Task was canceled. Marking node dirty and "
            "sending finished signal."
        )
        self._scheduler.mark_node_dirty(ledger_key)
        self._scheduler.generation_finished.send(
            self._scheduler,
            step=step,
            workpiece=workpiece,
            handle=None,
            generation_id=task_generation_id,
            task_status="canceled",
        )

    def handle_completed_task(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task: "Task",
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
        generation_id: int,
    ) -> tuple[bool, Optional["BaseArtifactHandle"]]:
        """
        Handles completed task status.
        Returns (True if processing should continue, handle) or
        (False, None) if task was relaunched due to stale result.
        """
        try:
            task.result()

            if self.check_result_stale(key, workpiece, generation_id):
                self._scheduler._launch_workpiece_task(step, workpiece)
                return False, None
        except Exception as e:
            logger.error(f"[{key}] Error processing result for {key}: {e}")

        handle = self._artifact_manager.get_workpiece_handle(
            key, generation_id
        )
        if handle is None:
            node = self._scheduler.graph.find_node(ledger_key)
            if node is not None and node.state == NodeState.VALID:
                logger.debug(
                    f"[{key}] Task completed with no handle, "
                    f"node already VALID (empty workpiece)."
                )
            elif node is not None and node.state == NodeState.PROCESSING:
                logger.debug(
                    f"[{key}] Handle not yet available, "
                    f"waiting for artifact_created event."
                )
            else:
                logger.warning(
                    f"[{key}] Task completed but node in unexpected state: "
                    f"{node.state if node else 'None'}"
                )

        return True, handle

    def handle_failed_task(
        self,
        key: ArtifactKey,
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
    ):
        """
        Handles failed task status.
        """
        wp_name = workpiece.name
        error_msg = f"Ops generation for '{step.name}' on '{wp_name}' failed."
        logger.warning(f"[{key}] {error_msg}")
        node = self._scheduler.graph.find_node(key)
        if node is not None:
            node.state = NodeState.ERROR

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
            if not sizes_are_close(handle.generation_size, workpiece_size):
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
