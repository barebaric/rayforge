from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple
from copy import deepcopy
from blinker import Signal
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
    from ..context import GenerationContext
    from ..artifact import BaseArtifactHandle
    from ..artifact.manager import ArtifactManager

logger = logging.getLogger(__name__)


class WorkPiecePipelineStage(PipelineStage):
    """
    Provides access to workpiece artifacts and handles invalidation.
    Task launching creation, and completion handling reside here.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self.generation_starting = Signal()
        self.generation_finished = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.visual_chunk_available = Signal()

    def invalidate_for_step(self, step_uid: str):
        """Invalidates all workpiece artifacts associated with a step."""
        logger.debug(f"Invalidating workpiece artifacts for step '{step_uid}'")
        step_key = ArtifactKey.for_step(step_uid)
        keys_to_clean = self._artifact_manager.get_dependents(step_key)
        logger.debug(
            f"WorkPiecePipelineStage: keys_to_clean for step '{step_uid}': "
            f"{keys_to_clean}"
        )
        for key in keys_to_clean:
            self._cleanup_entry(key)
        self._artifact_manager.invalidate_for_step(step_key)
        self._emit_node_state(step_key, NodeState.DIRTY)

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
        entry = self._artifact_manager.get_ledger_entry(composite_key)
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

        self._emit_node_state(ledger_key, NodeState.VALID)

        self.workpiece_artifact_adopted.send(
            self, step_uid=step_uid, workpiece_uid=key.id
        )

    def handle_visual_chunk_ready(
        self, key: ArtifactKey, handle, generation_id: int, step_uid: str
    ):
        """
        Processes visual_chunk_ready event.
        """
        self.visual_chunk_available.send(
            self,
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
            self.generation_finished.send(
                self,
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
        self._emit_node_state(ledger_key, NodeState.DIRTY)
        self.generation_finished.send(
            self,
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
        context: Optional["GenerationContext"],
    ) -> tuple[bool, Optional["BaseArtifactHandle"]]:
        """
        Handles completed task status.
        Returns (True if processing should continue, handle) or
        (False, None) if task was relaunched due to stale result.
        """
        try:
            task.result()

            if self.check_result_stale(key, workpiece, generation_id):
                self.launch_task(step, workpiece, generation_id, context)
                return False, None
        except Exception as e:
            logger.error(f"[{key}] Error processing result for {key}: {e}")

        handle = self._artifact_manager.get_workpiece_handle(
            key, generation_id
        )
        if handle is None:
            logger.debug(
                f"[{key}] Task completed with no handle "
                f"(empty workpiece or pending artifact_created event)."
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
        self._emit_node_state(key, NodeState.ERROR)

    def launch_task(
        self,
        step: "Step",
        workpiece: "WorkPiece",
        generation_id: int,
        context: Optional["GenerationContext"],
    ):
        """Starts the asynchronous task to generate operations."""
        key = ArtifactKey.for_workpiece(workpiece.uid, step.uid)
        ledger_key = key

        if not self.validate_for_launch(key, workpiece):
            logger.debug(
                f"Validation failed for launch "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return

        self.generation_starting.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=generation_id,
        )

        prep_result = self.prepare_task_settings(step)
        if prep_result is None:
            logger.debug(
                f"prep_result is None for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return
        settings, laser_dict = prep_result

        workpiece_dict = self.prepare_workpiece_dict(workpiece)

        self._emit_node_state(ledger_key, NodeState.PROCESSING)

        if context is not None:
            context.add_task(key)

        self.create_and_register_task(
            key,
            ledger_key,
            workpiece_dict,
            step,
            workpiece,
            settings,
            laser_dict,
            generation_id,
            workpiece.size,
            context,
        )

    def create_and_register_task(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        workpiece_dict: Dict,
        step: "Step",
        workpiece: "WorkPiece",
        settings: Dict,
        laser_dict: Optional[Dict],
        generation_id: int,
        workpiece_size: Tuple[float, float],
        context: Optional["GenerationContext"],
    ):
        """
        Creates subprocess task and registers it in active_tasks.
        """
        from .workpiece_runner import (
            make_workpiece_artifact_in_subprocess,
        )

        task_key = key

        self._task_manager.run_process(
            make_workpiece_artifact_in_subprocess,
            self._artifact_manager.get_store(),
            workpiece_dict,
            step.opsproducer_dict,
            step.per_workpiece_transformers_dicts,
            laser_dict,
            settings,
            generation_id,
            workpiece_size,
            "workpiece",
            key=task_key,
            when_done=lambda t: self._on_task_complete(
                t,
                task_key,
                ledger_key,
                generation_id,
                step,
                workpiece,
                context,
            ),
            when_event=lambda task, event_name, data: self._on_task_event(
                task, event_name, data, step.uid
            ),
        )

    def _on_task_event(
        self, task: "Task", event_name: str, data: dict, step_uid: str
    ):
        """Handles events from a background task."""
        key = task.key
        ledger_key = key

        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")

        if generation_id is None:
            logger.error(
                f"[{key}] Task event '{event_name}' missing "
                f"generation id. Ignoring."
            )
            return

        if handle_dict is None and event_name != "artifact_created":
            logger.error(
                f"[{key}] Task event '{event_name}' missing handle. Ignoring."
            )
            return

        try:
            if handle_dict is None:
                handle = None
            else:
                handle = self._artifact_manager.adopt_artifact(
                    key, handle_dict
                )

            if event_name == "artifact_created":
                self.handle_artifact_created(
                    key, ledger_key, handle, generation_id, step_uid
                )
                return

            if event_name == "visual_chunk_ready":
                self.handle_visual_chunk_ready(
                    key, handle, generation_id, step_uid
                )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}", exc_info=True
            )

    def _on_task_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task_generation_id: int,
        step: "Step",
        workpiece: "WorkPiece",
        context: Optional["GenerationContext"],
    ):
        """Callback for when an ops generation task finishes."""
        if context is not None:
            context.task_did_finish(key)

        if not self.validate_task_completion(
            key, ledger_key, task_generation_id
        ):
            return

        task_status = task.get_status()
        logger.debug(f"[{key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            self.handle_canceled_task(
                key, ledger_key, step, workpiece, task_generation_id
            )
            return

        handle = None
        if task_status == "completed":
            continue_processing, handle = self.handle_completed_task(
                key,
                ledger_key,
                task,
                step,
                workpiece,
                task_generation_id,
                task_generation_id,
                context,
            )
            if not continue_processing:
                return
        else:
            self.handle_failed_task(
                ledger_key, step, workpiece, task_generation_id
            )

        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            handle=handle,
            generation_id=task_generation_id,
            task_status=task_status,
        )

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
