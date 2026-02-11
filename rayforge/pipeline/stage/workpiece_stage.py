from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Tuple, Optional
from blinker import Signal
from copy import deepcopy
from ...core.ops import Ops, ScanLinePowerCommand
from ..artifact import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from ..artifact.key import ArtifactKey
from ..artifact.lifecycle import ArtifactLifecycle
from ..artifact.manager import extract_base_key, make_composite_key
from .base import PipelineStage
from .workpiece_runner import make_workpiece_artifact_in_subprocess

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.matrix import Matrix
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager

logger = logging.getLogger(__name__)


class WorkPiecePipelineStage(PipelineStage):
    """
    Generates and caches base artifacts for (Step, WorkPiece) pairs.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._current_generation_id = 0
        self.generation_starting = Signal()
        self.visual_chunk_available = Signal()
        self.generation_finished = Signal()
        self.generation_finished_with_status = Signal()
        self.workpiece_artifact_adopted = Signal()

    def _sizes_are_close(
        self,
        size1: Optional[Tuple[float, float]],
        size2: Optional[Tuple[float, float]],
    ) -> bool:
        """Compares two size tuples with a safe tolerance for float errors."""
        if size1 is None or size2 is None:
            return False  # If either is None, they can't be close.
        # Use an absolute tolerance suitable for physical dimensions in mm.
        # 1e-6 is one-millionth of a millimeter, robust enough for any UI op.
        return math.isclose(size1[0], size2[0], abs_tol=1e-6) and math.isclose(
            size1[1], size2[1], abs_tol=1e-6
        )

    def reconcile(self, doc: "Doc", generation_id: int):
        """
        Queries the ArtifactManager for work and launches generation tasks
        for any workpiece artifacts that are in the INITIAL state.
        """
        logger.debug("WorkPiecePipelineStage reconciling...")
        self._current_generation_id = generation_id

        # Query for keys that need generation, as declared by the Pipeline
        keys_to_generate = self._artifact_manager.query_work_for_stage(
            "workpiece", self._current_generation_id
        )
        logger.debug(
            f"WorkPiecePipelineStage: found {len(keys_to_generate)} keys "
            f"to generate: {keys_to_generate}"
        )

        # Launch tasks for each key that needs generation
        for key in keys_to_generate:
            workpiece = self._find_workpiece(doc, key.id)
            if workpiece:
                # Find the step associated with this workpiece for context
                step = self._find_step_for_workpiece(doc, workpiece)
                if step:
                    logger.debug(
                        f"WorkPiecePipelineStage: Launching task for "
                        f"step_uid={step.uid}, workpiece_uid={key.id}"
                    )
                    self._launch_task(step, workpiece)

    def _find_step(self, doc: "Doc", step_uid: str) -> Optional["Step"]:
        """Finds a step by its UID in the document."""
        for layer in doc.layers:
            if layer.workflow is not None:
                for step in layer.workflow.steps:
                    if step.uid == step_uid:
                        return step
        return None

    def _find_step_for_workpiece(
        self, doc: "Doc", workpiece: "WorkPiece"
    ) -> Optional["Step"]:
        """Finds the step associated with a workpiece in the document."""
        for layer in doc.layers:
            if layer.workflow is not None:
                for step in layer.workflow.steps:
                    if workpiece in layer.all_workpieces:
                        return step
        return None

    def _find_workpiece(
        self, doc: "Doc", workpiece_uid: str
    ) -> Optional["WorkPiece"]:
        """Finds a workpiece by its UID in the document."""
        for layer in doc.layers:
            for workpiece in layer.all_workpieces:
                if workpiece.uid == workpiece_uid:
                    return workpiece
        return None

    def _is_stale(self, step: "Step", workpiece: "WorkPiece") -> bool:
        """
        Checks if the artifact for a (step, workpiece) pair is missing
        or invalid.
        """
        handle = self._artifact_manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(workpiece.uid),
            self._current_generation_id,
        )
        if handle is None:
            # Artifact is missing, so it's stale.
            return True

        # If the artifact's content is scalable (e.g., pure vectors), it does
        # not need to be regenerated when the workpiece is resized. The
        # scaling is applied dynamically by downstream stages.
        if handle.is_scalable:
            return False

        # For non-scalable artifacts (like rasters), the content is baked to a
        # specific size. It IS stale if the workpiece's current size doesn't
        # match the size it was generated for.
        if not self._sizes_are_close(handle.generation_size, workpiece.size):
            logger.debug(
                f"Artifact size mismatch for {(step.uid, workpiece.uid)}: "
                f"handle={handle.generation_size}, wp={workpiece.size}"
            )
            return True

        # If we reach here, the artifact exists, is non-scalable, and its
        # size matches. It is not stale.
        return False

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
        self._artifact_manager.invalidate(step_key)

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

    def _validate_workpiece_for_launch(
        self, key: ArtifactKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Validates workpiece size and active tasks before launching.
        Returns True if launch should proceed, False otherwise.
        """
        if any(s <= 0 for s in workpiece.size):
            logger.warning(
                f"Skipping launch for {key}: invalid size {workpiece.size}"
            )
            self._cleanup_entry(key)
            return False

        return True

    def _prepare_generation_id(
        self,
        key: ArtifactKey,
        step: "Step",
        workpiece: "WorkPiece",
    ) -> int:
        """
        Sends generation_starting signal with the current generation ID.
        Returns the generation ID.
        """
        generation_id = self._current_generation_id

        self.generation_starting.send(
            self, step=step, workpiece=workpiece, generation_id=generation_id
        )

        return generation_id

    def _prepare_task_settings(
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

    def _prepare_workpiece_dict(self, workpiece: "WorkPiece") -> Dict:
        """
        Prepares the fully-hydrated, serializable WorkPiece dictionary.
        """
        world_workpiece = workpiece.in_world()
        return world_workpiece.to_dict()

    def _create_and_register_task(
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
    ):
        """
        Creates subprocess task and registers it in active_tasks.
        """
        self._task_manager.run_process(
            make_workpiece_artifact_in_subprocess,
            self._artifact_manager._store,
            workpiece_dict,
            step.opsproducer_dict,
            step.modifiers_dicts,
            step.per_workpiece_transformers_dicts,
            laser_dict,
            settings,
            generation_id,
            workpiece_size,
            "workpiece",
            key=key,
            when_done=lambda t: self._on_task_complete(
                t, key, ledger_key, generation_id, step, workpiece
            ),
            when_event=self._on_task_event_received,
        )

    def _launch_task(self, step: "Step", workpiece: "WorkPiece"):
        """Starts the asynchronous task to generate operations."""
        key = ArtifactKey.for_workpiece(workpiece.uid)
        ledger_key = key

        if not self._validate_workpiece_for_launch(key, workpiece):
            logger.debug(
                f"WorkPiecePipelineStage: Validation failed for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return

        generation_id = self._prepare_generation_id(key, step, workpiece)

        prep_result = self._prepare_task_settings(step)
        if prep_result is None:
            logger.debug(
                f"WorkPiecePipelineStage: prep_result is None for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return
        settings, laser_dict = prep_result

        workpiece_dict = self._prepare_workpiece_dict(workpiece)

        self._artifact_manager.mark_processing(ledger_key, generation_id)

        self._create_and_register_task(
            key,
            ledger_key,
            workpiece_dict,
            step,
            workpiece,
            settings,
            laser_dict,
            generation_id,
            workpiece.size,
        )

    def _handle_artifact_created_event(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        handle: WorkPieceArtifactHandle,
        generation_id: int,
    ):
        """
        Processes artifact_created event.
        """
        self._artifact_manager.commit_artifact(
            ledger_key, handle, generation_id
        )

        self.workpiece_artifact_adopted.send(
            self, step_uid=None, workpiece_uid=key.id
        )

    def _handle_visual_chunk_ready_event(
        self, key: ArtifactKey, handle, generation_id: int
    ):
        """
        Processes visual_chunk_ready event.
        """
        self.visual_chunk_available.send(
            self,
            key=key,
            chunk_handle=handle,
            generation_id=generation_id,
        )

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles events from a background task."""
        key = ArtifactKey.for_workpiece(task.key.id)
        ledger_key = key

        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")

        if not handle_dict or generation_id is None:
            logger.error(
                f"[{key}] Task event '{event_name}' missing handle or "
                f"generation id ({generation_id}). Ignoring."
            )
            return

        # Validate that there's a PROCESSING entry for this key.
        # We don't compare generation_ids because the subprocess sends
        # back the same generation_id it received. The commit method
        # handles this by finding any PROCESSING entry with the same base key.
        has_processing = any(
            extract_base_key(k) == ledger_key
            and e.state == ArtifactLifecycle.PROCESSING
            for k, e in self._artifact_manager._ledger.items()
        )
        if not has_processing:
            logger.debug(
                f"[{key}] No PROCESSING entry found for event '{event_name}'. "
                f"Ignoring."
            )
            return

        try:
            handle = self._artifact_manager.adopt_artifact(key, handle_dict)

            if event_name == "artifact_created":
                if not isinstance(handle, WorkPieceArtifactHandle):
                    raise TypeError("Expected a WorkPieceArtifactHandle")
                self._handle_artifact_created_event(
                    key, ledger_key, handle, generation_id
                )
                return

            if event_name == "visual_chunk_ready":
                self._handle_visual_chunk_ready_event(
                    key, handle, generation_id
                )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}", exc_info=True
            )

    def _validate_task_completion(
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

    def _handle_canceled_task(
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
        logger.debug(
            f"[{key}] Task was canceled. Sending 'finished' signal "
            "with canceled status to trigger cleanup."
        )
        self._artifact_manager.invalidate(ledger_key)
        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=task_generation_id,
            task_status="canceled",
        )

    def _check_result_stale_due_to_size(
        self, key: ArtifactKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Checks if result is stale due to size change during generation.
        Returns True if stale, False otherwise.
        """
        handle = self._artifact_manager.get_workpiece_handle(
            key, self._current_generation_id
        )

        if handle and not handle.is_scalable:
            if not self._sizes_are_close(
                handle.generation_size, workpiece.size
            ):
                logger.info(
                    f"[{key}] Result for {key} is stale due to size "
                    "change during generation. Regenerating."
                )
                return True

        return False

    def _handle_completed_task(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task: "Task",
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
    ) -> bool:
        """
        Handles completed task status.
        Returns True if processing should continue, False if task was
        relaunched due to stale result.
        """
        try:
            task.result()

            if self._check_result_stale_due_to_size(key, workpiece):
                self._launch_task(step, workpiece)
                return False
        except Exception as e:
            logger.error(f"[{key}] Error processing result for {key}: {e}")

        handle = self._artifact_manager.get_workpiece_handle(
            key, self._current_generation_id
        )
        if handle is None:
            self._artifact_manager.invalidate(ledger_key)

        return True

    def _handle_failed_task(
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
        self._artifact_manager.fail_generation(
            key, error_msg, task_generation_id
        )

    def _send_generation_finished_signal(
        self,
        key: ArtifactKey,
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
        task_status: str = "completed",
    ):
        """
        Sends generation_finished signal.
        """
        logger.debug(
            f"[{key}] Sending 'generation_finished' signal for task gen_id "
            f"{task_generation_id}."
        )
        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=task_generation_id,
            task_status=task_status,
        )

    def _on_task_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task_generation_id: int,
        step: "Step",
        workpiece: "WorkPiece",
    ):
        """Callback for when an ops generation task finishes."""
        if not self._validate_task_completion(
            key, ledger_key, task_generation_id
        ):
            return

        task_status = task.get_status()
        logger.debug(f"[{key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            self._handle_canceled_task(
                key, ledger_key, step, workpiece, task_generation_id
            )
            return

        if task_status == "completed":
            if not self._handle_completed_task(
                key, ledger_key, task, step, workpiece, task_generation_id
            ):
                return
        else:
            self._handle_failed_task(
                ledger_key, step, workpiece, task_generation_id
            )

        self._send_generation_finished_signal(
            key, step, workpiece, task_generation_id, task_status
        )

    def get_artifact(
        self,
        step_uid: str,
        workpiece_uid: str,
        workpiece_size: Tuple[float, float],
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete, validated artifact from the cache."""
        handle = self._artifact_manager.get_workpiece_handle(
            ArtifactKey.for_workpiece(workpiece_uid),
            self._current_generation_id,
        )
        if handle is None:
            return None

        # For non-scalable artifacts, the generation size must match.
        # For scalable artifacts, this check is skipped.
        if not handle.is_scalable:
            if not self._sizes_are_close(
                handle.generation_size, workpiece_size
            ):
                return None

        artifact = self._artifact_manager.get_artifact(handle)
        return artifact if isinstance(artifact, WorkPieceArtifact) else None

    def get_scaled_ops(
        self, step_uid: str, workpiece_uid: str, world_transform: "Matrix"
    ) -> Optional[Ops]:
        """
        Retrieves generated operations from the cache and scales them to the
        final world size.
        """
        final_size = world_transform.get_abs_scale()
        if any(s <= 0 for s in final_size):
            return None

        artifact = self.get_artifact(step_uid, workpiece_uid, final_size)
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
