from __future__ import annotations
import logging
import math
import multiprocessing as mp
from typing import TYPE_CHECKING, Dict, Tuple, Optional
from blinker import Signal
from copy import deepcopy

from .base import PipelineStage
from ...core.ops import Ops, ScanLinePowerCommand
from ..artifact import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
)
from .workpiece_runner import make_workpiece_artifact_in_subprocess

if TYPE_CHECKING:
    import threading
    from ...core.doc import Doc
    from ...core.matrix import Matrix
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager

logger = logging.getLogger(__name__)

WorkpieceKey = Tuple[str, str]  # (step_uid, workpiece_uid)


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
        self._generation_id_map: Dict[WorkpieceKey, int] = {}
        self._active_tasks: Dict[WorkpieceKey, "Task"] = {}
        self._adoption_events: Dict[WorkpieceKey, "threading.Event"] = {}

        # Signals for notifying the pipeline of generation progress
        self.generation_starting = Signal()
        self.visual_chunk_available = Signal()
        self.generation_finished = Signal()
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

    @property
    def is_busy(self) -> bool:
        return bool(self._active_tasks)

    def shutdown(self):
        logger.debug("WorkPiecePipelineStage shutting down.")
        for key in list(self._active_tasks.keys()):
            self._cleanup_task(key)
        self._active_tasks.clear()
        self._adoption_events.clear()

    def reconcile(self, doc: "Doc"):
        """
        Synchronizes the cache with the document, generating artifacts
        for new or invalid items and cleaning up obsolete ones.
        """
        logger.debug("WorkPiecePipelineStage reconciling...")
        all_current_pairs = {
            (step.uid, workpiece.uid)
            for layer in doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        cached_pairs = self._artifact_manager.get_all_workpiece_keys()

        # Clean up artifacts for (step, workpiece) pairs that no longer exist
        for s_uid, w_uid in cached_pairs - all_current_pairs:
            self._cleanup_entry((s_uid, w_uid))

        # Check all valid pairs and generate artifacts for those that are stale
        for layer in doc.layers:
            if layer.workflow is None:
                continue
            for step in layer.workflow.steps:
                if not step.visible:
                    continue
                for workpiece in layer.all_workpieces:
                    if self._is_stale(step, workpiece):
                        self._launch_task(step, workpiece)

    def _is_stale(self, step: "Step", workpiece: "WorkPiece") -> bool:
        """
        Checks if the artifact for a (step, workpiece) pair is missing
        or invalid.
        """
        handle = self._artifact_manager.get_workpiece_handle(
            step.uid, workpiece.uid
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
            return True

        # If we reach here, the artifact exists, is non-scalable, and its
        # size matches. It is not stale.
        return False

    def invalidate_for_step(self, step_uid: str):
        """Invalidates all workpiece artifacts associated with a step."""
        logger.debug(f"Invalidating workpiece artifacts for step '{step_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_manager.get_all_workpiece_keys()
            if k[0] == step_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """Invalidates all artifacts for a workpiece across all steps."""
        logger.debug(f"Invalidating artifacts for workpiece '{workpiece_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_manager.get_all_workpiece_keys()
            if k[1] == workpiece_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def _cleanup_task(self, key: WorkpieceKey):
        """
        Requests cancellation of an active task but does NOT remove it from
        the active_tasks dict. The removal is handled by _on_task_complete.
        """
        if key in self._active_tasks:
            task = self._active_tasks[key]
            logger.debug(f"Requesting cancellation for active task {key}")
            self._task_manager.cancel_task(task.key)
        self._adoption_events.pop(key, None)

    def _cleanup_entry(self, key: WorkpieceKey):
        """
        Removes a workpiece cache entry, releases its resources, and
        requests cancellation of its task.
        """
        logger.debug(f"WorkPiecePipelineStage: Cleaning up entry {key}.")
        s_uid, w_uid = key

        # NOTE: We do NOT pop the generation ID here. If we remove it, the next
        # task starts at 1. If an event from the old task (ID 1) arrives late,
        # it matches the new ID (1) and corrupts the state. By keeping the
        # ID in the map, the next task becomes 2, correctly invalidating 1.
        # self._generation_id_map.pop(key, None)

        self._cleanup_task(key)
        self._artifact_manager.invalidate_for_workpiece(s_uid, w_uid)

    def _validate_workpiece_for_launch(
        self, key: WorkpieceKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Validates workpiece size and active tasks before launching.
        Returns True if launch should proceed, False otherwise.
        """
        if any(s <= 0 for s in workpiece.size):
            self._cleanup_entry(key)
            return False

        if key in self._active_tasks:
            logger.debug(
                f"Requesting cancellation of existing task for key {key}."
            )
            self._cleanup_task(key)

        return True

    def _prepare_generation_id(
        self,
        key: WorkpieceKey,
        step: "Step",
        workpiece: "WorkPiece",
    ) -> int:
        """
        Generates new generation ID and sends generation_starting signal.
        Returns the new generation ID.
        """
        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

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

    def _create_adoption_event(self, key: WorkpieceKey) -> "threading.Event":
        """
        Creates an adoption event for the handshake protocol.
        """
        manager = mp.Manager()
        adoption_event = manager.Event()
        self._adoption_events[key] = adoption_event
        return adoption_event

    def _create_and_register_task(
        self,
        key: WorkpieceKey,
        workpiece_dict: Dict,
        step: "Step",
        workpiece: "WorkPiece",
        settings: Dict,
        laser_dict: Optional[Dict],
        generation_id: int,
        workpiece_size: Tuple[float, float],
        adoption_event: "threading.Event",
    ):
        """
        Creates subprocess task and registers it in active_tasks.
        """
        task = self._task_manager.run_process(
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
            adoption_event=adoption_event,
            key=key,
            when_done=lambda t: self._on_task_complete(
                t, key, generation_id, step, workpiece
            ),
            when_event=self._on_task_event_received,
        )
        self._active_tasks[key] = task

    def _launch_task(self, step: "Step", workpiece: "WorkPiece"):
        """Starts the asynchronous task to generate operations."""
        key = (step.uid, workpiece.uid)

        if not self._validate_workpiece_for_launch(key, workpiece):
            return

        generation_id = self._prepare_generation_id(key, step, workpiece)

        prep_result = self._prepare_task_settings(step)
        if prep_result is None:
            return
        settings, laser_dict = prep_result

        workpiece_dict = self._prepare_workpiece_dict(workpiece)
        adoption_event = self._create_adoption_event(key)

        self._create_and_register_task(
            key,
            workpiece_dict,
            step,
            workpiece,
            settings,
            laser_dict,
            generation_id,
            workpiece.size,
            adoption_event,
        )

    def _validate_task_event(
        self, key: WorkpieceKey, generation_id: Optional[int]
    ) -> Tuple[bool, bool]:
        """
        Validates task event generation ID.
        Returns (is_valid, is_stale).
        """
        if generation_id is None:
            return False, False

        current_id = self._generation_id_map.get(key)
        is_stale = current_id != generation_id
        return not is_stale, is_stale

    def _cleanup_stale_artifact(self, key: WorkpieceKey, handle_dict: Dict):
        """
        Cleans up orphaned artifact from stale event.
        """
        s_uid, w_uid = key
        try:
            stale_handle = self._artifact_manager.adopt_artifact(
                (s_uid, w_uid), handle_dict
            )
            self._artifact_manager.release_handle(stale_handle)
        except Exception as e:
            logger.error(f"Error cleaning up stale artifact: {e}")

    def _handle_artifact_created_event(
        self, key: WorkpieceKey, handle: WorkPieceArtifactHandle
    ):
        """
        Processes artifact_created event.
        """
        s_uid, w_uid = key
        self._artifact_manager.put_workpiece_handle(s_uid, w_uid, handle)

        self._set_adoption_event(key)

        self.workpiece_artifact_adopted.send(
            self, step_uid=s_uid, workpiece_uid=w_uid
        )

    def _handle_visual_chunk_ready_event(
        self, key: WorkpieceKey, handle, generation_id: int
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

    def _set_adoption_event(self, key: WorkpieceKey):
        """
        Signals the worker that we've adopted the artifact.
        """
        adoption_event = self._adoption_events.get(key)
        if adoption_event is not None:
            adoption_event.set()

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles events from a background task."""
        key = task.key
        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")

        if not handle_dict or generation_id is None:
            return

        is_valid, is_stale = self._validate_task_event(key, generation_id)

        if is_stale:
            logger.debug(
                f"Received stale event '{event_name}' for {key}. "
                "Cleaning up orphaned artifact."
            )
            self._cleanup_stale_artifact(key, handle_dict)
            return

        s_uid, w_uid = key
        try:
            handle = self._artifact_manager.adopt_artifact(
                (s_uid, w_uid), handle_dict
            )

            if event_name == "artifact_created":
                if not isinstance(handle, WorkPieceArtifactHandle):
                    raise TypeError("Expected a WorkPieceArtifactHandle")
                self._handle_artifact_created_event(key, handle)
                return

            if event_name == "visual_chunk_ready":
                self._handle_visual_chunk_ready_event(
                    key, handle, generation_id
                )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}", exc_info=True
            )
            self._set_adoption_event(key)

    def _validate_task_completion(
        self, key: WorkpieceKey, task: "Task", task_generation_id: int
    ) -> bool:
        """
        Validates task completion: checks if task is active and generation IDs
        match. Returns True if task should be processed, False otherwise.
        """
        current_expected_id = self._generation_id_map.get(key)
        active_task = self._active_tasks.get(key)
        active_task_id = id(active_task) if active_task else "None"

        logger.debug(
            f"[{key}] _on_task_complete called for task {id(task)}. "
            f"Task Gen ID: {task_generation_id}, "
            f"Expected Gen ID: {current_expected_id}. "
            f"Active task in slot: {active_task_id}. "
            f"Is this the active task? {task is active_task}"
        )

        if self._active_tasks.get(key) is task:
            self._active_tasks.pop(key, None)
            self._adoption_events.pop(key, None)
            logger.debug(
                f"[{key}] Popped active task {id(task)} from tracking."
            )
        else:
            logger.debug(
                f"[{key}] Ignoring 'when_done' for task {id(task)} because it "
                "is not the currently active task."
            )
            return False

        if current_expected_id != task_generation_id:
            logger.debug(
                f"[{key}] Ignoring stale ops callback. "
                f"Task ID {task_generation_id} "
                f"does not match expected ID {current_expected_id}."
            )
            return False

        return True

    def _handle_canceled_task(
        self,
        key: WorkpieceKey,
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
        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=task_generation_id,
        )

    def _check_result_stale_due_to_size(
        self, key: WorkpieceKey, workpiece: "WorkPiece"
    ) -> bool:
        """
        Checks if result is stale due to size change during generation.
        Returns True if stale, False otherwise.
        """
        s_uid, w_uid = key
        handle = self._artifact_manager.get_workpiece_handle(s_uid, w_uid)

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
        key: WorkpieceKey,
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
        logger.debug(f"[{key}] Task completed. Processing result.")
        try:
            result_gen_id = task.result()
            logger.debug(
                f"[{key}] Task result (gen_id): {result_gen_id}. "
                f"Re-checking against expected ID: "
                f"{self._generation_id_map.get(key)}"
            )
            if self._generation_id_map.get(key) != result_gen_id:
                logger.warning(
                    f"[{key}] Stale result for {key}. Invalidating."
                )
                self._cleanup_entry(key)
                return False

            if self._check_result_stale_due_to_size(key, workpiece):
                self._launch_task(step, workpiece)
                return False
        except Exception as e:
            logger.error(f"[{key}] Error processing result for {key}: {e}")

        return True

    def _handle_failed_task(
        self, key: WorkpieceKey, step: "Step", workpiece: "WorkPiece"
    ):
        """
        Handles failed task status.
        """
        wp_name = workpiece.name
        logger.warning(
            f"[{key}] Ops generation for '{step.name}' on '{wp_name}' failed."
        )
        self._cleanup_entry(key)

    def _send_generation_finished_signal(
        self,
        key: WorkpieceKey,
        step: "Step",
        workpiece: "WorkPiece",
        task_generation_id: int,
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
        )

    def _on_task_complete(
        self,
        task: "Task",
        key: WorkpieceKey,
        task_generation_id: int,
        step: "Step",
        workpiece: "WorkPiece",
    ):
        """Callback for when an ops generation task finishes."""
        if not self._validate_task_completion(key, task, task_generation_id):
            return

        task_status = task.get_status()
        logger.debug(f"[{key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            self._handle_canceled_task(
                key, step, workpiece, task_generation_id
            )
            return

        if task_status == "completed":
            if not self._handle_completed_task(
                key, task, step, workpiece, task_generation_id
            ):
                return
        else:
            self._handle_failed_task(key, step, workpiece)

        self._send_generation_finished_signal(
            key, step, workpiece, task_generation_id
        )

    def get_artifact(
        self,
        step_uid: str,
        workpiece_uid: str,
        workpiece_size: Tuple[float, float],
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete, validated artifact from the cache."""
        handle = self._artifact_manager.get_workpiece_handle(
            step_uid, workpiece_uid
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
            f"Returning final ops for key {(step_uid, workpiece_uid)} with "
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
