from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Tuple, Optional
from blinker import Signal
from copy import deepcopy

from .base import PipelineStage
from ...context import get_context
from ...core.ops import Ops, ScanLinePowerCommand
from ..artifact import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    create_handle_from_dict,
)
from .workpiece_runner import make_workpiece_artifact_in_subprocess

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.matrix import Matrix
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache

logger = logging.getLogger(__name__)

WorkpieceKey = Tuple[str, str]  # (step_uid, workpiece_uid)


class WorkpieceGeneratorStage(PipelineStage):
    """
    Generates and caches base artifacts for (Step, WorkPiece) pairs.
    """

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._generation_id_map: Dict[WorkpieceKey, int] = {}
        self._active_tasks: Dict[WorkpieceKey, "Task"] = {}
        self._pending_handles: Dict[WorkpieceKey, WorkPieceArtifactHandle] = {}

        # Signals for notifying the pipeline of generation progress
        self.generation_starting = Signal()
        self.visual_chunk_available = Signal()
        self.generation_finished = Signal()

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
        logger.debug("WorkpieceGeneratorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            self._cleanup_task(key)
        self._active_tasks.clear()

    def reconcile(self, doc: "Doc"):
        """
        Synchronizes the cache with the document, generating artifacts
        for new or invalid items and cleaning up obsolete ones.
        """
        logger.debug("WorkpieceGeneratorStage reconciling...")
        all_current_pairs = {
            (step.uid, workpiece.uid)
            for layer in doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        cached_pairs = self._artifact_cache.get_all_workpiece_keys()

        for s_uid, w_uid in cached_pairs - all_current_pairs:
            self._cleanup_entry((s_uid, w_uid))

        for layer in doc.layers:
            if layer.workflow is None:
                continue
            for step in layer.workflow.steps:
                for workpiece in layer.all_workpieces:
                    if self._is_stale(step, workpiece):
                        self._launch_task(step, workpiece)

    def on_workpiece_transform_changed(self, workpiece: "WorkPiece"):
        """
        Handles transform changes. Invalidates non-scalable artifacts only
        if their size has changed, as position/rotation do not affect the
        base artifact generation for rasters.
        """
        if not workpiece.layer or not workpiece.layer.workflow:
            return
        for step in workpiece.layer.workflow.steps:
            key = (step.uid, workpiece.uid)
            handle = self._artifact_cache.get_workpiece_handle(*key)

            # Guard clauses for clarity
            if not handle or handle.is_scalable:
                continue

            # If the size is still close enough, the artifact is valid.
            if self._sizes_are_close(handle.generation_size, workpiece.size):
                continue

            # If we reach here, the artifact is non-scalable and its size has
            # meaningfully changed. Invalidate it.
            logger.debug(
                f"Invalidating non-scalable artifact for {key} due to size "
                f"change from {handle.generation_size} to {workpiece.size}."
            )
            self._cleanup_entry(key)

    def _is_stale(self, step: "Step", workpiece: "WorkPiece") -> bool:
        """
        Checks if the artifact for a (step, workpiece) pair is missing
        or invalid (e.g., due to a size change on a non-scalable item).
        """
        handle = self._artifact_cache.get_workpiece_handle(
            step.uid, workpiece.uid
        )
        if handle is None:
            return True

        if isinstance(handle, WorkPieceArtifactHandle):
            if not handle.is_scalable:
                # A non-scalable artifact is stale if its generation size
                # doesn't match the workpiece's current size (within
                # tolerance).
                if not self._sizes_are_close(
                    handle.generation_size, workpiece.size
                ):
                    return True
        return False

    def invalidate_for_step(self, step_uid: str):
        """Invalidates all workpiece artifacts associated with a step."""
        logger.debug(f"Invalidating workpiece artifacts for step '{step_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_cache.get_all_workpiece_keys()
            if k[0] == step_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """Invalidates all artifacts for a workpiece across all steps."""
        logger.debug(f"Invalidating artifacts for workpiece '{workpiece_uid}'")
        keys_to_clean = [
            k
            for k in self._artifact_cache.get_all_workpiece_keys()
            if k[1] == workpiece_uid
        ]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def _cleanup_task(self, key: WorkpieceKey):
        """Cancels a task if it's active."""
        if key in self._active_tasks:
            task = self._active_tasks.pop(key, None)
            if task:
                logger.debug(f"Cancelling active workpiece task for {key}")
                self._task_manager.cancel_task(task.key)

    def _cleanup_entry(self, key: WorkpieceKey):
        """
        Removes a workpiece cache entry, releases its resources, and
        cancels its task.
        """
        logger.debug(f"WorkpieceGeneratorStage: Cleaning up entry {key}.")
        s_uid, w_uid = key
        self._generation_id_map.pop(key, None)
        self._pending_handles.pop(key, None)
        self._cleanup_task(key)
        self._artifact_cache.invalidate_for_workpiece(s_uid, w_uid)

    def _launch_task(self, step: "Step", workpiece: "WorkPiece"):
        """Starts the asynchronous task to generate operations."""
        key = (step.uid, workpiece.uid)

        if any(s <= 0 for s in workpiece.size):
            self._cleanup_entry(key)
            return

        if key in self._active_tasks:
            logger.debug(f"Cancelling existing task for key {key}.")
            self._cleanup_task(key)

        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

        self.generation_starting.send(
            self, step=step, workpiece=workpiece, generation_id=generation_id
        )

        self._artifact_cache.invalidate_for_workpiece(step.uid, workpiece.uid)

        def when_done_callback(task: "Task"):
            self._on_task_complete(task, key, generation_id, step, workpiece)

        settings = step.get_settings()

        config = get_context().config
        if not config or not config.machine:
            logger.error("Cannot generate ops: No machine is configured.")
            return

        try:
            selected_laser = step.get_selected_laser(config.machine)
        except ValueError as e:
            logger.error(f"Cannot select laser for step '{step.name}': {e}")
            return

        world_workpiece = workpiece.in_world()
        workpiece_dict = world_workpiece.to_dict()
        renderer = workpiece.renderer
        if renderer:
            workpiece_dict["data"] = workpiece.data
            workpiece_dict["renderer_name"] = renderer.__class__.__name__

        task = self._task_manager.run_process(
            make_workpiece_artifact_in_subprocess,
            workpiece_dict,
            step.opsproducer_dict,
            step.modifiers_dicts,
            step.per_workpiece_transformers_dicts,
            selected_laser.to_dict(),
            settings,
            generation_id,
            workpiece.size,
            key=key,
            when_done=when_done_callback,
            when_event=self._on_task_event_received,
        )
        self._active_tasks[key] = task

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """Handles events from a background task."""
        key = task.key
        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")
        if not handle_dict or generation_id is None:
            return

        try:
            handle = create_handle_from_dict(handle_dict)
            if not isinstance(handle, WorkPieceArtifactHandle):
                raise TypeError("Expected a WorkPieceArtifactHandle")

            # Adopt the memory block as soon as we know about it.
            get_context().artifact_store.adopt(handle)

            if event_name == "__internal_artifact_created":
                self._pending_handles[key] = handle
                return

            if event_name == "visual_chunk_ready":
                self.visual_chunk_available.send(
                    self,
                    key=key,
                    chunk_handle=handle,
                    generation_id=generation_id,
                )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}", exc_info=True
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
        self._active_tasks.pop(key, None)

        if self._generation_id_map.get(key) != task_generation_id:
            logger.debug(f"Ignoring stale ops callback for {key}.")
            return

        if task.get_status() == "completed":
            self._handle_completed_task_result(task, key, step, workpiece)
        else:
            # Cleanup pending handle if task failed or was cancelled
            self._pending_handles.pop(key, None)
            wp_name = workpiece.name
            logger.warning(
                f"Ops generation for '{step.name}' on '{wp_name}' failed "
                f"with status: {task.get_status()}."
            )

        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=task_generation_id,
        )

    def _handle_completed_task_result(
        self,
        task: "Task",
        key: WorkpieceKey,
        step: "Step",
        workpiece: "WorkPiece",
    ):
        s_uid, w_uid = key
        artifact_store = get_context().artifact_store
        try:
            result_gen_id = task.result()
            handle = self._pending_handles.pop(key, None)

            # This case handles empty workpieces that correctly finish
            # without producing an artifact.
            if handle is None:
                logger.debug(
                    f"Task for {key} completed but produced no artifact "
                    f"handle via event."
                )
                return

            if self._generation_id_map.get(key) != result_gen_id:
                logger.warning(f"Stale result for {key}. Releasing.")
                artifact_store.release(handle)
                return

            if not handle.is_scalable:
                # After a task completes, check if its result is now
                # stale because the workpiece was resized while it ran.
                if not self._sizes_are_close(
                    handle.generation_size, workpiece.size
                ):
                    logger.info(
                        f"Result for {key} is stale due to size "
                        "change during generation. Regenerating."
                    )
                    artifact_store.release(handle)
                    self._launch_task(step, workpiece)
                    return

            self._artifact_cache.put_workpiece_handle(s_uid, w_uid, handle)

        except Exception as e:
            self._pending_handles.pop(key, None)
            logger.error(
                f"Error processing result for {key}: {e}", exc_info=True
            )

    def get_artifact(
        self,
        step_uid: str,
        workpiece_uid: str,
        workpiece_size: Tuple[float, float],
    ) -> Optional[WorkPieceArtifact]:
        """Retrieves the complete, validated artifact from the cache."""
        handle = self._artifact_cache.get_workpiece_handle(
            step_uid, workpiece_uid
        )
        if handle is None:
            return None

        if isinstance(handle, WorkPieceArtifactHandle):
            if not handle.is_scalable:
                # When fetching an artifact, ensure its generation size
                # matches the requested size within tolerance.
                if not self._sizes_are_close(
                    handle.generation_size, workpiece_size
                ):
                    return None

        artifact = get_context().artifact_store.get(handle)
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
