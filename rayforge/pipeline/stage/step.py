from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict
from blinker import Signal

from .base import PipelineStage
from ..artifact import StepArtifactHandle, create_handle_from_dict
from ..artifact.store import ArtifactStore

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.step import Step
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache
    from ...shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)

StepKey = str  # step_uid


class StepGeneratorStage(PipelineStage):
    """
    A pipeline stage that assembles workpiece artifacts into a step
    artifact.
    """

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._generation_id_map: Dict[StepKey, int] = {}
        self._active_tasks: Dict[StepKey, "Task"] = {}
        self.generation_finished = Signal()

    @property
    def is_busy(self) -> bool:
        return bool(self._active_tasks)

    def shutdown(self):
        logger.debug("StepGeneratorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            self._cleanup_task(key)

    def reconcile(self, doc: "Doc"):
        """
        Triggers assembly for steps where dependencies are met and the
        artifact is missing or stale. This is primarily for initial load
        and cleanup.
        """
        if not doc:
            return

        all_current_steps = {
            step.uid
            for layer in doc.layers
            if layer.workflow
            for step in layer.workflow.steps
        }
        cached_steps = set(self._artifact_cache._step_handles.keys())
        for step_uid in cached_steps - all_current_steps:
            self._cleanup_entry(step_uid)

        for layer in doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    # Check if the step artifact is completely missing
                    if step.uid not in self._artifact_cache._step_handles:
                        self._trigger_assembly(step)

    def invalidate(self, key: StepKey):
        """
        Invalidates a step artifact, ensuring it will be regenerated.
        This does NOT invalidate upstream workpiece artifacts.
        """
        self._cleanup_entry(key)

    def mark_stale_and_trigger(self, step: "Step"):
        """
        Marks a step as stale (because a dependency changed) and immediately
        tries to trigger assembly. This is the primary entry point for
        upstream generators (Workpiece).
        """
        # Invalidate the current artifact (if it exists)
        self._cleanup_entry(step.uid)
        # Try to trigger the new assembly right away
        self._trigger_assembly(step)

    def _cleanup_task(self, key: StepKey):
        """Cancels a task if it's active."""
        if key in self._active_tasks:
            task = self._active_tasks.pop(key, None)
            if task:
                logger.debug(f"Cancelling active step task for {key}")
                self._task_manager.cancel_task(task.key)

    def _cleanup_entry(self, key: StepKey):
        """Removes a step artifact and cancels its task."""
        logger.debug(f"StepGeneratorStage: Cleaning up entry {key}.")
        self._generation_id_map.pop(key, None)
        self._cleanup_task(key)
        handle = self._artifact_cache._step_handles.pop(key, None)
        if handle:
            ArtifactStore.release(handle)
        self._artifact_cache.invalidate_for_job()

    def _trigger_assembly(self, step: "Step"):
        """
        Checks dependencies and launches the assembly task if ready.
        """
        if not step.layer or step.uid in self._active_tasks:
            return

        assembly_info = []
        for wp in step.layer.all_workpieces:
            handle = self._artifact_cache.get_workpiece_handle(
                step.uid, wp.uid
            )
            if handle is None:
                return  # A dependency is not ready; abort.

            info = {
                "artifact_handle_dict": handle.to_dict(),
                "world_transform_list": wp.get_world_transform().to_list(),
                "workpiece_dict": wp.in_world().to_dict(),
            }
            assembly_info.append(info)

        if not assembly_info:
            self._artifact_cache.invalidate_for_step(step.uid)
            return

        generation_id = self._generation_id_map.get(step.uid, 0) + 1
        self._generation_id_map[step.uid] = generation_id

        from ..step_assembler import run_step_assembly_in_subprocess

        def when_done_callback(task: "Task"):
            self._on_assembly_complete(task, step, generation_id)

        task = self._task_manager.run_process(
            run_step_assembly_in_subprocess,
            assembly_info,
            step.uid,
            generation_id,
            step.per_step_transformers_dicts,
            key=step.uid,
            when_done=when_done_callback,
        )
        self._active_tasks[step.uid] = task

    def _on_assembly_complete(
        self, task: "Task", step: "Step", task_generation_id: int
    ):
        """Callback for when a step assembly task finishes."""
        step_uid = step.uid
        self._active_tasks.pop(step_uid, None)

        if self._generation_id_map.get(step_uid) != task_generation_id:
            return

        if task.get_status() == "completed":
            try:
                result = task.result()
                if not result:
                    raise ValueError("Step assembly returned no result")
                handle_dict, result_gen_id = result
                if self._generation_id_map.get(step_uid) == result_gen_id:
                    handle = create_handle_from_dict(handle_dict)
                    if not isinstance(handle, StepArtifactHandle):
                        raise TypeError("Expected a StepArtifactHandle")
                    self._artifact_cache.put_step_handle(step_uid, handle)
            except Exception as e:
                logger.error(f"Error on step assembly result: {e}")
        else:
            logger.warning(f"Step assembly for {step_uid} failed.")

        self.generation_finished.send(
            self, step=step, generation_id=task_generation_id
        )
