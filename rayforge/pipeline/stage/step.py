from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Optional
from blinker import Signal

from .base import PipelineStage
from ..artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    create_handle_from_dict,
)
from ..artifact.store import ArtifactStore
from ... import config

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
        # Local cache for the accurate, post-transformer time estimates
        self._time_cache: Dict[StepKey, Optional[float]] = {}

        # Signals
        self.generation_finished = Signal()
        self.render_artifact_ready = Signal()
        self.time_estimate_ready = Signal()

    @property
    def is_busy(self) -> bool:
        return bool(self._active_tasks)

    def get_estimate(self, step_uid: StepKey) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        return self._time_cache.get(step_uid)

    def shutdown(self):
        logger.debug("StepGeneratorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            self._cleanup_task(key)

    def reconcile(self, doc: "Doc"):
        """
        Triggers assembly for steps where dependencies are met and the
        artifact is missing or stale.
        """
        if not doc:
            return

        all_current_steps = {
            step.uid
            for layer in doc.layers
            if layer.workflow
            for step in layer.workflow.steps
        }
        # The source of truth is now the render handle cache.
        cached_steps = set(self._artifact_cache._step_render_handles.keys())
        for step_uid in cached_steps - all_current_steps:
            self._cleanup_entry(step_uid, full_invalidation=True)

        for layer in doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    # Trigger assembly if the render artifact is missing.
                    if (
                        step.uid
                        not in self._artifact_cache._step_render_handles
                    ):
                        self._trigger_assembly(step)

    def invalidate(self, key: StepKey):
        """Invalidates a step artifact, ensuring it will be regenerated."""
        self._cleanup_entry(key, full_invalidation=True)

    def mark_stale_and_trigger(self, step: "Step"):
        """Marks a step as stale and immediately tries to trigger assembly."""
        # When marking as stale, we do NOT do a full invalidation, to
        # prevent UI flicker. The old render artifact will be replaced
        # atomically when the new one is ready.
        self._cleanup_entry(step.uid, full_invalidation=False)
        self._trigger_assembly(step)

    def _cleanup_task(self, key: StepKey):
        """Cancels a task if it's active."""
        if key in self._active_tasks:
            task = self._active_tasks.pop(key, None)
            if task:
                logger.debug(f"Cancelling active step task for {key}")
                self._task_manager.cancel_task(task.key)

    def _cleanup_entry(self, key: StepKey, full_invalidation: bool):
        """Removes a step artifact, clears time cache, and cancels its task."""
        logger.debug(f"StepGeneratorStage: Cleaning up entry {key}.")
        self._generation_id_map.pop(key, None)
        self._time_cache.pop(key, None)  # Clear the time cache
        self._cleanup_task(key)

        # The ops artifact is always stale and can be removed.
        ops_handle = self._artifact_cache._step_ops_handles.pop(key, None)
        if ops_handle:
            ArtifactStore.release(ops_handle)

        # Only remove the render artifact if this is a full invalidation
        # (e.g., the step was deleted), not a simple regeneration.
        if full_invalidation:
            render_handle = self._artifact_cache._step_render_handles.pop(
                key, None
            )
            if render_handle:
                ArtifactStore.release(render_handle)

        self._artifact_cache.invalidate_for_job()

    def _trigger_assembly(self, step: "Step"):
        """Checks dependencies and launches the assembly task if ready."""
        if not step.layer or step.uid in self._active_tasks:
            return

        machine = config.config.machine
        if not machine:
            logger.warning(
                f"Cannot assemble step {step.uid}, no machine configured."
            )
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
            self._cleanup_entry(step.uid, full_invalidation=True)
            return

        generation_id = self._generation_id_map.get(step.uid, 0) + 1
        self._generation_id_map[step.uid] = generation_id

        # Mark time as pending in the cache
        self._time_cache[step.uid] = None

        from .step_runner import make_step_artifact_in_subprocess

        def when_done_callback(task: "Task"):
            self._on_assembly_complete(task, step, generation_id)

        # Define callback for events from subprocess
        def when_event_callback(task: "Task", event_name: str, data: dict):
            self._on_task_event(task, event_name, data, step)

        task = self._task_manager.run_process(
            make_step_artifact_in_subprocess,
            assembly_info,
            step.uid,
            generation_id,
            step.per_step_transformers_dicts,
            machine.max_cut_speed,
            machine.max_travel_speed,
            machine.acceleration,
            key=step.uid,
            when_done=when_done_callback,
            when_event=when_event_callback,  # Connect event listener
        )
        self._active_tasks[step.uid] = task

    def _on_task_event(
        self, task: "Task", event_name: str, data: dict, step: "Step"
    ):
        """Handles events broadcast from the subprocess."""
        step_uid = step.uid
        # Ignore events from stale tasks
        if self._generation_id_map.get(step_uid) != data.get("generation_id"):
            logger.debug(f"Ignoring stale time event for {step_uid}")
            return

        if event_name == "render_artifact_ready":
            try:
                # The visual artifact is ready. Store handle and notify.
                handle_dict = data["handle_dict"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, StepRenderArtifactHandle):
                    raise TypeError("Expected a StepRenderArtifactHandle")
                self._artifact_cache.put_step_render_handle(step_uid, handle)
                self.render_artifact_ready.send(self, step=step)
            except Exception as e:
                logger.error(f"Error handling render_artifact_ready: {e}")
        elif event_name == "ops_artifact_ready":
            try:
                handle_dict = data["handle_dict"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, StepOpsArtifactHandle):
                    raise TypeError("Expected a StepOpsArtifactHandle")
                self._artifact_cache.put_step_ops_handle(step_uid, handle)
            except Exception as e:
                logger.error(f"Error handling ops_artifact_ready: {e}")

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
                # The task returns the time estimate
                result = task.result()
                if not result:
                    raise ValueError("Step assembly returned no result")
                time_estimate, result_gen_id = result

                if self._generation_id_map.get(step_uid) == result_gen_id:
                    # Cache the time and notify
                    self._time_cache[step_uid] = time_estimate
                    self.time_estimate_ready.send(
                        self, step=step, time=time_estimate
                    )
            except Exception as e:
                logger.error(f"Error on step assembly result (time): {e}")
                self._time_cache[step_uid] = -1.0  # Mark error
        else:
            logger.warning(f"Step assembly for {step_uid} failed.")
            self._time_cache[step_uid] = -1.0  # Mark error

        self.generation_finished.send(
            self, step=step, generation_id=task_generation_id
        )
