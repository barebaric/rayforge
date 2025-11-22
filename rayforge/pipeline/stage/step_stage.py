from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Optional
from blinker import Signal

from ...context import get_context
from ..artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    create_handle_from_dict,
)
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.step import Step
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache


logger = logging.getLogger(__name__)

StepKey = str  # step_uid


class StepPipelineStage(PipelineStage):
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
        logger.debug("StepPipelineStage shutting down.")
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
        cached_steps = self._artifact_cache.get_all_step_render_uids()
        for step_uid in cached_steps - all_current_steps:
            self._cleanup_entry(step_uid, full_invalidation=True)

        for layer in doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    if not step.visible:
                        continue
                    # Trigger assembly if the render artifact is missing.
                    if not self._artifact_cache.has_step_render_handle(
                        step.uid
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
        logger.debug(f"StepPipelineStage: Cleaning up entry {key}.")
        self._generation_id_map.pop(key, None)
        self._time_cache.pop(key, None)  # Clear the time cache
        self._cleanup_task(key)

        # The ops artifact is always stale and can be removed.
        ops_handle = self._artifact_cache.pop_step_ops_handle(key)
        if ops_handle:
            get_context().artifact_store.release(ops_handle)

        # Only remove the render artifact if this is a full invalidation
        # (e.g., the step was deleted), not a simple regeneration.
        if full_invalidation:
            render_handle = self._artifact_cache.pop_step_render_handle(key)
            if render_handle:
                logger.debug(
                    f"Popped and released stale render handle for step {key}."
                )
                get_context().artifact_store.release(render_handle)

        self._artifact_cache.invalidate_for_job()

    def _trigger_assembly(self, step: "Step"):
        """Checks dependencies and launches the assembly task if ready."""
        if not step.layer or step.uid in self._active_tasks:
            return

        config = get_context().config
        if not config:
            return
        machine = config.machine
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
            "step",
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
        generation_id = data.get("generation_id")
        # Ignore events from stale tasks
        if self._generation_id_map.get(step_uid) != generation_id:
            logger.debug(f"Ignoring stale event '{event_name}' for {step_uid}")
            return

        try:
            if event_name == "render_artifact_ready":
                handle_dict = data["handle_dict"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, StepRenderArtifactHandle):
                    raise TypeError("Expected a StepRenderArtifactHandle")

                get_context().artifact_store.adopt(handle)
                self._artifact_cache.put_step_render_handle(step_uid, handle)
                self.render_artifact_ready.send(self, step=step)

            elif event_name == "ops_artifact_ready":
                handle_dict = data["handle_dict"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, StepOpsArtifactHandle):
                    raise TypeError("Expected a StepOpsArtifactHandle")

                get_context().artifact_store.adopt(handle)
                self._artifact_cache.put_step_ops_handle(step_uid, handle)

            elif event_name == "time_estimate_ready":
                time_estimate = data["time_estimate"]
                self._time_cache[step_uid] = time_estimate
                self.time_estimate_ready.send(
                    self, step=step, time=time_estimate
                )
        except Exception as e:
            logger.error(f"Error handling task event '{event_name}': {e}")

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
                # The task now only returns the generation ID for validation
                result_gen_id = task.result()
                if self._generation_id_map.get(step_uid) != result_gen_id:
                    logger.warning(
                        f"Step assembly for {step_uid} finished with stale "
                        f"generation ID."
                    )
            except Exception as e:
                logger.error(f"Error on step assembly result: {e}")
                self._time_cache[step_uid] = -1.0  # Mark error
        else:
            logger.warning(f"Step assembly for {step_uid} failed.")
            self._time_cache[step_uid] = -1.0  # Mark error

        self.generation_finished.send(
            self, step=step, generation_id=task_generation_id
        )
