from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional
import multiprocessing as mp
from contextlib import ExitStack
from blinker import Signal
from ..artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
)
from .base import PipelineStage

if TYPE_CHECKING:
    import threading
    from ...core.doc import Doc
    from ...core.step import Step
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)

StepKey = str  # step_uid


class StepPipelineStage(PipelineStage):
    """
    A pipeline stage that assembles workpiece artifacts into a step
    artifact.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._generation_id_map: Dict[StepKey, int] = {}
        self._active_tasks: Dict[StepKey, "Task"] = {}
        self._adoption_events: Dict[StepKey, "threading.Event"] = {}
        # Local cache for accurate, post-transformer time estimates
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
        self._adoption_events.clear()

    def reconcile(self, doc: "Doc"):
        """
        Triggers assembly for steps where dependencies are met and
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
        cached_steps = self._artifact_manager.get_all_step_render_uids()
        for step_uid in cached_steps - all_current_steps:
            self._cleanup_entry(step_uid, full_invalidation=True)

        for layer in doc.layers:
            if layer.workflow:
                for step in layer.workflow.steps:
                    if not step.visible:
                        continue
                    # Trigger assembly if render artifact is missing.
                    if not self._artifact_manager.has_step_render_handle(
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
        self._adoption_events.pop(key, None)

    def _cleanup_entry(self, key: StepKey, full_invalidation: bool):
        """Removes a step artifact, clears time cache, and cancels its task."""
        logger.debug(f"StepPipelineStage: Cleaning up entry {key}.")
        self._generation_id_map.pop(key, None)
        self._time_cache.pop(key, None)
        self._cleanup_task(key)

        # The ops artifact is always stale and can be removed.
        ops_handle = self._artifact_manager.pop_step_ops_handle(key)
        self._artifact_manager.release_handle(ops_handle)

        # Only remove render artifact if this is a full invalidation
        # (e.g., step was deleted), not a simple regeneration.
        if full_invalidation:
            render_handle = self._artifact_manager.pop_step_render_handle(key)
            if render_handle:
                logger.debug(
                    f"Popped and released stale render handle for step {key}."
                )
                self._artifact_manager.release_handle(render_handle)

        self._artifact_manager.invalidate_for_job()

    def _validate_assembly_dependencies(self, step: "Step") -> bool:
        """Validates that assembly dependencies are met."""
        if not step.layer:
            return False
        if step.uid in self._active_tasks:
            return False
        if not self._machine:
            logger.warning(
                f"Cannot assemble step {step.uid}, no machine configured."
            )
            return False
        return True

    def _validate_handle_geometry_match(self, handle, workpiece) -> bool:
        """
        Validates that handle geometry matches current workpiece size.
        Returns True if geometry matches or handle is scalable.
        """
        if handle.is_scalable:
            return True
        hw, hh = handle.generation_size
        ww, wh = workpiece.size
        return math.isclose(hw, ww, abs_tol=1e-6) and math.isclose(
            hh, wh, abs_tol=1e-6
        )

    def _collect_assembly_info(self, step: "Step") -> Optional[list]:
        """
        Collects assembly info from all workpieces.
        Returns None if any dependency is not ready.
        """
        assert step.layer is not None
        assembly_info = []
        with ExitStack() as stack:
            for wp in step.layer.all_workpieces:
                handle = self._artifact_manager.get_workpiece_handle(
                    step.uid, wp.uid
                )
                if handle is None:
                    return None

                if not self._validate_handle_geometry_match(handle, wp):
                    return None

                stack.enter_context(
                    self._artifact_manager.checkout((step.uid, wp.uid))
                )

                info = {
                    "artifact_handle_dict": handle.to_dict(),
                    "world_transform_list": wp.get_world_transform().to_list(),
                    "workpiece_dict": wp.in_world().to_dict(),
                }
                assembly_info.append(info)
        return assembly_info

    def _prepare_assembly_task(
        self, step: "Step", assembly_info: list
    ) -> tuple:
        """
        Prepares generation ID, callbacks, and adoption event.
        Returns (generation_id, when_done, when_event, adoption_event).
        """
        generation_id = self._generation_id_map.get(step.uid, 0) + 1
        self._generation_id_map[step.uid] = generation_id
        self._time_cache[step.uid] = None

        def when_done_callback(task: "Task"):
            self._on_assembly_complete(task, step, generation_id)

        def when_event_callback(task: "Task", event_name: str, data: dict):
            self._on_task_event(task, event_name, data, step)

        manager = mp.Manager()
        adoption_event = manager.Event()
        self._adoption_events[step.uid] = adoption_event

        return (
            generation_id,
            when_done_callback,
            when_event_callback,
            adoption_event,
        )

    def _launch_assembly_task(
        self,
        step: "Step",
        assembly_info: list,
        generation_id: int,
        when_done,
        when_event,
        adoption_event,
    ):
        """Launches the subprocess assembly task."""
        machine = self._machine
        assert machine is not None
        from .step_runner import make_step_artifact_in_subprocess

        task = self._task_manager.run_process(
            make_step_artifact_in_subprocess,
            self._artifact_manager._store,
            assembly_info,
            step.uid,
            generation_id,
            step.per_step_transformers_dicts,
            machine.max_cut_speed,
            machine.max_travel_speed,
            machine.acceleration,
            "step",
            adoption_event=adoption_event,
            key=step.uid,
            when_done=when_done,
            when_event=when_event,
        )
        self._active_tasks[step.uid] = task

    def _trigger_assembly(self, step: "Step"):
        """Checks dependencies and launches the assembly task if ready."""
        if not self._validate_assembly_dependencies(step):
            return

        assembly_info = self._collect_assembly_info(step)
        if not assembly_info:
            self._cleanup_entry(step.uid, full_invalidation=True)
            return

        generation_id, when_done, when_event, adoption_event = (
            self._prepare_assembly_task(step, assembly_info)
        )

        self._launch_assembly_task(
            step,
            assembly_info,
            generation_id,
            when_done,
            when_event,
            adoption_event,
        )

    def _is_stale_generation_id(
        self, step_uid: str, generation_id: Optional[int]
    ) -> bool:
        """Checks if the generation ID is stale."""
        return self._generation_id_map.get(step_uid) != generation_id

    def _handle_render_artifact_ready(
        self, step_uid: str, step: "Step", handle_dict: dict
    ):
        """Handles the render artifact ready event."""
        handle = self._artifact_manager.adopt_artifact(step_uid, handle_dict)
        if not isinstance(handle, StepRenderArtifactHandle):
            raise TypeError("Expected a StepRenderArtifactHandle")

        self._artifact_manager.put_step_render_handle(step_uid, handle)
        self.render_artifact_ready.send(self, step=step)

    def _handle_ops_artifact_ready(self, step_uid: str, handle_dict: dict):
        """Handles the ops artifact ready event."""
        handle = self._artifact_manager.adopt_artifact(step_uid, handle_dict)
        if not isinstance(handle, StepOpsArtifactHandle):
            raise TypeError("Expected a StepOpsArtifactHandle")

        self._artifact_manager.put_step_ops_handle(step_uid, handle)

    def _handle_time_estimate_ready(
        self, step_uid: str, step: "Step", time_estimate: float
    ):
        """Handles the time estimate ready event."""
        self._time_cache[step_uid] = time_estimate
        self.time_estimate_ready.send(self, step=step, time=time_estimate)

    def _set_adoption_event(self, step_uid: str):
        """Sets the adoption event to unblock the worker."""
        adoption_event = self._adoption_events.get(step_uid)
        if adoption_event is not None:
            adoption_event.set()

    def _on_task_event(
        self, task: "Task", event_name: str, data: dict, step: "Step"
    ):
        """Handles events broadcast from the subprocess."""
        step_uid = step.uid
        generation_id = data.get("generation_id")

        if self._is_stale_generation_id(step_uid, generation_id):
            logger.debug(f"Ignoring stale event '{event_name}' for {step_uid}")
            return

        try:
            if event_name == "render_artifact_ready":
                self._handle_render_artifact_ready(
                    step_uid, step, data["handle_dict"]
                )

            elif event_name == "ops_artifact_ready":
                self._handle_ops_artifact_ready(step_uid, data["handle_dict"])
                self._set_adoption_event(step_uid)

            elif event_name == "time_estimate_ready":
                self._handle_time_estimate_ready(
                    step_uid, step, data["time_estimate"]
                )
        except Exception as e:
            logger.error(f"Error handling task event '{event_name}': {e}")
            self._set_adoption_event(step_uid)

    def _on_assembly_complete(
        self, task: "Task", step: "Step", task_generation_id: int
    ):
        """Callback for when a step assembly task finishes."""
        step_uid = step.uid
        self._active_tasks.pop(step_uid, None)
        self._adoption_events.pop(step_uid, None)

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
