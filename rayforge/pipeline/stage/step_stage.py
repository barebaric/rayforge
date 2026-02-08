from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, List, Tuple
from blinker import Signal
from ..artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    BaseArtifactHandle,
)
from ..artifact.lifecycle import ArtifactLifecycle
from .base import PipelineStage

if TYPE_CHECKING:
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
        self._next_generation_id = 0
        # Local cache for accurate, post-transformer time estimates
        self._time_cache: Dict[StepKey, Optional[float]] = {}

        # Signals
        self.generation_finished = Signal()
        self.render_artifact_ready = Signal()
        self.time_estimate_ready = Signal()

    def get_estimate(self, step_uid: StepKey) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        return self._time_cache.get(step_uid)

    def shutdown(self):
        logger.debug("StepPipelineStage shutting down.")

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
        for step_uid in cached_steps:
            if step_uid not in all_current_steps:
                self._cleanup_entry(
                    step_uid, full_invalidation=True, invalidate_job=False
                )

        # Build valid keys for the ledger in format ('step', step_uid)
        valid_keys = {
            ("step", step.uid)
            for layer in doc.layers
            if layer.workflow
            for step in layer.workflow.steps
            if step.visible
        }

        # Sync the ledger with the current valid keys
        self._artifact_manager.sync_keys("step", valid_keys)

        # Query for keys that need generation
        keys_to_generate = self._artifact_manager.query_work_for_stage("step")

        # Launch tasks for each key that needs generation
        for key in keys_to_generate:
            _, step_uid = key
            step = self._find_step(doc, step_uid)
            if step:
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

    def _find_step(self, doc: "Doc", step_uid: str) -> Optional["Step"]:
        """Finds a step by its UID in the document."""
        for layer in doc.layers:
            if layer.workflow is not None:
                for step in layer.workflow.steps:
                    if step.uid == step_uid:
                        return step
        return None

    def _cleanup_entry(
        self,
        key: StepKey,
        full_invalidation: bool,
        invalidate_job: bool = True,
    ):
        """
        Removes a step artifact, clears time cache, and cancels its task.

        Args:
            key: The step key to clean up.
            full_invalidation: Whether to do a full invalidation.
            invalidate_job: Whether to invalidate the job artifact. Defaults to
                True. Set to False when cleaning up during job generation
                to avoid invalidating the job we're trying to create.
        """
        logger.debug(f"StepPipelineStage: Cleaning up entry {key}.")
        self._time_cache.pop(key, None)

        # Only remove render artifact if this is a full invalidation
        # (e.g., step was deleted), not a simple regeneration.
        if full_invalidation:
            render_handle = self._artifact_manager.pop_step_render_handle(key)
            if render_handle:
                logger.debug(
                    f"Popped and released stale render handle for step {key}."
                )
                self._artifact_manager.release_handle(render_handle)

        self._artifact_manager.invalidate(("step", key))
        if invalidate_job:
            self._artifact_manager.invalidate_for_job()

    def _validate_assembly_dependencies(self, step: "Step") -> bool:
        """Validates that assembly dependencies are met."""
        if not step.layer:
            return False
        if not self._machine:
            logger.warning(
                f"Cannot assemble step {step.uid}, no machine configured."
            )
            return False
        # Check if the step is already in PENDING state in the ledger
        ledger_key = ("step", step.uid)
        entry = self._artifact_manager._ledger.get(ledger_key)
        if entry and entry.state == ArtifactLifecycle.PENDING:
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

    def _collect_assembly_info(
        self, step: "Step"
    ) -> Tuple[Optional[list], List[BaseArtifactHandle]]:
        """
        Collects assembly info from all workpieces and retains handles.
        Returns (assembly_info, retained_handles).
        """
        assert step.layer is not None
        assembly_info = []
        retained_handles = []

        try:
            for wp in step.layer.all_workpieces:
                handle = self._artifact_manager.get_workpiece_handle(
                    step.uid, wp.uid
                )
                if handle is None:
                    raise ValueError(f"Missing handle for {wp.uid}")

                if not self._validate_handle_geometry_match(handle, wp):
                    raise ValueError(f"Geometry mismatch for {wp.uid}")

                # Retain the handle for the duration of the task
                self._artifact_manager.retain_handle(handle)
                retained_handles.append(handle)

                info = {
                    "artifact_handle_dict": handle.to_dict(),
                    "world_transform_list": wp.get_world_transform().to_list(),
                    "workpiece_dict": wp.in_world().to_dict(),
                }
                assembly_info.append(info)
        except ValueError:
            # Rollback retained handles on failure
            for handle in retained_handles:
                self._artifact_manager.release_handle(handle)
            return None, []

        return assembly_info, retained_handles

    def _prepare_assembly_task(self, step: "Step") -> tuple:
        """
        Prepares generation ID, callbacks.
        Returns (generation_id, when_done, when_event).
        """
        self._next_generation_id += 1
        generation_id = self._next_generation_id
        self._time_cache[step.uid] = None

        def when_done_callback(task: "Task"):
            self._on_assembly_complete(task, step, generation_id)

        def when_event_callback(task: "Task", event_name: str, data: dict):
            self._on_task_event(task, event_name, data, step)

        return (
            generation_id,
            when_done_callback,
            when_event_callback,
        )

    def _launch_assembly_task(
        self,
        step: "Step",
        assembly_info: list,
        generation_id: int,
        when_done,
        when_event,
    ):
        """Launches the subprocess assembly task."""
        machine = self._machine
        assert machine is not None
        from .step_runner import make_step_artifact_in_subprocess

        # Namespace the task key to prevent collisions
        task_key = ("step", step.uid)

        self._task_manager.run_process(
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
            key=task_key,
            when_done=when_done,
            when_event=when_event,
        )

    def _trigger_assembly(self, step: "Step"):
        """Checks dependencies and launches the assembly task if ready."""
        if not self._validate_assembly_dependencies(step):
            return

        assembly_info, retained_handles = self._collect_assembly_info(step)
        if not assembly_info:
            self._cleanup_entry(step.uid, full_invalidation=True)
            # Release retained handles on failure
            for handle in retained_handles:
                self._artifact_manager.release_handle(handle)
            return

        generation_id, when_done, when_event = self._prepare_assembly_task(
            step
        )

        ledger_key = ("step", step.uid)
        self._artifact_manager.mark_pending(ledger_key, generation_id)

        self._launch_assembly_task(
            step,
            assembly_info,
            generation_id,
            when_done,
            when_event,
        )

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
        # Store ops artifact in ledger for job generation to use
        self._artifact_manager.put_step_ops_handle(step_uid, handle)

    def _handle_time_estimate_ready(
        self, step_uid: str, step: "Step", time_estimate: float
    ):
        """Handles the time estimate ready event."""
        self._time_cache[step_uid] = time_estimate
        self.time_estimate_ready.send(self, step=step, time=time_estimate)

    def _on_task_event(
        self, task: "Task", event_name: str, data: dict, step: "Step"
    ):
        """Handles events broadcast from the subprocess."""
        step_uid = step.uid

        try:
            if event_name == "render_artifact_ready":
                self._handle_render_artifact_ready(
                    step_uid, step, data["handle_dict"]
                )

            elif event_name == "ops_artifact_ready":
                self._handle_ops_artifact_ready(step_uid, data["handle_dict"])

            elif event_name == "time_estimate_ready":
                self._handle_time_estimate_ready(
                    step_uid, step, data["time_estimate"]
                )
        except Exception as e:
            logger.error(f"Error handling task event '{event_name}': {e}")

    def _on_assembly_complete(
        self, task: "Task", step: "Step", task_generation_id: int
    ):
        """Callback for when a step assembly task finishes."""
        step_uid = step.uid
        ledger_key = ("step", step_uid)

        entry = self._artifact_manager._get_ledger_entry(ledger_key)
        if entry is None or entry.generation_id != task_generation_id:
            logger.debug(f"Ignoring stale step completion for {step_uid}")
            return

        if task.get_status() == "completed":
            try:
                # The task now only returns the generation ID for validation
                task.result()
                # Get the render handle and commit it
                render_handle = self._artifact_manager.get_step_render_handle(
                    step_uid
                )
                if render_handle:
                    self._artifact_manager.commit(
                        ledger_key, render_handle, task_generation_id
                    )
            except Exception as e:
                logger.error(f"Error on step assembly result: {e}")
                self._time_cache[step_uid] = -1.0  # Mark error
                error_msg = f"Step assembly for '{step.name}' failed: {e}"
                self._artifact_manager.mark_error(
                    ledger_key, error_msg, task_generation_id
                )
        else:
            logger.warning(f"Step assembly for {step_uid} failed.")
            self._time_cache[step_uid] = -1.0  # Mark error
            error_msg = f"Step assembly for '{step.name}' failed."
            self._artifact_manager.mark_error(
                ledger_key, error_msg, task_generation_id
            )

        self.generation_finished.send(
            self, step=step, generation_id=task_generation_id
        )
