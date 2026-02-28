from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from blinker import Signal
from ...shared.util.size import sizes_are_close
from ...shared.tasker.task import Task
from ..artifact import StepOpsArtifactHandle, StepRenderArtifactHandle
from ..artifact.key import ArtifactKey
from ..artifact.manager import StaleGenerationError
from ..artifact.store import SharedMemoryNotFoundError
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact import BaseArtifactHandle
    from ..artifact.manager import ArtifactManager
    from ..context import GenerationContext


logger = logging.getLogger(__name__)

StepKey = str


class StepPipelineStage(PipelineStage):
    """
    Provides access to step artifacts and handles invalidation.
    Task launching, creation, and completion handling reside here.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._time_cache: Dict[StepKey, Optional[float]] = {}
        self._retained_handles_by_task: Dict[
            int, List["BaseArtifactHandle"]
        ] = {}

        self.generation_finished = Signal()
        self.assembly_starting = Signal()
        self.render_artifact_ready = Signal()
        self.time_estimate_ready = Signal()

    def handle_render_artifact_ready(
        self, step_uid: str, step: "Step", handle: BaseArtifactHandle
    ):
        """Handles the render artifact ready event."""
        if not isinstance(handle, StepRenderArtifactHandle):
            raise TypeError("Expected a StepRenderArtifactHandle")

        self._artifact_manager.put_step_render_handle(step_uid, handle)
        self.render_artifact_ready.send(self, step=step)

    def handle_ops_artifact_ready(
        self,
        step_uid: str,
        step: "Step",
        handle: BaseArtifactHandle,
        generation_id: int,
    ):
        """Handles the ops artifact ready event."""
        if not isinstance(handle, StepOpsArtifactHandle):
            raise TypeError("Expected a StepOpsArtifactHandle")
        self._artifact_manager.put_step_ops_handle(
            ArtifactKey.for_step(step_uid), handle, generation_id
        )

    def handle_time_estimate_ready(
        self, step_uid: str, step: "Step", time_estimate: float
    ):
        """Handles the time estimate ready event."""
        self._time_cache[step_uid] = time_estimate
        self.time_estimate_ready.send(self, step=step, time=time_estimate)

    def handle_task_event(
        self, task: "Task", event_name: str, data: dict, step: "Step"
    ):
        """Handles events broadcast from the subprocess."""
        step_uid = step.uid
        ledger_key = ArtifactKey.for_step(step_uid)

        generation_id = data.get("generation_id")
        if generation_id is None:
            return

        handle_dict = data.get("handle_dict")
        is_deleted = step.layer is None
        is_stale = not self._artifact_manager.is_generation_current(
            ledger_key, generation_id
        )

        try:
            if event_name == "time_estimate_ready":
                if not is_deleted and not is_stale:
                    self.handle_time_estimate_ready(
                        step_uid, step, data["time_estimate"]
                    )
                return

            if not handle_dict:
                return

            # safe_adoption checks staleness automatically
            with self._artifact_manager.safe_adoption(
                ledger_key, handle_dict
            ) as handle:
                if event_name == "render_artifact_ready":
                    self.handle_render_artifact_ready(step_uid, step, handle)
                elif event_name == "ops_artifact_ready":
                    self.handle_ops_artifact_ready(
                        step_uid, step, handle, generation_id
                    )
        except StaleGenerationError as e:
            logger.debug(
                f"Discarding stale artifact event for {ledger_key}: {e}"
            )
        except SharedMemoryNotFoundError as e:
            logger.debug(
                f"Shared memory not found for {ledger_key}, event may be from "
                f"terminated worker: {e}"
            )
        except Exception as e:
            logger.error(
                f"Error handling task event '{event_name}': {e}",
                exc_info=True,
            )

    def on_task_complete(
        self,
        task: "Task",
        task_key: ArtifactKey,
        step: "Step",
        task_generation_id: int,
        context: Optional["GenerationContext"],
    ):
        """Callback for when a step assembly task finishes."""
        if context is not None:
            context.task_did_finish(task_key)

        step_uid = step.uid
        ledger_key = ArtifactKey.for_step(step_uid)

        retained = self._retained_handles_by_task.pop(task.id, [])
        for handle in retained:
            self._artifact_manager.release_handle(handle)

        task_status = task.get_status()
        logger.debug(f"[{ledger_key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            self._cleanup_step_render_handle(step_uid)
            with self._artifact_manager.report_cancellation(
                ledger_key, task_generation_id
            ):
                self.generation_finished.send(
                    self, step=step, generation_id=task_generation_id
                )
            return

        if task_status == "completed":
            try:
                task.result()
            except Exception as e:
                logger.error(f"Error on step assembly result: {e}")
                task_status = "failed"

        if task_status == "completed":
            with self._artifact_manager.report_completion(
                ledger_key, task_generation_id
            ):
                self.generation_finished.send(
                    self, step=step, generation_id=task_generation_id
                )
        else:
            self._cleanup_step_render_handle(step_uid)
            self._emit_node_state(ledger_key, NodeState.ERROR)
            with self._artifact_manager.report_failure(
                ledger_key, task_generation_id
            ):
                self.generation_finished.send(
                    self, step=step, generation_id=task_generation_id
                )

    def _cleanup_step_render_handle(self, step_uid: str):
        """Release the step render handle if one exists."""
        render_handle = self._artifact_manager.pop_step_render_handle(step_uid)
        if render_handle:
            self._artifact_manager.release_handle(render_handle)

    def launch_task(
        self,
        step: "Step",
        generation_id: int,
        context: Optional["GenerationContext"],
    ):
        """Starts the asynchronous task to assemble a step artifact."""
        if not self.validate_dependencies(step):
            logger.debug(
                f"Step assembly dependencies not met for step_uid={step.uid}"
            )
            return

        assembly_info, retained_handles = self.collect_assembly_info(
            step, generation_id
        )
        if not assembly_info:
            for handle in retained_handles:
                self._artifact_manager.release_handle(handle)
            logger.debug(f"No assembly info for step_uid={step.uid}")
            return

        if step.layer:
            for wp in step.layer.all_workpieces:
                handle = self._artifact_manager.get_workpiece_handle(
                    ArtifactKey.for_workpiece(wp.uid, step.uid),
                    generation_id,
                )
                if handle:
                    self.assembly_starting.send(
                        self,
                        step=step,
                        workpiece=wp,
                        handle=handle,
                    )

        ledger_key = ArtifactKey.for_step(step.uid)
        self._emit_node_state(ledger_key, NodeState.PROCESSING)

        from ..stage.step_runner import make_step_artifact_in_subprocess

        task_key = ArtifactKey.for_step(step.uid)

        if context is not None:
            context.add_task(task_key)

        assert self._machine is not None

        task = self._task_manager.run_process(
            make_step_artifact_in_subprocess,
            self._artifact_manager.get_store(),
            assembly_info,
            step.uid,
            generation_id,
            step.per_step_transformers_dicts,
            self._machine.max_cut_speed,
            self._machine.max_travel_speed,
            self._machine.acceleration,
            "step",
            key=task_key,
            when_done=lambda t: self.on_task_complete(
                t, task_key, step, generation_id, context
            ),
            when_event=lambda task, event_name, data: (
                self.handle_task_event(task, event_name, data, step)
            ),
        )
        self._retained_handles_by_task[task.id] = retained_handles

    def get_estimate(self, step_uid: StepKey) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        return self._time_cache.get(step_uid)

    def validate_dependencies(self, step: "Step") -> bool:
        """Validates that step assembly dependencies are met."""
        if not step.layer:
            return False
        if not self._machine:
            logger.warning(
                f"Cannot assemble step {step.uid}, no machine configured."
            )
            return False
        return True

    def validate_geometry_match(self, handle, workpiece: "WorkPiece") -> bool:
        """
        Validates that handle geometry matches current workpiece size.
        Returns True if geometry matches or handle is scalable.
        """
        if handle.is_scalable:
            return True
        return sizes_are_close(handle.generation_size, workpiece.size)

    def collect_assembly_info(
        self, step: "Step", generation_id: int
    ) -> Tuple[Optional[list], List["BaseArtifactHandle"]]:
        """
        Collects assembly info from all workpieces and retains handles.
        Returns (assembly_info, retained_handles).

        Checks both current and previous generation for handles to allow
        reuse of valid artifacts across generations.
        """
        assert step.layer is not None
        assembly_info = []
        retained_handles = []

        try:
            for wp in step.layer.all_workpieces:
                handle = self._artifact_manager.get_workpiece_handle(
                    ArtifactKey.for_workpiece(wp.uid, step.uid),
                    generation_id,
                )
                if handle is None and generation_id > 1:
                    handle = self._artifact_manager.get_workpiece_handle(
                        ArtifactKey.for_workpiece(wp.uid, step.uid),
                        generation_id - 1,
                    )
                if handle is None:
                    raise ValueError(
                        f"Missing handle for workpiece {wp.uid}, "
                        f"step {step.uid}"
                    )

                if not self.validate_geometry_match(handle, wp):
                    raise ValueError(f"Geometry mismatch for {wp.uid}")

                self._artifact_manager.retain_handle(handle)
                retained_handles.append(handle)

                info = {
                    "artifact_handle_dict": handle.to_dict(),
                    "world_transform_list": wp.get_world_transform().to_list(),
                    "workpiece_dict": wp.in_world().to_dict(),
                }
                assembly_info.append(info)
        except ValueError:
            for handle in retained_handles:
                self._artifact_manager.release_handle(handle)
            return None, []

        return assembly_info, retained_handles

    def shutdown(self):
        logger.debug("StepPipelineStage shutting down.")
        for handles in self._retained_handles_by_task.values():
            for handle in handles:
                self._artifact_manager.release_handle(handle)
        self._retained_handles_by_task.clear()

    def invalidate(self, key: StepKey):
        """Invalidates a step artifact, ensuring it will be regenerated."""
        self._cleanup_entry(key, full_invalidation=True)

    def _cleanup_entry(
        self,
        key: StepKey,
        full_invalidation: bool,
    ):
        """
        Removes a step artifact, clears time cache, and cancels its task.

        Args:
            key: The step key to clean up.
            full_invalidation: Whether to do a full invalidation.
        """
        logger.debug(f"StepPipelineStage: Cleaning up entry {key}.")
        self._time_cache.pop(key, None)

        if full_invalidation:
            render_handle = self._artifact_manager.pop_step_render_handle(key)
            if render_handle:
                logger.debug(
                    f"Popped and released stale render handle for step {key}."
                )
                self._artifact_manager.release_handle(render_handle)
            self._artifact_manager.invalidate_for_step(
                ArtifactKey.for_step(key)
            )
