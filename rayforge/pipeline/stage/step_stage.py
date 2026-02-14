from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from blinker import Signal
from ...shared.util.size import sizes_are_close
from ..artifact import StepOpsArtifactHandle, StepRenderArtifactHandle
from ..artifact.key import ArtifactKey
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...machine.models.machine import Machine
    from ...shared.tasker.task import Task
    from ...shared.tasker.manager import TaskManager
    from ..artifact import BaseArtifactHandle
    from ..artifact.manager import ArtifactManager
    from ..dag.scheduler import DagScheduler


logger = logging.getLogger(__name__)

StepKey = str


class StepPipelineStage(PipelineStage):
    """
    Provides access to step artifacts and handles invalidation.
    Task launching is delegated to the DagScheduler.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
        scheduler: "DagScheduler",
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._scheduler = scheduler
        self._time_cache: Dict[StepKey, Optional[float]] = {}
        self._retained_handles: Dict[str, List["BaseArtifactHandle"]] = {}

        self.generation_finished = Signal()
        self.render_artifact_ready = Signal()
        self.time_estimate_ready = Signal()

    def handle_render_artifact_ready(
        self, step_uid: str, step: "Step", handle_dict: dict
    ):
        """Handles the render artifact ready event."""
        handle = self._artifact_manager.adopt_artifact(
            ArtifactKey.for_step(step_uid), handle_dict
        )
        if not isinstance(handle, StepRenderArtifactHandle):
            raise TypeError("Expected a StepRenderArtifactHandle")

        self._artifact_manager.put_step_render_handle(step_uid, handle)
        self.render_artifact_ready.send(self, step=step)

    def handle_ops_artifact_ready(
        self, step_uid: str, handle_dict: dict, generation_id: int
    ):
        """Handles the ops artifact ready event."""
        handle = self._artifact_manager.adopt_artifact(
            ArtifactKey.for_step(step_uid), handle_dict
        )
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
            logger.error(
                f"[{step_uid}] Task event '{event_name}' missing "
                f"generation_id. Ignoring."
            )
            return

        if not self._artifact_manager.is_generation_current(
            ledger_key, generation_id
        ):
            logger.debug(
                f"[{step_uid}] Stale event '{event_name}' with "
                f"generation_id {generation_id}. Ignoring."
            )
            return

        try:
            if event_name == "render_artifact_ready":
                self.handle_render_artifact_ready(
                    step_uid, step, data["handle_dict"]
                )

            elif event_name == "ops_artifact_ready":
                self.handle_ops_artifact_ready(
                    step_uid, data["handle_dict"], generation_id
                )

            elif event_name == "time_estimate_ready":
                self.handle_time_estimate_ready(
                    step_uid, step, data["time_estimate"]
                )
        except Exception as e:
            logger.error(f"Error handling task event '{event_name}': {e}")

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
        base_key = ArtifactKey.for_step(step.uid)
        node = self._scheduler.graph.find_node(base_key)
        if node is not None and node.state == NodeState.PROCESSING:
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

    def store_retained_handles(
        self, step_uid: str, handles: List["BaseArtifactHandle"]
    ) -> None:
        """Store retained handles for a step."""
        self._retained_handles[step_uid] = handles

    def release_retained_handles(self, step_uid: str) -> None:
        """Release retained handles for a step."""
        retained = self._retained_handles.pop(step_uid, [])
        for handle in retained:
            self._artifact_manager.release_handle(handle)

    def shutdown(self):
        logger.debug("StepPipelineStage shutting down.")

    def invalidate(self, key: StepKey):
        """Invalidates a step artifact, ensuring it will be regenerated."""
        self._cleanup_entry(key, full_invalidation=True)

    def mark_stale_and_trigger(self, step: "Step", generation_id: int):
        """Marks a step as stale and immediately tries to trigger assembly."""
        self._cleanup_entry(step.uid, full_invalidation=False)
        self._scheduler.mark_node_dirty(ArtifactKey.for_step(step.uid))
        self._scheduler.process_graph()

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
