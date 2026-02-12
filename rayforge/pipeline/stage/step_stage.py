from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Optional
from blinker import Signal
from ..artifact.key import ArtifactKey
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.step import Step
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
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

        self.generation_finished = Signal()
        self.render_artifact_ready = Signal()
        self.time_estimate_ready = Signal()

        self._scheduler.step_render_artifact_ready.connect(
            self._on_scheduler_render_ready
        )
        self._scheduler.step_time_estimate_ready.connect(
            self._on_scheduler_time_ready
        )
        self._scheduler.step_generation_finished.connect(
            self._on_scheduler_generation_finished
        )

    def _on_scheduler_render_ready(self, scheduler, step: "Step"):
        """Forward render artifact ready signal from scheduler."""
        self.render_artifact_ready.send(self, step=step)

    def _on_scheduler_time_ready(self, scheduler, step: "Step", time: float):
        """Forward time estimate ready signal from scheduler."""
        self._time_cache[step.uid] = time
        self.time_estimate_ready.send(self, step=step, time=time)

    def _on_scheduler_generation_finished(
        self, scheduler, step: "Step", generation_id: int
    ):
        """Forward generation finished signal from scheduler."""
        self.generation_finished.send(
            self, step=step, generation_id=generation_id
        )

    def get_estimate(self, step_uid: StepKey) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        return self._time_cache.get(step_uid)

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
