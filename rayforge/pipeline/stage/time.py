from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Tuple, Optional
from blinker import Signal
from .base import PipelineStage
from ... import config

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...core.step import Step
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache
    from ...shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)

# Key: (step_uid, workpiece_uid, width, height)
# Value: float (time in seconds) or None for pending
TimeCacheKey = Tuple[str, str, float, float]
TimeCacheType = Dict[TimeCacheKey, Optional[float]]


class TimeEstimatorStage(PipelineStage):
    """A pipeline stage that calculates time estimates for artifacts."""

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._time_cache: TimeCacheType = {}
        self._generation_id_map: Dict[TimeCacheKey, int] = {}
        self._active_tasks: Dict[TimeCacheKey, "Task"] = {}
        self.estimation_updated = Signal()
        logger.debug("TimeEstimatorStage initialized.")

    @property
    def is_busy(self) -> bool:
        """Returns True if the stage has any active tasks."""
        return bool(self._active_tasks)

    def reconcile(self, doc: "Doc"):
        """Removes time estimates for objects no longer in the document."""
        logger.debug("TimeEstimatorStage reconciling...")
        all_current_pairs = {
            (step.uid, workpiece.uid)
            for layer in doc.layers
            if layer.workflow is not None
            for step in layer.workflow.steps
            for workpiece in layer.all_workpieces
        }
        obsolete_keys = [
            k
            for k in self._time_cache
            if (k[0], k[1]) not in all_current_pairs
        ]
        if obsolete_keys:
            logger.debug(f"Found {len(obsolete_keys)} obsolete time entries.")
        for key in obsolete_keys:
            self._cleanup_entry(key)

    def shutdown(self):
        """Cancels all active time estimation tasks."""
        logger.debug("TimeEstimatorStage shutting down.")
        for key in list(self._active_tasks.keys()):
            self._cleanup_task(key)
        self._active_tasks.clear()

    def get_estimate(
        self, step: "Step", workpiece: "WorkPiece"
    ) -> Optional[float]:
        """Retrieves a cached time estimate if available."""
        key = (step.uid, workpiece.uid, workpiece.size[0], workpiece.size[1])
        return self._time_cache.get(key)

    def generate_estimate(self, step: "Step", workpiece: "WorkPiece"):
        """
        Starts an asynchronous task to estimate machining time if needed.
        """
        if any(s <= 0 for s in workpiece.size) or not config.config.machine:
            return

        key = (step.uid, workpiece.uid, workpiece.size[0], workpiece.size[1])
        if key in self._time_cache and self._time_cache[key] is not None:
            logger.debug(f"Time estimate for {key} already cached.")
            return  # Already have a valid estimate for this exact state

        handle = self._artifact_cache.get_workpiece_handle(
            step.uid, workpiece.uid
        )
        if handle is None:
            logger.debug(
                f"Cannot estimate time for {key}, no artifact handle."
            )
            return

        if key in self._active_tasks:
            logger.debug(f"Cancelling existing time task for key {key}.")
            self._cleanup_task(key)

        generation_id = self._generation_id_map.get(key, 0) + 1
        self._generation_id_map[key] = generation_id

        self._time_cache[key] = None  # Mark as pending
        logger.info(
            f"Triggering time estimation for {key} (gen {generation_id})"
        )

        def when_done_callback(task: "Task"):
            self._on_estimation_complete(task, key, generation_id)

        from ..timerunner import run_time_estimation_in_subprocess

        machine = config.config.machine
        task = self._task_manager.run_process(
            run_time_estimation_in_subprocess,
            handle.to_dict(),
            workpiece.size,
            step.per_step_transformers_dicts,
            machine.max_cut_speed,
            machine.max_travel_speed,
            machine.acceleration,
            generation_id,
            key=key,
            when_done=when_done_callback,
        )
        self._active_tasks[key] = task

    def invalidate_for_step(self, step_uid: str):
        """Removes all time cache entries associated with a given step."""
        logger.debug(f"Invalidating time cache for step '{step_uid}'.")
        keys_to_clean = [k for k in self._time_cache if k[0] == step_uid]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def invalidate_for_workpiece(self, workpiece_uid: str):
        """
        Removes all time cache entries associated with a given workpiece.
        """
        logger.debug(
            f"Invalidating time cache for workpiece '{workpiece_uid}'."
        )
        keys_to_clean = [k for k in self._time_cache if k[1] == workpiece_uid]
        for key in keys_to_clean:
            self._cleanup_entry(key)

    def _on_estimation_complete(
        self, task: "Task", key: TimeCacheKey, task_generation_id: int
    ):
        """Callback for when a time estimation task finishes."""
        self._active_tasks.pop(key, None)

        current_gen_id = self._generation_id_map.get(key)
        if current_gen_id != task_generation_id:
            logger.debug(f"Ignoring stale time estimation callback for {key}.")
            return

        if task.get_status() == "completed":
            try:
                result_time, result_gen_id = task.result()
                if self._generation_id_map.get(key) == result_gen_id:
                    logger.info(
                        f"Time estimation for {key} complete: {result_time}s"
                    )
                    self._time_cache[key] = result_time
                else:
                    logger.warning(
                        f"Ignoring stale time estimation result for {key}."
                    )
            except Exception as e:
                logger.error(
                    f"Error getting time estimation result for {key}: {e}",
                    exc_info=True,
                )
                self._time_cache[key] = -1.0  # Mark as errored
        else:
            logger.warning(
                f"Time estimation for {key} failed. "
                f"Status: {task.get_status()}"
            )
            self._time_cache[key] = -1.0  # Mark as errored
        self.estimation_updated.send(self)

    def _cleanup_task(self, key: TimeCacheKey):
        """Cancels a task if it's active."""
        if key in self._active_tasks:
            task = self._active_tasks.pop(key, None)
            if task:
                logger.debug(f"Cancelling active time task for {key}")
                self._task_manager.cancel_task(task.key)

    def _cleanup_entry(self, key: TimeCacheKey):
        """Removes a time cache entry and cancels its associated task."""
        logger.debug(f"TimeEstimatorStage: Cleaning up time key {key}.")
        self._time_cache.pop(key, None)
        self._generation_id_map.pop(key, None)
        self._cleanup_task(key)
