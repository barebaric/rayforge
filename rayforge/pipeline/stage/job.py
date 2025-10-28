from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional
from blinker import Signal

from .base import PipelineStage
from ..artifact import JobArtifactHandle, create_handle_from_dict
from .job_runner import JobDescription
from ...context import get_context

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.task import Task
    from ..artifact.cache import ArtifactCache
    from ...shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)

# The constant key for the single, final job artifact in the cache
JobKey = "final_job"


class JobGeneratorStage(PipelineStage):
    """A pipeline stage that assembles the final job artifact."""

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._active_task: Optional["Task"] = None
        self.generation_finished = Signal()

    @property
    def is_busy(self) -> bool:
        return self._active_task is not None

    def reconcile(self, doc: "Doc"):
        """
        Job generation is triggered on-demand, so reconcile does nothing.
        """
        pass

    def shutdown(self):
        """Cancels the active job generation task."""
        logger.debug("JobGeneratorStage shutting down.")
        if self._active_task:
            self._task_manager.cancel_task(self._active_task.key)
            self._active_task = None

    def generate_job(self, doc: "Doc"):
        """
        Starts the asynchronous task to assemble and encode the final job.
        """
        if self.is_busy:
            logger.warning("Job generation is already in progress.")
            return

        machine = get_context().machine
        if not machine:
            logger.error("Cannot generate job: No machine is configured.")
            return

        step_handles = {
            step.uid: handle.to_dict()
            for layer in doc.layers
            if layer.workflow
            for step in layer.workflow.steps
            if (handle := self._artifact_cache.get_step_ops_handle(step.uid))
        }

        if not step_handles:
            logger.warning("No step artifacts to assemble for the job.")
            self.generation_finished.send(self, handle=None)
            return

        logger.info(f"Starting job generation with {len(step_handles)} steps.")
        self._artifact_cache.invalidate_for_job()

        job_desc = JobDescription(
            step_artifact_handles_by_uid=step_handles,
            machine_dict=machine.to_dict(),
            doc_dict=doc.to_dict(),
        )

        from .job_runner import make_job_artifact_in_subprocess

        def when_done_callback(task: "Task"):
            self._on_job_assembly_complete(task)

        task = self._task_manager.run_process(
            make_job_artifact_in_subprocess,
            job_description_dict=job_desc.__dict__,
            key=JobKey,
            when_done=when_done_callback,
            when_event=self._on_job_task_event,
        )
        self._active_task = task

    def _on_job_task_event(self, task: "Task", event_name: str, data: dict):
        """Handles events broadcast from the job runner subprocess."""
        if event_name == "artifact_created":
            try:
                handle_dict = data["handle_dict"]
                handle = create_handle_from_dict(handle_dict)
                if not isinstance(handle, JobArtifactHandle):
                    raise TypeError("Expected a JobArtifactHandle")

                get_context().artifact_store.adopt(handle)
                self._artifact_cache.put_job_handle(handle)
            except Exception as e:
                logger.error(f"Error handling job artifact event: {e}")

    def _on_job_assembly_complete(self, task: "Task"):
        """Callback for when the final job assembly task finishes."""
        self._active_task = None
        final_handle = self._artifact_cache.get_job_handle()

        if task.get_status() == "completed":
            if final_handle:
                logger.info("Job generation successful.")
            else:
                logger.info(
                    "Job generation finished with no artifact produced."
                )
        else:
            logger.error(
                f"Job generation failed with status: {task.get_status()}"
            )
            # The artifact put in the cache by the event handler is now
            # invalid. Invalidate it to ensure it's released.
            self._artifact_cache.invalidate_for_job()
            final_handle = None

        self.generation_finished.send(self, handle=final_handle)
