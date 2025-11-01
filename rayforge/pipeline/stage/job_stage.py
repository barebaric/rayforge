from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Callable, List
from blinker import Signal

from .base import PipelineStage
from ..artifact import (
    JobArtifactHandle,
    create_handle_from_dict,
    StepOpsArtifactHandle,
)
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


class JobPipelineStage(PipelineStage):
    """A pipeline stage that assembles the final job artifact."""

    def __init__(
        self, task_manager: "TaskManager", artifact_cache: "ArtifactCache"
    ):
        super().__init__(task_manager, artifact_cache)
        self._active_task: Optional["Task"] = None
        self.generation_finished = Signal()
        self.generation_failed = Signal()

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
        logger.debug("JobPipelineStage shutting down.")
        if self._active_task:
            self._task_manager.cancel_task(self._active_task.key)
            self._active_task = None

    def generate_job(self, doc: "Doc", on_done: Optional[Callable] = None):
        """
        Starts the asynchronous task to assemble and encode the final job.

        Args:
            doc: The document to generate the job from.
            on_done: An optional one-shot callback to execute upon completion.
        """
        if self.is_busy:
            logger.warning("Job generation is already in progress.")
            # If a callback was provided, immediately call it with an error
            if on_done:
                on_done(
                    None,
                    RuntimeError("Job generation is already in progress."),
                )
            return

        machine = get_context().machine
        if not machine:
            logger.error("Cannot generate job: No machine is configured.")
            if on_done:
                on_done(None, RuntimeError("No machine is configured."))
            return

        acquired_handles: List[StepOpsArtifactHandle] = []
        step_handles_dict = {}
        try:
            # Collect and acquire handles to protect them from invalidation
            step_handles_to_acquire: List[StepOpsArtifactHandle] = []
            for layer in doc.layers:
                if layer.workflow:
                    for step in layer.workflow.steps:
                        handle = self._artifact_cache.get_step_ops_handle(
                            step.uid
                        )
                        if isinstance(handle, StepOpsArtifactHandle):
                            step_handles_to_acquire.append(handle)

            # Atomically acquire all handles. If any fail, release all that
            # were successfully acquired and abort.
            for handle in step_handles_to_acquire:
                if get_context().artifact_store.acquire(handle):
                    acquired_handles.append(handle)
                else:
                    # An artifact was invalidated during collection. Abort.
                    logger.warning(
                        f"Failed to acquire artifact {handle.shm_name}, "
                        f"aborting job gen."
                    )
                    raise RuntimeError("Dependency artifact became invalid.")

            for handle in acquired_handles:
                # We need to find the step UID associated with this handle
                for layer in doc.layers:
                    if layer.workflow:
                        for step in layer.workflow.steps:
                            cached_handle = (
                                self._artifact_cache.get_step_ops_handle(
                                    step.uid
                                )
                            )
                            if cached_handle == handle:
                                step_handles_dict[step.uid] = handle.to_dict()
                                break

            if not acquired_handles:
                logger.warning("No step artifacts to assemble for the job.")
                self.generation_finished.send(
                    self, handle=None, task_status="completed"
                )
                if on_done:
                    on_done(None, None)
                return

            logger.info(
                f"Starting job generation with {len(step_handles_dict)} steps."
            )
            self._artifact_cache.invalidate_for_job()

            job_desc = JobDescription(
                step_artifact_handles_by_uid=step_handles_dict,
                machine_dict=machine.to_dict(),
                doc_dict=doc.to_dict(),
            )

            from .job_runner import make_job_artifact_in_subprocess

            def when_done_callback(task: "Task"):
                """
                This closure captures the `on_done` callback and the list
                of acquired handles for release.
                """
                self._active_task = None
                task_status = task.get_status()
                final_handle = None
                error = None

                try:
                    if task_status == "completed":
                        final_handle = self._artifact_cache.get_job_handle()
                        if final_handle:
                            logger.info("Job generation successful.")
                        else:
                            logger.info(
                                "Job generation finished with "
                                "no artifact produced."
                            )
                        if on_done:
                            on_done(final_handle, None)
                        self.generation_finished.send(
                            self,
                            handle=final_handle,
                            task_status=task_status,
                        )
                    else:
                        logger.error(
                            f"Job generation failed with status: {task_status}"
                        )
                        self._artifact_cache.invalidate_for_job()
                        try:
                            task.result()
                        except Exception as e:
                            error = e

                        if on_done:
                            on_done(None, error)

                        if task_status == "failed":
                            self.generation_failed.send(
                                self, error=error, task_status=task_status
                            )
                        else:
                            self.generation_finished.send(
                                self, handle=None, task_status=task_status
                            )
                finally:
                    # Release handles regardless of outcome.
                    for h in acquired_handles:
                        get_context().artifact_store.release(h)

            task = self._task_manager.run_process(
                make_job_artifact_in_subprocess,
                job_description_dict=job_desc.__dict__,
                creator_tag="job",
                key=JobKey,
                when_done=when_done_callback,
                when_event=self._on_job_task_event,
            )
            self._active_task = task

        except Exception as e:
            logger.error(f"Failed to create job generation task: {e}")
            for h in acquired_handles:
                get_context().artifact_store.release(h)
            if on_done:
                on_done(None, e)

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
