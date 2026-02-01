from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Callable
from blinker import Signal
import multiprocessing as mp
from asyncio.exceptions import CancelledError
from ..artifact import JobArtifactHandle
from .base import PipelineStage
from .job_runner import JobDescription
from contextlib import ExitStack

if TYPE_CHECKING:
    import threading
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)

# The constant key for the single, final job artifact in cache
JobKey = "final_job"


class JobPipelineStage(PipelineStage):
    """A pipeline stage that assembles the final job artifact."""

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: "Machine",
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._active_task: Optional["Task"] = None
        self._adoption_event: Optional["threading.Event"] = None
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
        """Cancels active job generation task."""
        logger.debug("JobPipelineStage shutting down.")
        if self._active_task:
            self._task_manager.cancel_task(self._active_task.key)
            self._active_task = None
        self._adoption_event = None

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

        machine = self._machine
        if not machine:
            logger.error("Cannot generate job: No machine is configured.")
            # Fire callback with an error if provided
            if on_done:
                on_done(None, RuntimeError("No machine is configured."))
            return

        # Hydrate the machine to capture the current dialect state
        machine.hydrate()

        step_handles = {}
        with ExitStack() as stack:
            for layer in doc.layers:
                if not layer.workflow:
                    continue
                for step in layer.workflow.steps:
                    if not step.visible:
                        continue
                    handle = self._artifact_manager.get_step_ops_handle(
                        step.uid
                    )
                    if handle is None:
                        continue
                    step_handles[step.uid] = handle.to_dict()
                    # Checkout the handle so it won't be released while job
                    # runs
                    stack.enter_context(
                        self._artifact_manager.checkout(step.uid)
                    )

        # Allow job generation to continue even with no steps. The runner will
        # produce a job with only a preamble and postscript.
        logger.info(f"Starting job generation with {len(step_handles)} steps.")

        self._artifact_manager.invalidate_for_job()

        # Create an adoption event for the handshake protocol
        manager = mp.Manager()
        self._adoption_event = manager.Event()

        job_desc = JobDescription(
            step_artifact_handles_by_uid=step_handles,
            machine_dict=machine.to_dict(),
            doc_dict=doc.to_dict(),
        )

        from .job_runner import make_job_artifact_in_subprocess

        def when_done_callback(task: "Task"):
            """
            This nested function is a closure. It captures the `on_done`
            callback specific to this `generate_job` call.
            """
            # This is now the ONLY place self._active_task is reset to None
            self._active_task = None

            task_status = task.get_status()
            final_handle = None
            error = None

            if task_status == "completed":
                final_handle = self._artifact_manager.get_job_handle()
                if final_handle:
                    logger.info("Job generation successful.")
                else:
                    logger.info(
                        "Job generation finished with no artifact produced."
                    )
                if on_done:
                    on_done(final_handle, None)
                self.generation_finished.send(
                    self, handle=final_handle, task_status=task_status
                )
            else:
                logger.error(
                    f"Job generation failed with status: {task_status}"
                )
                self._artifact_manager.invalidate_for_job()
                try:
                    task.result()
                except CancelledError as e:
                    error = e
                    logger.info(f"Job generation was cancelled: {e}")
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

        # We no longer need _on_job_assembly_complete
        task = self._task_manager.run_process(
            make_job_artifact_in_subprocess,
            self._artifact_manager._store,
            job_description_dict=job_desc.__dict__,
            creator_tag="job",
            key=JobKey,
            when_done=when_done_callback,
            when_event=self._on_job_task_event,
            adoption_event=self._adoption_event,
        )
        self._active_task = task

    def _on_job_task_event(self, task: "Task", event_name: str, data: dict):
        """Handles events broadcast from the job runner subprocess."""
        if event_name == "artifact_created":
            # Ignore artifact events from tasks that are no longer active
            # (e.g., cancelled tasks). This prevents stale artifacts from
            # being added to the cache after cancellation.
            if self._active_task is not task:
                logger.debug(
                    "Ignoring artifact_created event from inactive task"
                )
                # Still set the adoption event to unblock the worker
                if self._adoption_event is not None:
                    self._adoption_event.set()
                return

            try:
                handle_dict = data["handle_dict"]
                handle = self._artifact_manager.adopt_artifact(
                    JobKey, handle_dict
                )
                if not isinstance(handle, JobArtifactHandle):
                    raise TypeError("Expected a JobArtifactHandle")

                self._artifact_manager.put_job_handle(handle)

                # Signal the worker that we've adopted the artifact
                if self._adoption_event is not None:
                    self._adoption_event.set()
                    logger.debug(
                        "Adoption handshake completed for job artifact"
                    )
            except Exception as e:
                logger.error(f"Error handling job artifact event: {e}")
                # Still set the event to unblock the worker
                if self._adoption_event is not None:
                    self._adoption_event.set()
