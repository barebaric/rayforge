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
        validation_error = self._validate_job_generation_state(on_done)
        if validation_error:
            return

        machine = self._machine
        machine.hydrate()

        step_handles = self._collect_step_handles(doc)
        logger.info(f"Starting job generation with {len(step_handles)} steps.")

        self._artifact_manager.invalidate_for_job()

        self._adoption_event = self._create_adoption_event()

        job_desc = self._create_job_description(step_handles, machine, doc)

        when_done_callback = self._create_when_done_callback(on_done)

        task = self._launch_job_task(job_desc, when_done_callback)
        self._active_task = task

    def _validate_job_generation_state(
        self, on_done: Optional[Callable]
    ) -> Optional[RuntimeError]:
        """
        Validates the state before job generation.

        Returns:
            An error if validation fails, None otherwise.
        """
        if self.is_busy:
            logger.warning("Job generation is already in progress.")
            if on_done:
                on_done(
                    None,
                    RuntimeError("Job generation is already in progress."),
                )
            return RuntimeError("Job generation is already in progress.")

        if not self._machine:
            logger.error("Cannot generate job: No machine is configured.")
            if on_done:
                on_done(None, RuntimeError("No machine is configured."))
            return RuntimeError("No machine is configured.")

        return None

    def _collect_step_handles(self, doc: "Doc") -> dict:
        """
        Collects step artifact handles from document layers.

        Returns:
            A dictionary mapping step UIDs to handle dictionaries.
        """
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
                    stack.enter_context(
                        self._artifact_manager.checkout(step.uid)
                    )
        return step_handles

    def _create_adoption_event(self) -> "threading.Event":
        """
        Creates an adoption event for the handshake protocol.

        Returns:
            A multiprocessing Event for artifact adoption handshake.
        """
        manager = mp.Manager()
        return manager.Event()

    def _create_job_description(
        self,
        step_handles: dict,
        machine: "Machine",
        doc: "Doc",
    ) -> JobDescription:
        """
        Creates a job description from the provided components.

        Returns:
            A JobDescription object.
        """
        return JobDescription(
            step_artifact_handles_by_uid=step_handles,
            machine_dict=machine.to_dict(),
            doc_dict=doc.to_dict(),
        )

    def _create_when_done_callback(
        self, on_done: Optional[Callable]
    ) -> Callable[["Task"], None]:
        """
        Creates a callback function to handle task completion.

        Args:
            on_done: An optional one-shot callback to execute upon completion.

        Returns:
            A callback function that handles task completion.
        """

        def when_done_callback(task: "Task"):
            """
            This nested function is a closure. It captures the `on_done`
            callback specific to this `generate_job` call.
            """
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
                        handle=None, task_status=task_status
                    )

        return when_done_callback

    def _launch_job_task(
        self, job_desc: JobDescription, when_done_callback: Callable
    ) -> "Task":
        """
        Launches the subprocess task for job generation.

        Args:
            job_desc: The job description to use.
            when_done_callback: The callback to execute on completion.

        Returns:
            The created task.
        """
        from .job_runner import make_job_artifact_in_subprocess

        return self._task_manager.run_process(
            make_job_artifact_in_subprocess,
            self._artifact_manager._store,
            job_description_dict=job_desc.__dict__,
            creator_tag="job",
            key=JobKey,
            when_done=when_done_callback,
            when_event=self._on_job_task_event,
            adoption_event=self._adoption_event,
        )

    def _on_job_task_event(self, task: "Task", event_name: str, data: dict):
        """Handles events broadcast from the job runner subprocess."""
        if event_name == "artifact_created":
            self._handle_artifact_created(task, data)

    def _is_task_active(self, task: "Task") -> bool:
        """
        Checks if the given task is the currently active task.

        Args:
            task: The task to check.

        Returns:
            True if the task is active, False otherwise.
        """
        return self._active_task is task

    def _handle_artifact_created(self, task: "Task", data: dict):
        """
        Handles the artifact_created event from the job runner subprocess.

        Args:
            task: The task that generated the event.
            data: The event data containing the handle dictionary.
        """
        if not self._is_task_active(task):
            logger.debug("Ignoring artifact_created event from inactive task")
            self._complete_adoption_handshake()
            return

        try:
            handle = self._adopt_job_artifact(data)
            self._artifact_manager.put_job_handle(handle)
            logger.debug("Adoption handshake completed for job artifact")
        except Exception as e:
            logger.error(f"Error handling job artifact event: {e}")
        finally:
            self._complete_adoption_handshake()

    def _adopt_job_artifact(self, data: dict) -> JobArtifactHandle:
        """
        Adopts a job artifact from the subprocess.

        Args:
            data: The event data containing the handle dictionary.

        Returns:
            The adopted JobArtifactHandle.

        Raises:
            TypeError: If the handle is not a JobArtifactHandle.
        """
        handle_dict = data["handle_dict"]
        handle = self._artifact_manager.adopt_artifact(JobKey, handle_dict)
        if not isinstance(handle, JobArtifactHandle):
            raise TypeError("Expected a JobArtifactHandle")
        return handle

    def _complete_adoption_handshake(self):
        """
        Completes the adoption handshake by setting the adoption event.
        This unblocks the worker process.
        """
        if self._adoption_event is not None:
            self._adoption_event.set()
