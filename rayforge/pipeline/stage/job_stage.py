from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Optional, Callable
from blinker import Signal
from asyncio.exceptions import CancelledError
from ..artifact import JobArtifactHandle
from ..artifact.manager import ArtifactManager
from .base import PipelineStage
from .job_runner import JobDescription

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ...shared.tasker.task import Task


logger = logging.getLogger(__name__)


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

        self.generation_finished = Signal()
        self.generation_failed = Signal()

    def reconcile(self, doc: "Doc"):
        """
        Job generation is triggered on-demand, so reconcile does nothing.
        """
        pass

    def shutdown(self):
        """Cancels active job generation task."""
        logger.debug("JobPipelineStage shutting down.")

    def generate_job(
        self,
        job_description: JobDescription,
        on_done: Optional[Callable] = None,
    ):
        """
        Starts the asynchronous task to assemble and encode the final job.

        This method receives a fully prepared JobDescription and is
        responsible for launching the subprocess task and handling its
        completion.

        Args:
            job_description: A complete description of the job to generate.
            on_done: An optional one-shot callback to execute upon completion.
        """
        generation_id = int(time.time() * 1000)
        self._artifact_manager.mark_pending(
            self._artifact_manager.JOB_KEY, generation_id
        )

        logger.info(
            "Starting job generation with "
            f"{len(job_description.step_artifact_handles_by_uid)} steps."
        )

        when_done_callback = self._create_when_done_callback(
            on_done, generation_id
        )

        self._launch_job_task(
            job_description, when_done_callback, generation_id
        )

    def _create_when_done_callback(
        self, on_done: Optional[Callable], generation_id: Optional[int]
    ) -> Callable[["Task"], None]:
        """
        Creates a callback function to handle task completion.

        Args:
            on_done: An optional one-shot callback to execute upon completion.
            generation_id: The generation ID for this job generation.

        Returns:
            A callback function that handles task completion.
        """

        def when_done_callback(task: "Task"):
            """
            This nested function is a closure. It captures the `on_done`
            callback specific to this `generate_job` call.
            """
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
                try:
                    task.result()
                except CancelledError as e:
                    error = e
                    logger.info(f"Job generation was cancelled: {e}")
                except Exception as e:
                    error = e

                if generation_id is not None:
                    error_msg = str(error) if error else "Unknown error"
                    self._artifact_manager.mark_error(
                        self._artifact_manager.JOB_KEY,
                        error_msg,
                        generation_id,
                    )

                self._artifact_manager.invalidate_for_job()

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
        self,
        job_desc: JobDescription,
        when_done_callback: Callable,
        generation_id: int,
    ) -> "Task":
        """
        Launches the subprocess task for job generation.

        Args:
            job_desc: The job description to use.
            when_done_callback: The callback to execute on completion.
            generation_id: The generation ID for this job generation.

        Returns:
            The created task.
        """
        from .job_runner import make_job_artifact_in_subprocess

        return self._task_manager.run_process(
            make_job_artifact_in_subprocess,
            self._artifact_manager._store,
            job_description_dict=job_desc.__dict__,
            creator_tag="job",
            generation_id=generation_id,
            key=self._artifact_manager.JOB_KEY,
            when_done=when_done_callback,
            when_event=self._on_job_task_event,
        )

    def _on_job_task_event(self, task: "Task", event_name: str, data: dict):
        """Handles events broadcast from the job runner subprocess."""
        if event_name == "artifact_created":
            self._handle_artifact_created(data)

    def _handle_artifact_created(self, data: dict):
        """
        Handles the artifact_created event from the job runner subprocess.

        Args:
            data: The event data containing the handle dictionary and
                generation_id.
        """
        try:
            handle = self._adopt_job_artifact(data)
            generation_id = data.get("generation_id")
            if generation_id is not None:
                self._artifact_manager.commit(
                    self._artifact_manager.JOB_KEY, handle, generation_id
                )
            logger.debug("Adopted job artifact")
        except Exception as e:
            logger.error(f"Error handling job artifact event: {e}")

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
        handle = self._artifact_manager.adopt_artifact(
            self._artifact_manager.JOB_KEY, handle_dict
        )
        if not isinstance(handle, JobArtifactHandle):
            raise TypeError("Expected a JobArtifactHandle")
        return handle
