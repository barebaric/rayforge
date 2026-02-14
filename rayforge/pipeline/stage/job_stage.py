from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Callable
from asyncio.exceptions import CancelledError
from blinker import Signal
from ...shared.tasker.task import Task
from ..artifact import JobArtifactHandle
from ..artifact.key import ArtifactKey
from ..artifact.manager import make_composite_key
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..context import GenerationContext
    from ..artifact import BaseArtifactHandle
    from ..artifact.manager import ArtifactManager
    from ..dag.scheduler import DagScheduler


logger = logging.getLogger(__name__)


class JobPipelineStage(PipelineStage):
    """
    Provides access to job artifacts and handles invalidation.
    Task launching, creation, and completion handling reside here.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: "Machine",
        scheduler: Optional["DagScheduler"] = None,
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine
        self._scheduler = scheduler
        self._retained_handles: List["BaseArtifactHandle"] = []
        self._job_running: bool = False
        self.job_generation_finished = Signal()
        self.job_generation_failed = Signal()

    @property
    def is_running(self) -> bool:
        """Check if a job generation is currently in progress."""
        return self._job_running

    def _adopt_artifact(
        self, data: dict, job_key: ArtifactKey
    ) -> "BaseArtifactHandle":
        """
        Adopt a job artifact from the subprocess.

        Args:
            data: The event data containing the handle dictionary.
            job_key: The ArtifactKey for this job.

        Returns:
            The adopted JobArtifactHandle.

        Raises:
            TypeError: If the handle is not a JobArtifactHandle.
        """
        handle_dict = data["handle_dict"]
        handle = self._artifact_manager.adopt_artifact(job_key, handle_dict)
        if not isinstance(handle, JobArtifactHandle):
            raise TypeError("Expected a JobArtifactHandle")
        return handle

    def handle_task_event(
        self,
        task: "Task",
        event_name: str,
        data: dict,
        job_key: ArtifactKey,
        generation_id: int,
    ):
        """Handle events broadcast from the job runner subprocess."""
        if event_name != "artifact_created":
            return

        received_gen_id = data.get("generation_id")
        if received_gen_id is None:
            logger.error(
                "Job event 'artifact_created' missing generation_id. Ignoring."
            )
            return

        job_key_dict = data.get("job_key")
        if job_key_dict is None:
            logger.error(
                "Job event 'artifact_created' missing job_key. Ignoring."
            )
            return

        received_job_key = ArtifactKey(
            id=job_key_dict["id"], group=job_key_dict["group"]
        )

        composite_key = make_composite_key(received_job_key, generation_id)
        entry = self._artifact_manager._get_ledger_entry(composite_key)
        if entry is not None and entry.generation_id != generation_id:
            logger.debug(
                f"Stale job event with generation_id {generation_id}, "
                f"current is {entry.generation_id}. Ignoring."
            )
            return

        try:
            handle = self._adopt_artifact(data, received_job_key)
            self._artifact_manager.cache_handle(
                received_job_key, handle, generation_id
            )
            if self._scheduler is not None:
                node = self._scheduler.graph.find_node(received_job_key)
                if node is not None:
                    node.state = NodeState.VALID
            logger.debug("Adopted job artifact")
        except Exception as e:
            logger.error(f"Error handling job artifact event: {e}")

    def on_task_complete(
        self,
        task: "Task",
        job_key: ArtifactKey,
        generation_id: int,
        on_done: Optional[Callable],
        context: Optional["GenerationContext"],
    ):
        """Callback for when a job generation task finishes."""
        if context is not None:
            context.task_did_finish(job_key)

        self.release_retained_handles()

        task_status = task.get_status()
        final_handle = None
        error = None

        if task_status == "completed":
            final_handle = self._artifact_manager.get_job_handle(
                job_key, generation_id
            )
            if final_handle:
                logger.info("Job generation successful.")
                if self._scheduler:
                    node = self._scheduler.graph.find_node(job_key)
                    if node is not None:
                        node.state = NodeState.VALID
            else:
                logger.info(
                    "Job generation finished with no artifact produced."
                )
            if on_done:
                on_done(final_handle, None)
            self.job_generation_finished.send(
                self, handle=final_handle, task_status=task_status
            )
        else:
            logger.error(f"Job generation failed with status: {task_status}")

            try:
                task.result()
            except CancelledError as e:
                error = e
                logger.info(f"Job generation was cancelled: {e}")
            except Exception as e:
                error = e

            if generation_id is not None and self._scheduler:
                node = self._scheduler.graph.find_node(job_key)
                if node is not None:
                    node.state = NodeState.ERROR

            self._artifact_manager.invalidate_for_job(job_key)
            if on_done:
                on_done(None, error)

            if task_status == "failed":
                self.job_generation_failed.send(
                    self, error=error, task_status=task_status
                )
            else:
                self.job_generation_finished.send(
                    self, handle=None, task_status=task_status
                )
        self._job_running = False

    def validate_dependencies(
        self, step_uids: List[str], generation_id: int
    ) -> bool:
        """Validate that all step dependencies are ready for job generation."""
        if not self._machine:
            logger.warning("Cannot generate job, no machine configured.")
            return False
        for step_uid in step_uids:
            step_key = ArtifactKey.for_step(step_uid)
            handle = self._artifact_manager.get_step_ops_handle(
                step_key, generation_id
            )
            if handle is None:
                logger.debug(f"Step {step_uid} not ready for job generation")
                return False
        return True

    def collect_step_handles(
        self, step_uids: List[str], generation_id: int
    ) -> Optional[Dict[str, Dict]]:
        """
        Collect step artifact handles for job generation.

        Returns a dict mapping step_uid -> handle_dict, or None if any
        step handle is missing. Also retains handles to prevent premature
        pruning while the job task is running.
        """
        step_handles = {}
        for step_uid in step_uids:
            step_key = ArtifactKey.for_step(step_uid)
            handle = self._artifact_manager.get_step_ops_handle(
                step_key, generation_id
            )
            if handle is None:
                for h in self._retained_handles:
                    self._artifact_manager.release_handle(h)
                self._retained_handles.clear()
                return None
            self._artifact_manager.retain_handle(handle)
            self._retained_handles.append(handle)
            step_handles[step_uid] = handle.to_dict()
        return step_handles

    def release_retained_handles(self) -> None:
        """Release all retained handles after job completion."""
        retained = self._retained_handles
        self._retained_handles = []
        for handle in retained:
            self._artifact_manager.release_handle(handle)

    def shutdown(self):
        logger.debug("JobPipelineStage shutting down.")

    def generate_job(
        self,
        step_uids: List[str],
        generation_id: int,
        context: Optional["GenerationContext"],
        doc: "Doc",
        on_done: Optional[Callable] = None,
        job_key: Optional[ArtifactKey] = None,
    ):
        """
        Start the asynchronous task to assemble and encode the final job.
        """
        if job_key is None:
            job_key = ArtifactKey.for_job()

        if not step_uids:
            logger.warning("Job generation called with no steps.")
            if on_done:
                on_done(None, None)
            self.job_generation_finished.send(
                self, handle=None, task_status="completed"
            )
            return

        if not self.validate_dependencies(step_uids, generation_id):
            logger.warning("Job dependencies not ready.")
            if on_done:
                on_done(
                    None,
                    RuntimeError(
                        "Job dependencies are not ready. "
                        "Please wait and try again."
                    ),
                )
            return

        step_handles = self.collect_step_handles(step_uids, generation_id)
        if step_handles is None:
            logger.error("Failed to collect step handles for job generation.")
            if on_done:
                on_done(None, RuntimeError("Failed to collect step handles."))
            return

        if self._scheduler:
            node = self._scheduler.graph.find_node(job_key)
            if node is not None:
                node.state = NodeState.PROCESSING

        logger.info(f"Starting job generation with {len(step_handles)} steps.")

        assert self._machine is not None

        job_desc_dict = {
            "step_artifact_handles_by_uid": step_handles,
            "machine_dict": self._machine.to_dict(),
            "doc_dict": doc.to_dict(),
        }

        self._launch_task(
            job_desc_dict, job_key, on_done, generation_id, context
        )

    def _launch_task(
        self,
        job_desc_dict: Dict,
        job_key: ArtifactKey,
        on_done: Optional[Callable],
        generation_id: int,
        context: Optional["GenerationContext"],
    ):
        """
        Launch the subprocess task for job generation.
        """
        from ..stage.job_runner import make_job_artifact_in_subprocess

        self._job_running = True

        if context is not None:
            context.add_task(job_key)

        self._task_manager.run_process(
            make_job_artifact_in_subprocess,
            self._artifact_manager._store,
            job_description_dict=job_desc_dict,
            creator_tag="job",
            generation_id=generation_id,
            job_key=job_key,
            key=job_key,
            when_done=lambda t: self.on_task_complete(
                t, job_key, generation_id, on_done, context
            ),
            when_event=lambda task, event_name, data: (
                self.handle_task_event(
                    task, event_name, data, job_key, generation_id
                )
            ),
        )
