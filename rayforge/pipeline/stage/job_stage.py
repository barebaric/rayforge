from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, List, Optional
from ..artifact import JobArtifactHandle
from ..artifact.key import ArtifactKey
from ..artifact.manager import make_composite_key
from ..dag.node import NodeState
from .base import PipelineStage

if TYPE_CHECKING:
    from ...machine.models.machine import Machine
    from ...shared.tasker.task import Task
    from ...shared.tasker.manager import TaskManager
    from ..artifact import BaseArtifactHandle
    from ..artifact.manager import ArtifactManager
    from ..dag.scheduler import DagScheduler


logger = logging.getLogger(__name__)


class JobPipelineStage(PipelineStage):
    """
    Provides access to job artifacts and handles invalidation.
    Task launching is delegated to the DagScheduler.
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
