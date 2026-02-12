from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager


class PipelineStage:
    """
    Base class for a stage in the artifact generation pipeline.
    Stages are now thin wrappers that provide access to artifacts.
    Task launching is delegated to the DagScheduler.
    """

    def __init__(
        self, task_manager: "TaskManager", artifact_manager: "ArtifactManager"
    ):
        self._task_manager = task_manager
        self._artifact_manager = artifact_manager

    @property
    def is_busy(self) -> bool:
        """Returns True if the stage has any active tasks."""
        return False

    def reconcile(self, doc: "Doc", generation_id: int):
        """
        Synchronizes the stage's state with the document.
        Default implementation does nothing; task launching is
        handled by DagScheduler.
        """
        pass

    def shutdown(self):
        """Clean up any resources held by this stage."""
        pass
