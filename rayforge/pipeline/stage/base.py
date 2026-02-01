from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager


class PipelineStage(ABC):
    """
    Abstract base class for a stage in the artifact generation pipeline.
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

    @abstractmethod
    def reconcile(self, doc: "Doc"):
        """Synchronizes the stage's state with the document."""
        raise NotImplementedError

    def shutdown(self):
        """Clean up any resources held by this stage."""
        pass
