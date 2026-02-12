from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from .base import PipelineStage

if TYPE_CHECKING:
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)


class JobPipelineStage(PipelineStage):
    """
    Job generation is now handled by DagScheduler.
    This stage is kept for compatibility but does nothing.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: "Machine",
    ):
        super().__init__(task_manager, artifact_manager)
        self._machine = machine

    def shutdown(self):
        logger.debug("JobPipelineStage shutting down.")
