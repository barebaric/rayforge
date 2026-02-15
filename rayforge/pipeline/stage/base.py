from __future__ import annotations
from typing import TYPE_CHECKING
from blinker import Signal

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager
    from ..artifact.key import ArtifactKey
    from ..dag.node import NodeState


class PipelineStage:
    """
    Base class for a stage in the artifact generation pipeline.

    Stages are responsible for launching tasks and emitting signals about
    state changes. The scheduler subscribes to these signals and manages
    the DAG state accordingly.

    Attributes:
        node_state_changed: Signal emitted when a node's state changes.
            Emits with (key: ArtifactKey, state: NodeState).
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
    ):
        self._task_manager = task_manager
        self._artifact_manager = artifact_manager
        self.node_state_changed = Signal()

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

    def _emit_node_state(self, key: "ArtifactKey", state: "NodeState") -> None:
        """Helper to emit node state changes."""
        self.node_state_changed.send(self, key=key, state=state)
