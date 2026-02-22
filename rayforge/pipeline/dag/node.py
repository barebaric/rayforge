"""
Defines the ArtifactNode class for representing artifacts in the DAG.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from ..artifact.key import ArtifactKey


if TYPE_CHECKING:
    from ..artifact.manager import ArtifactManager


logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Lifecycle states for nodes in the dependency graph."""

    DIRTY = "dirty"
    PROCESSING = "processing"
    VALID = "valid"
    ERROR = "error"


class ArtifactNode:
    """
    Represents a single artifact in the dependency graph.

    Each node holds its ArtifactKey, current state, and lists of its
    dependencies (children) and dependents (parents).

    State is delegated to the ArtifactManager. An error is raised if
    state is queried without a manager and generation_id being set.
    """

    def __init__(
        self,
        key: ArtifactKey,
        generation_id: int,
        dependencies: Optional[List[ArtifactNode]] = None,
        dependents: Optional[List[ArtifactNode]] = None,
        _artifact_manager: Optional["ArtifactManager"] = None,
    ):
        self.key = key
        self.generation_id = generation_id
        self.dependencies = dependencies if dependencies is not None else []
        self.dependents = dependents if dependents is not None else []
        self._artifact_manager = _artifact_manager

    def _check_manager(self) -> None:
        """Raise an error if manager is not set."""
        if self._artifact_manager is None:
            raise RuntimeError(
                f"ArtifactNode {self.key} has no ArtifactManager. "
                "State cannot be queried without a manager."
            )

    @property
    def state(self) -> NodeState:
        """Get the node state from ArtifactManager."""
        self._check_manager()
        assert self._artifact_manager is not None
        return self._artifact_manager.get_state(self.key, self.generation_id)

    @state.setter
    def state(self, value: NodeState) -> None:
        """Set the node state in ArtifactManager."""
        self._check_manager()
        assert self._artifact_manager is not None
        self._artifact_manager.set_state(self.key, self.generation_id, value)

    def add_dependency(self, node: ArtifactNode) -> None:
        """Add a dependency node (child)."""
        if node not in self.dependencies:
            self.dependencies.append(node)
            if self not in node.dependents:
                node.dependents.append(self)

    def add_dependent(self, node: ArtifactNode) -> None:
        """Add a dependent node (parent)."""
        if node not in self.dependents:
            self.dependents.append(node)
            if self not in node.dependencies:
                node.dependencies.append(self)

    def mark_dirty(self) -> None:
        """
        Mark this node and all its dependents as dirty.

        This recursively propagates up the graph to invalidate all
        artifacts that depend on this one.
        """
        if self.state != NodeState.DIRTY:
            logger.debug(f"Marking node {self.key} as DIRTY")
            self.state = NodeState.DIRTY
            for dependent in self.dependents:
                dependent.mark_dirty()

    def is_ready(self) -> bool:
        """
        Check if this node is ready to be processed.

        A node is ready if it is DIRTY and all its dependencies are VALID.
        """
        if self.state != NodeState.DIRTY:
            return False
        return all(dep.state == NodeState.VALID for dep in self.dependencies)

    def __repr__(self) -> str:
        return f"ArtifactNode(key={self.key}, state={self.state.value})"
