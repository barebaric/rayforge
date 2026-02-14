"""
Defines the ArtifactNode class for representing artifacts in the DAG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from ..artifact.key import ArtifactKey


logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Lifecycle states for nodes in the dependency graph."""

    DIRTY = "dirty"
    PROCESSING = "processing"
    VALID = "valid"
    ERROR = "error"


@dataclass
class ArtifactNode:
    """
    Represents a single artifact in the dependency graph.

    Each node holds its ArtifactKey, current state, and lists of its
    dependencies (children) and dependents (parents).
    """

    key: ArtifactKey
    state: NodeState = field(default=NodeState.DIRTY)
    dependencies: List[ArtifactNode] = field(default_factory=list)
    dependents: List[ArtifactNode] = field(default_factory=list)

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
