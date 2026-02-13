"""
Defines the PipelineGraph class for managing the dependency graph.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from ...core.doc import Doc
from ...core.layer import Layer
from ...core.workpiece import WorkPiece
from ..artifact.key import ArtifactKey
from .node import ArtifactNode


logger = logging.getLogger(__name__)


class PipelineGraph:
    """
    Container for all ArtifactNodes in the pipeline.

    Provides methods to build the graph from the Doc, find nodes,
    and identify dependencies.
    """

    def __init__(self):
        """Initialize an empty pipeline graph."""
        self._nodes: Dict[ArtifactKey, ArtifactNode] = {}

    def add_node(self, node: ArtifactNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.key] = node

    def get_node(self, key: ArtifactKey) -> Optional[ArtifactNode]:
        """Get a node by its ArtifactKey."""
        return self._nodes.get(key)

    def find_node(self, key: ArtifactKey) -> Optional[ArtifactNode]:
        """
        Find a node by its ArtifactKey.

        Alias for get_node for consistency with plan.md.
        """
        return self.get_node(key)

    def get_all_nodes(self) -> List[ArtifactNode]:
        """Return all nodes in the graph."""
        return list(self._nodes.values())

    def get_nodes_by_group(self, group: str) -> List[ArtifactNode]:
        """Return all nodes belonging to a specific group."""
        return [
            node for node in self._nodes.values() if node.key.group == group
        ]

    def build_from_doc(self, doc: Doc) -> None:
        """
        Build the graph from the Doc structure.

        This method traverses the document and creates nodes for:
        - All (WorkPiece, Step) pairs
        - All Steps
        - A single Job node

        Dependencies are established:
        - Steps depend on their (WorkPiece, Step) pair nodes
        - Job depends on all Steps
        """
        logger.debug("Building pipeline graph from Doc")

        self._nodes.clear()

        workpiece_step_nodes: Dict[Tuple[str, str], ArtifactNode] = {}
        step_nodes: Dict[str, ArtifactNode] = {}

        for layer in doc.children:
            if not isinstance(layer, Layer):
                continue

            workpieces = [
                child
                for child in layer.children
                if isinstance(child, WorkPiece)
            ]

            if layer.workflow is not None:
                for step in layer.workflow.steps:
                    step_key = ArtifactKey.for_step(step.uid)
                    step_node = ArtifactNode(key=step_key)
                    self.add_node(step_node)
                    step_nodes[step.uid] = step_node
                    logger.debug(f"Created Step node: {step_key}")

                    for wp in workpieces:
                        wp_step_key = ArtifactKey.for_workpiece(
                            wp.uid, step.uid
                        )
                        wp_step_node = ArtifactNode(key=wp_step_key)
                        self.add_node(wp_step_node)
                        workpiece_step_nodes[(wp.uid, step.uid)] = wp_step_node
                        logger.debug(
                            f"Created WorkPiece-Step node: {wp_step_key}"
                        )

        for layer in doc.children:
            if not isinstance(layer, Layer):
                continue

            if layer.workflow is not None:
                for step in layer.workflow.steps:
                    step_node = step_nodes.get(step.uid)
                    if step_node is None:
                        continue

                    for wp in layer.all_workpieces:
                        wp_step_node = workpiece_step_nodes.get(
                            (wp.uid, step.uid)
                        )
                        if wp_step_node is not None:
                            step_node.add_dependency(wp_step_node)
                            logger.debug(
                                f"Added dependency: Step {step.uid} -> "
                                f"WorkPiece-Step ({wp.uid}, {step.uid})"
                            )

        job_key = ArtifactKey.for_job()
        job_node = ArtifactNode(key=job_key)
        self.add_node(job_node)

        for step_node in step_nodes.values():
            job_node.add_dependency(step_node)
            logger.debug(f"Added dependency: Job -> Step {step_node.key.id}")

        logger.debug(f"Pipeline graph built: {len(self._nodes)} nodes total")

    def __repr__(self) -> str:
        return f"PipelineGraph(nodes={len(self._nodes)})"
