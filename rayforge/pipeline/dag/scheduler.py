from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING, Callable
from blinker import Signal
from ...core.step import Step
from ...core.workpiece import WorkPiece
from ..artifact.key import ArtifactKey
from ..context import GenerationContext
from .graph import PipelineGraph
from .node import ArtifactNode, NodeState


if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...shared.tasker.manager import TaskManager
    from ..artifact.manager import ArtifactManager
    from ..stage import (
        JobPipelineStage,
        StepPipelineStage,
        WorkPiecePipelineStage,
    )


logger = logging.getLogger(__name__)


class DagScheduler:
    """
    The scheduler for the DAG-based pipeline execution.

    Owns the PipelineGraph and is responsible for identifying
    which nodes are ready to be processed and orchestrating the
    launching of tasks.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        artifact_manager: "ArtifactManager",
        machine: Optional["Machine"],
    ):
        """Initialize the scheduler with required dependencies."""
        self.graph = PipelineGraph()
        self._task_manager = task_manager
        self._artifact_manager = artifact_manager
        self._machine = machine
        self._doc: Optional[Doc] = None
        self._generation_id: int = 0
        self._active_context: Optional[GenerationContext] = None
        self._invalidated_keys: set = set()

        self._workpiece_stage: Optional["WorkPiecePipelineStage"] = None
        self._step_stage: Optional["StepPipelineStage"] = None
        self._job_stage: Optional["JobPipelineStage"] = None

        self.generation_starting = Signal()
        self.visual_chunk_available = Signal()
        self.generation_finished = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.step_assembly_starting = Signal()
        self.step_render_artifact_ready = Signal()
        self.step_time_estimate_ready = Signal()
        self.step_generation_finished = Signal()
        self.job_generation_finished = Signal()
        self.job_generation_failed = Signal()

    @property
    def is_job_running(self) -> bool:
        """Check if a job generation is currently in progress."""
        if self._job_stage:
            return self._job_stage.is_running
        return False

    def set_doc(self, doc: Optional[Doc]) -> None:
        """Set the document reference for the scheduler."""
        self._doc = doc

    def set_generation_id(self, generation_id: int) -> None:
        """Set the current generation ID."""
        self._generation_id = generation_id

    def set_context(self, context: GenerationContext) -> None:
        """Set the current generation context."""
        self._active_context = context
        self._generation_id = context.generation_id

    def set_workpiece_stage(self, stage: "WorkPiecePipelineStage") -> None:
        """Set the workpiece pipeline stage."""
        self._workpiece_stage = stage
        self._workpiece_stage.generation_finished.connect(
            self._bridge_workpiece_finished
        )
        self._workpiece_stage.generation_starting.connect(
            self._bridge_workpiece_starting
        )

    def set_step_stage(self, stage: "StepPipelineStage") -> None:
        """Set the step pipeline stage."""
        self._step_stage = stage
        self._step_stage.assembly_starting.connect(
            self._bridge_assembly_starting
        )

    def set_job_stage(self, stage: "JobPipelineStage") -> None:
        """Set the job pipeline stage."""
        self._job_stage = stage
        self._job_stage.job_generation_finished.connect(
            self._bridge_job_finished
        )
        self._job_stage.job_generation_failed.connect(self._bridge_job_failed)

    def _bridge_workpiece_finished(self, sender, **kwargs):
        """Re-emit workpiece finished signal for Pipeline compatibility."""
        self.generation_finished.send(self, **kwargs)

    def _bridge_workpiece_starting(self, sender, **kwargs):
        """Re-emit workpiece starting signal for Pipeline compatibility."""
        self.generation_starting.send(self, **kwargs)

    def _bridge_assembly_starting(self, sender, **kwargs):
        """Re-emit assembly starting signal for Pipeline compatibility."""
        self.step_assembly_starting.send(self, **kwargs)

    def _bridge_job_finished(self, sender, **kwargs):
        """Re-emit job finished signal for Pipeline compatibility."""
        self.job_generation_finished.send(self, **kwargs)

    def _bridge_job_failed(self, sender, **kwargs):
        """Re-emit job failed signal for Pipeline compatibility."""
        self.job_generation_failed.send(self, **kwargs)

    def sync_graph_with_artifact_manager(self) -> None:
        """
        Sync graph node states with the artifact manager.
        """
        for node in self.graph.get_all_nodes():
            if any(
                dep.key in self._invalidated_keys for dep in node.dependencies
            ):
                logger.debug(
                    f"Node {node.key} has invalidated dependencies. "
                    "Skipping sync to VALID."
                )
                continue
            if node.key.group == "workpiece":
                handle = self._artifact_manager.get_workpiece_handle(
                    node.key, self._generation_id
                )
                if handle is None and self._generation_id > 1:
                    handle = self._artifact_manager.get_workpiece_handle(
                        node.key, self._generation_id - 1
                    )
                if handle is not None:
                    node.state = NodeState.VALID
                    logger.debug(
                        f"Synced node {node.key} to VALID (handle exists)"
                    )
            elif node.key.group == "step":
                ops_handle = self._artifact_manager.get_step_ops_handle(
                    node.key, self._generation_id
                )
                if ops_handle is None and self._generation_id > 1:
                    ops_handle = self._artifact_manager.get_step_ops_handle(
                        node.key, self._generation_id - 1
                    )
                if ops_handle is not None:
                    node.state = NodeState.VALID
                    logger.debug(
                        f"Synced node {node.key} to VALID (ops handle exists)"
                    )
            elif node.key.group == "job":
                job_handle = self._artifact_manager.get_job_handle(
                    node.key, self._generation_id
                )
                if job_handle is None and self._generation_id > 1:
                    job_handle = self._artifact_manager.get_job_handle(
                        node.key, self._generation_id - 1
                    )
                if job_handle is not None:
                    node.state = NodeState.VALID
                    logger.debug(
                        f"Synced node {node.key} to VALID (job handle exists)"
                    )

        for key in self._invalidated_keys:
            node = self.graph.find_node(key)
            if node is not None:
                node.mark_dirty()
                logger.debug(f"Re-applied invalidation for node {node.key}")
        self._invalidated_keys.clear()

    def has_pending_work(self) -> bool:
        """
        Check if there is any pending work in the pipeline.
        """
        for node in self.graph.get_all_nodes():
            if node.state == NodeState.PROCESSING:
                return True
        return False

    def get_processing_generation_ids(self) -> set:
        """
        Get the set of generation IDs that may have running tasks.
        """
        for node in self.graph.get_all_nodes():
            if node.state == NodeState.PROCESSING:
                return {self._generation_id, self._generation_id - 1}
        return set()

    def get_ready_nodes(self, group: str) -> List[ArtifactKey]:
        """
        Get nodes that are ready to be processed.
        """
        nodes = self.graph.get_nodes_by_group(group)
        ready_keys = []
        for node in nodes:
            if node.is_ready():
                ready_keys.append(node.key)
            else:
                dep_states = ", ".join(
                    f"{d.key.id}:{d.state.value}" for d in node.dependencies
                )
                logger.debug(
                    f"Node {node.key} not ready: state={node.state.value}, "
                    f"deps=[{dep_states}]"
                )
        return ready_keys

    def find_node(self, key: ArtifactKey) -> Optional[ArtifactNode]:
        """
        Find a node in the graph by its ArtifactKey.
        """
        return self.graph.find_node(key)

    def set_node_state(self, key: ArtifactKey, state: NodeState) -> bool:
        """
        Set the state of a node.
        """
        node = self.graph.find_node(key)
        if node is None:
            return False
        node.state = state
        logger.debug(f"Set node {key} state to {state.value}")
        return True

    def mark_node_dirty(self, key: ArtifactKey) -> bool:
        """
        Mark a node and all its dependents as dirty.
        """
        node = self.graph.find_node(key)
        if node is None:
            self._invalidated_keys.add(key)
            return False
        node.mark_dirty()
        self._invalidated_keys.add(key)
        return True

    def on_artifact_state_changed(self, key: ArtifactKey, state: str) -> None:
        """
        Callback method for ArtifactManager to notify of state changes.
        """
        state_map = {
            "valid": NodeState.VALID,
            "processing": NodeState.PROCESSING,
            "error": NodeState.ERROR,
            "stale": NodeState.DIRTY,
        }
        node_state = state_map.get(state)
        if node_state is None:
            logger.warning(f"Unknown state string: {state}")
            return
        logger.debug(f"on_artifact_state_changed: key={key}, state={state}")
        self.set_node_state(key, node_state)

    def process_graph(self) -> None:
        """
        Process the graph and launch tasks for ready nodes.
        """
        logger.debug("Processing graph for workpiece nodes")
        keys_to_generate = self.get_ready_nodes("workpiece")
        logger.debug(
            f"DagScheduler: found {len(keys_to_generate)} workpiece keys "
            f"to generate: {keys_to_generate}"
        )

        for key in keys_to_generate:
            if ":" not in key.id:
                logger.warning(f"Invalid workpiece key format: {key.id}")
                continue
            workpiece_uid, step_uid = key.id.split(":", 1)
            workpiece = (
                self._doc.find_descendant_by_uid(workpiece_uid)
                if self._doc
                else None
            )
            step = (
                self._doc.find_descendant_by_uid(step_uid)
                if self._doc
                else None
            )

            if isinstance(workpiece, WorkPiece) and isinstance(step, Step):
                logger.debug(
                    f"DagScheduler: Launching task for "
                    f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
                )
                if self._workpiece_stage:
                    self._workpiece_stage.launch_task(
                        step,
                        workpiece,
                        self._generation_id,
                        self._active_context,
                    )
            elif not isinstance(workpiece, WorkPiece):
                logger.warning(
                    f"Workpiece {workpiece_uid} not found in doc. "
                    "Skipping generation."
                )
                self._artifact_manager.mark_done(key, self._generation_id)
            elif not isinstance(step, Step):
                logger.warning(
                    f"Step {step_uid} not found in doc. Skipping generation."
                )
                self._artifact_manager.mark_done(key, self._generation_id)

        logger.debug("Processing graph for step nodes")
        step_nodes = self.graph.get_nodes_by_group("step")
        logger.debug(
            f"DagScheduler: found {len(step_nodes)} step nodes in graph"
        )
        for node in step_nodes:
            dep_states = ", ".join(
                f"{d.key.id}:{d.state.value}" for d in node.dependencies
            )
            logger.debug(
                f"  Step node {node.key}: state={node.state.value}, "
                f"deps=[{dep_states}]"
            )
        step_keys_to_generate = self.get_ready_nodes("step")
        logger.debug(
            f"DagScheduler: found {len(step_keys_to_generate)} step keys "
            f"to generate: {step_keys_to_generate}"
        )

        for key in step_keys_to_generate:
            step_uid = key.id
            step = (
                self._doc.find_descendant_by_uid(step_uid)
                if self._doc
                else None
            )

            if isinstance(step, Step):
                logger.debug(
                    f"DagScheduler: Launching assembly task for "
                    f"step_uid={step_uid}"
                )
                if self._step_stage:
                    self._step_stage.launch_task(
                        step, self._generation_id, self._active_context
                    )
            else:
                logger.warning(
                    f"Step {step_uid} not found in doc. Skipping assembly."
                )
                self._artifact_manager.mark_done(key, self._generation_id)

    def generate_job(
        self,
        step_uids: List[str],
        on_done: Optional[Callable] = None,
        job_key: Optional[ArtifactKey] = None,
    ):
        """
        Start the asynchronous task to assemble and encode the final job.
        """
        assert self._job_stage is not None
        assert self._doc is not None
        self._job_stage.generate_job(
            step_uids,
            self._generation_id,
            self._active_context,
            self._doc,
            on_done,
            job_key,
        )
