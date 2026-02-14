"""
Defines the DagScheduler class for orchestrating pipeline execution.
"""

from __future__ import annotations

import logging
from asyncio.exceptions import CancelledError
from typing import List, Optional, TYPE_CHECKING, Tuple, Dict, Callable
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
    from ...shared.tasker.task import Task
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
        self._job_running: bool = False
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
        return self._job_running

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

    def set_step_stage(self, stage: "StepPipelineStage") -> None:
        """Set the step pipeline stage."""
        self._step_stage = stage

    def set_job_stage(self, stage: "JobPipelineStage") -> None:
        """Set the job pipeline stage."""
        self._job_stage = stage

    def sync_graph_with_artifact_manager(self) -> None:
        """
        Sync graph node states with the artifact manager.

        After building the graph from the document, this method checks
        which artifacts already exist and are valid in the artifact manager,
        and updates node states accordingly.

        This method checks both the current generation and the previous
        generation to allow reuse of valid artifacts across generations.

        After syncing, any keys that were marked dirty before the graph
        rebuild are re-invalidated to preserve the intended state.
        """
        for node in self.graph.get_all_nodes():
            # If any of this node's dependencies were explicitly invalidated,
            # then this node cannot be considered VALID from a cache hit,
            # as its inputs have changed. It must be regenerated.
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

        Returns True if any node is in PROCESSING state,
        indicating work is actively in progress.

        Note: DIRTY nodes are not counted as pending work because they
        may have unsatisfied dependencies (e.g., no view context) and
        won't be processed until those dependencies become available.

        Returns:
            True if there is pending work, False otherwise.
        """
        for node in self.graph.get_all_nodes():
            if node.state == NodeState.PROCESSING:
                return True
        return False

    def get_processing_generation_ids(self) -> set:
        """
        Get the set of generation IDs that may have running tasks.

        When there are PROCESSING nodes, we preserve both the current
        generation and the previous generation, because tasks might have
        been launched just before a generation ID increment.

        Returns:
            Set of generation IDs that should be preserved.
        """
        for node in self.graph.get_all_nodes():
            if node.state == NodeState.PROCESSING:
                return {self._generation_id, self._generation_id - 1}
        return set()

    def get_ready_nodes(self, group: str) -> List[ArtifactKey]:
        """
        Get nodes that are ready to be processed.

        A node is ready if it is DIRTY and all its dependencies are VALID.

        Args:
            group: The artifact group to filter (e.g., "workpiece", "step")

        Returns:
            List of ArtifactKeys for nodes that are ready.
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

        Args:
            key: The ArtifactKey to find

        Returns:
            The ArtifactNode if found, None otherwise
        """
        return self.graph.find_node(key)

    def set_node_state(self, key: ArtifactKey, state: NodeState) -> bool:
        """
        Set the state of a node.

        Args:
            key: The ArtifactKey of the node
            state: The new NodeState

        Returns:
            True if the node was found and state was set, False otherwise
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

        Args:
            key: The ArtifactKey of the node to mark dirty

        Returns:
            True if the node was found, False otherwise
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

        This method is called by the ArtifactManager when artifact state
        changes, allowing the DAG to stay synchronized.

        Args:
            key: The ArtifactKey whose state changed.
            state: The new state as a string (e.g., "valid", "processing").
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

        This method finds all DIRTY nodes whose dependencies are all VALID
        and launches generation tasks for workpiece and step nodes.
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
                self._launch_workpiece_task(step, workpiece)
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
                self._launch_step_task(step)
            else:
                logger.warning(
                    f"Step {step_uid} not found in doc. Skipping assembly."
                )
                self._artifact_manager.mark_done(key, self._generation_id)

    def _create_and_register_task(
        self,
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        workpiece_dict: Dict,
        step: "Step",
        workpiece: "WorkPiece",
        settings: Dict,
        laser_dict: Optional[Dict],
        generation_id: int,
        workpiece_size: Tuple[float, float],
    ):
        """
        Creates subprocess task and registers it in active_tasks.
        """
        from ..stage.workpiece_runner import (
            make_workpiece_artifact_in_subprocess,
        )

        task_key = key
        context = self._active_context

        self._task_manager.run_process(
            make_workpiece_artifact_in_subprocess,
            self._artifact_manager._store,
            workpiece_dict,
            step.opsproducer_dict,
            step.modifiers_dicts,
            step.per_workpiece_transformers_dicts,
            laser_dict,
            settings,
            generation_id,
            workpiece_size,
            "workpiece",
            key=task_key,
            when_done=lambda t: self._on_task_complete(
                t,
                task_key,
                ledger_key,
                generation_id,
                step,
                workpiece,
                context,
            ),
            when_event=lambda task,
            event_name,
            data: self._on_task_event_received(
                task, event_name, data, step.uid
            ),
        )

    def _launch_workpiece_task(self, step: "Step", workpiece: "WorkPiece"):
        """Starts the asynchronous task to generate operations."""
        key = ArtifactKey.for_workpiece(workpiece.uid, step.uid)
        ledger_key = key

        assert self._workpiece_stage is not None
        if not self._workpiece_stage.validate_for_launch(key, workpiece):
            logger.debug(
                f"DagScheduler: Validation failed for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return

        self.generation_starting.send(
            self,
            step=step,
            workpiece=workpiece,
            generation_id=self._generation_id,
        )

        prep_result = self._workpiece_stage.prepare_task_settings(step)
        if prep_result is None:
            logger.debug(
                f"DagScheduler: prep_result is None for "
                f"step_uid={step.uid}, workpiece_uid={workpiece.uid}"
            )
            return
        settings, laser_dict = prep_result

        workpiece_dict = self._workpiece_stage.prepare_workpiece_dict(
            workpiece
        )

        node = self.graph.find_node(ledger_key)
        if node is not None:
            node.state = NodeState.PROCESSING

        if self._active_context is not None:
            self._active_context.add_task(key)

        self._create_and_register_task(
            key,
            ledger_key,
            workpiece_dict,
            step,
            workpiece,
            settings,
            laser_dict,
            self._generation_id,
            workpiece.size,
        )

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict, step_uid: str
    ):
        """Handles events from a background task."""
        key = task.key
        ledger_key = key

        handle_dict = data.get("handle_dict")
        generation_id = data.get("generation_id")

        if generation_id is None:
            logger.error(
                f"[{key}] Task event '{event_name}' missing "
                f"generation id. Ignoring."
            )
            return

        if handle_dict is None and event_name != "artifact_created":
            logger.error(
                f"[{key}] Task event '{event_name}' missing handle. Ignoring."
            )
            return

        node = self.graph.find_node(ledger_key)
        if node is None or node.state != NodeState.PROCESSING:
            logger.debug(
                f"[{key}] No PROCESSING node found for event '{event_name}'. "
                f"Ignoring."
            )
            return

        try:
            if handle_dict is None:
                handle = None
            else:
                handle = self._artifact_manager.adopt_artifact(
                    key, handle_dict
                )

            if event_name == "artifact_created":
                assert self._workpiece_stage is not None
                self._workpiece_stage.handle_artifact_created(
                    key, ledger_key, handle, generation_id, step_uid
                )
                return

            if event_name == "visual_chunk_ready":
                assert self._workpiece_stage is not None
                self._workpiece_stage.handle_visual_chunk_ready(
                    key, handle, generation_id, step_uid
                )
        except Exception as e:
            logger.error(
                f"Failed to process event '{event_name}': {e}", exc_info=True
            )

    def _on_task_complete(
        self,
        task: "Task",
        key: ArtifactKey,
        ledger_key: ArtifactKey,
        task_generation_id: int,
        step: "Step",
        workpiece: "WorkPiece",
        context: Optional[GenerationContext],
    ):
        """Callback for when an ops generation task finishes."""
        if context is not None:
            context.task_did_finish(key)

        assert self._workpiece_stage is not None
        if not self._workpiece_stage.validate_task_completion(
            key, ledger_key, task_generation_id
        ):
            return

        task_status = task.get_status()
        logger.debug(f"[{key}] Task status is '{task_status}'.")

        if task_status == "canceled":
            assert self._workpiece_stage is not None
            self._workpiece_stage.handle_canceled_task(
                key, ledger_key, step, workpiece, task_generation_id
            )
            return

        handle = None
        if task_status == "completed":
            assert self._workpiece_stage is not None
            continue_processing, handle = (
                self._workpiece_stage.handle_completed_task(
                    key,
                    ledger_key,
                    task,
                    step,
                    workpiece,
                    task_generation_id,
                    task_generation_id,
                )
            )
            if not continue_processing:
                return
        else:
            assert self._workpiece_stage is not None
            self._workpiece_stage.handle_failed_task(
                ledger_key, step, workpiece, task_generation_id
            )

        self.generation_finished.send(
            self,
            step=step,
            workpiece=workpiece,
            handle=handle,
            generation_id=task_generation_id,
            task_status=task_status,
        )

    def _launch_step_task(self, step: "Step"):
        """Starts the asynchronous task to assemble a step artifact."""
        assert self._step_stage is not None
        if not self._step_stage.validate_dependencies(step):
            logger.debug(
                f"DagScheduler: Step assembly dependencies not met for "
                f"step_uid={step.uid}"
            )
            return

        assembly_info, retained_handles = (
            self._step_stage.collect_assembly_info(step, self._generation_id)
        )
        if not assembly_info:
            for handle in retained_handles:
                self._artifact_manager.release_handle(handle)
            logger.debug(
                f"DagScheduler: No assembly info for step_uid={step.uid}"
            )
            return

        self._step_stage.store_retained_handles(step.uid, retained_handles)

        if step.layer:
            for wp in step.layer.all_workpieces:
                handle = self._artifact_manager.get_workpiece_handle(
                    ArtifactKey.for_workpiece(wp.uid, step.uid),
                    self._generation_id,
                )
                if handle:
                    self.step_assembly_starting.send(
                        self,
                        step=step,
                        workpiece=wp,
                        handle=handle,
                    )

        ledger_key = ArtifactKey.for_step(step.uid)
        node = self.graph.find_node(ledger_key)
        if node is not None:
            node.state = NodeState.PROCESSING

        from ..stage.step_runner import make_step_artifact_in_subprocess

        task_key = ArtifactKey.for_step(step.uid)
        context = self._active_context

        if context is not None:
            context.add_task(task_key)

        assert self._machine is not None

        self._task_manager.run_process(
            make_step_artifact_in_subprocess,
            self._artifact_manager._store,
            assembly_info,
            step.uid,
            self._generation_id,
            step.per_step_transformers_dicts,
            self._machine.max_cut_speed,
            self._machine.max_travel_speed,
            self._machine.acceleration,
            "step",
            key=task_key,
            when_done=lambda t: self._on_step_task_complete(
                t, task_key, step, self._generation_id, context
            ),
            when_event=lambda task, event_name, data: (
                self._step_stage.handle_task_event(
                    task, event_name, data, step
                )
                if self._step_stage
                else None
            ),
        )

    def _on_step_task_complete(
        self,
        task: "Task",
        task_key: ArtifactKey,
        step: "Step",
        task_generation_id: int,
        context: Optional[GenerationContext],
    ):
        """Callback for when a step assembly task finishes."""
        if context is not None:
            context.task_did_finish(task_key)

        step_uid = step.uid
        ledger_key = ArtifactKey.for_step(step_uid)

        assert self._step_stage is not None
        self._step_stage.release_retained_handles(step_uid)

        if not self._artifact_manager.is_generation_current(
            ledger_key, task_generation_id
        ):
            logger.debug(f"Ignoring stale step completion for {step_uid}")
            return

        if task.get_status() == "completed":
            try:
                task.result()
                render_handle = self._artifact_manager.get_step_render_handle(
                    step_uid
                )
                logger.debug(
                    f"_on_step_task_complete: render_handle={render_handle}"
                )
                self._artifact_manager.complete_generation(
                    ledger_key,
                    task_generation_id,
                )
                node = self.graph.find_node(ledger_key)
                if node is not None:
                    node.state = NodeState.VALID
                logger.debug("_on_step_task_complete: ops_handle set to DONE")
            except Exception as e:
                logger.error(f"Error on step assembly result: {e}")
                node = self.graph.find_node(ledger_key)
                if node is not None:
                    node.state = NodeState.ERROR
        else:
            logger.warning(f"Step assembly for {step_uid} failed.")
            node = self.graph.find_node(ledger_key)
            if node is not None:
                node.state = NodeState.ERROR

        assert self._step_stage is not None
        self._step_stage.generation_finished.send(
            self._step_stage, step=step, generation_id=task_generation_id
        )

    def generate_job(
        self,
        step_uids: List[str],
        on_done: Optional[Callable] = None,
        job_key: Optional[ArtifactKey] = None,
    ):
        """
        Start the asynchronous task to assemble and encode the final job.

        This method validates that all step dependencies are ready, creates
        a JobDescription, and launches the job generation task.

        Args:
            step_uids: List of step UIDs to include in the job.
            on_done: Optional callback executed upon completion.
            job_key: Optional ArtifactKey for the job. If not provided,
                a new one will be created.
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

        assert self._job_stage is not None
        if not self._job_stage.validate_dependencies(
            step_uids, self._generation_id
        ):
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

        step_handles = self._job_stage.collect_step_handles(
            step_uids, self._generation_id
        )
        if step_handles is None:
            logger.error("Failed to collect step handles for job generation.")
            if on_done:
                on_done(None, RuntimeError("Failed to collect step handles."))
            return

        node = self.graph.find_node(job_key)
        if node is not None:
            node.state = NodeState.PROCESSING

        logger.info(f"Starting job generation with {len(step_handles)} steps.")

        assert self._doc is not None
        assert self._machine is not None

        job_desc_dict = {
            "step_artifact_handles_by_uid": step_handles,
            "machine_dict": self._machine.to_dict(),
            "doc_dict": self._doc.to_dict(),
        }

        self._launch_job_task(
            job_desc_dict, job_key, on_done, self._generation_id
        )

    def _launch_job_task(
        self,
        job_desc_dict: Dict,
        job_key: ArtifactKey,
        on_done: Optional[Callable],
        generation_id: int,
    ):
        """
        Launch the subprocess task for job generation.

        Args:
            job_desc_dict: The job description as a dictionary.
            job_key: The ArtifactKey for this job.
            on_done: Optional callback to execute on completion.
            generation_id: The generation ID for this job.
        """
        from ..stage.job_runner import make_job_artifact_in_subprocess

        self._job_running = True

        context = self._active_context

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
            when_done=lambda t: self._on_job_task_complete(
                t, job_key, generation_id, on_done, context
            ),
            when_event=lambda task, event_name, data: (
                self._job_stage.handle_task_event(
                    task, event_name, data, job_key, generation_id
                )
                if self._job_stage
                else None
            ),
        )

    def _on_job_task_complete(
        self,
        task: "Task",
        job_key: ArtifactKey,
        generation_id: int,
        on_done: Optional[Callable],
        context: Optional[GenerationContext],
    ):
        """Callback for when a job generation task finishes."""
        if context is not None:
            context.task_did_finish(job_key)

        assert self._job_stage is not None
        self._job_stage.release_retained_handles()

        task_status = task.get_status()
        final_handle = None
        error = None

        if task_status == "completed":
            final_handle = self._artifact_manager.get_job_handle(
                job_key, generation_id
            )
            if final_handle:
                logger.info("Job generation successful.")
                node = self.graph.find_node(job_key)
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

            if generation_id is not None:
                node = self.graph.find_node(job_key)
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
