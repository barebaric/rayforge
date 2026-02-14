"""
Tests for the DagScheduler class.
"""

import unittest
from typing import cast
from unittest.mock import MagicMock

from rayforge.pipeline.artifact import StepOpsArtifactHandle, JobArtifactHandle
from rayforge.pipeline.artifact.handle import BaseArtifactHandle
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.artifact.manager import ArtifactManager
from rayforge.pipeline.context import GenerationContext
from rayforge.pipeline.dag.node import ArtifactNode, NodeState
from rayforge.pipeline.dag.scheduler import DagScheduler
from rayforge.pipeline.stage import (
    JobPipelineStage,
    StepPipelineStage,
    WorkPiecePipelineStage,
)


WP_UID_1 = "550e8400-e29b-41d4-a716-446655440001"
STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440003"


class TestDagScheduler:
    """Tests for the DagScheduler class."""

    def _make_scheduler(self):
        """Helper to create a scheduler with mocked dependencies."""
        task_manager = MagicMock()
        artifact_manager = MagicMock()
        machine = MagicMock()
        scheduler = DagScheduler(task_manager, artifact_manager, machine)

        workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
        workpiece_stage.generation_finished = MagicMock()
        workpiece_stage.generation_starting = MagicMock()
        step_stage = MagicMock(spec=StepPipelineStage)
        step_stage.assembly_starting = MagicMock()
        job_stage = MagicMock(spec=JobPipelineStage)
        job_stage.job_generation_finished = MagicMock()
        job_stage.job_generation_failed = MagicMock()

        step_stage.validate_dependencies.return_value = True
        step_stage.validate_geometry_match.return_value = True
        job_stage.validate_dependencies.return_value = True
        workpiece_stage.validate_for_launch.return_value = True
        workpiece_stage.prepare_task_settings.return_value = ({}, {})

        scheduler.set_workpiece_stage(workpiece_stage)
        scheduler.set_step_stage(step_stage)
        scheduler.set_job_stage(job_stage)

        return scheduler

    def test_scheduler_initialization(self):
        """Test creating a scheduler."""
        scheduler = self._make_scheduler()
        assert scheduler.graph is not None
        assert len(scheduler.graph.get_all_nodes()) == 0

    def test_get_ready_nodes_empty_graph(self):
        """Test getting ready nodes from empty graph."""
        scheduler = self._make_scheduler()
        ready = scheduler.get_ready_nodes("workpiece")
        assert len(ready) == 0

    def test_get_ready_nodes_filters_by_group(self):
        """Test that get_ready_nodes filters by group."""
        scheduler = self._make_scheduler()

        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.DIRTY)
        step_node = ArtifactNode(key=step_key, state=NodeState.DIRTY)

        scheduler.graph.add_node(wp_node)
        scheduler.graph.add_node(step_node)

        wp_ready = scheduler.get_ready_nodes("workpiece")
        step_ready = scheduler.get_ready_nodes("step")

        assert len(wp_ready) == 1
        assert len(step_ready) == 1
        assert wp_ready[0] == wp_key
        assert step_ready[0] == step_key

    def test_get_ready_nodes_excludes_valid_nodes(self):
        """Test that VALID nodes are not included in ready list."""
        scheduler = self._make_scheduler()

        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)

        scheduler.graph.add_node(wp_node)

        ready = scheduler.get_ready_nodes("workpiece")
        assert len(ready) == 0

    def test_get_ready_nodes_excludes_nodes_with_dirty_deps(self):
        """Test that nodes with DIRTY dependencies are not ready."""
        scheduler = self._make_scheduler()

        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.DIRTY)
        step_node = ArtifactNode(key=step_key, state=NodeState.DIRTY)

        step_node.add_dependency(wp_node)

        scheduler.graph.add_node(wp_node)
        scheduler.graph.add_node(step_node)

        ready = scheduler.get_ready_nodes("step")
        assert len(ready) == 0

    def test_find_node(self):
        """Test finding a node by key."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key)

        scheduler.graph.add_node(node)

        found = scheduler.find_node(key)
        assert found is node

    def test_find_node_not_found(self):
        """Test finding a non-existent node."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
        found = scheduler.find_node(key)

        assert found is None

    def test_set_node_state(self):
        """Test setting the state of a node."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)

        scheduler.graph.add_node(node)

        result = scheduler.set_node_state(key, NodeState.VALID)

        assert result is True
        assert node.state == NodeState.VALID

    def test_set_node_state_not_found(self):
        """Test setting state on non-existent node."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
        result = scheduler.set_node_state(key, NodeState.VALID)

        assert result is False

    def test_mark_node_dirty(self):
        """Test marking a node dirty."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.VALID)

        scheduler.graph.add_node(node)

        result = scheduler.mark_node_dirty(key)

        assert result is True
        assert node.state == NodeState.DIRTY

    def test_mark_node_dirty_propagates(self):
        """Test that marking a node dirty propagates to dependents."""
        scheduler = self._make_scheduler()

        wp_key = ArtifactKey.for_workpiece(WP_UID_1)
        step_key = ArtifactKey.for_step(STEP_UID_1)

        wp_node = ArtifactNode(key=wp_key, state=NodeState.VALID)
        step_node = ArtifactNode(key=step_key, state=NodeState.VALID)

        step_node.add_dependency(wp_node)

        scheduler.graph.add_node(wp_node)
        scheduler.graph.add_node(step_node)

        scheduler.mark_node_dirty(wp_key)

        assert wp_node.state == NodeState.DIRTY
        assert step_node.state == NodeState.DIRTY

    def test_mark_node_dirty_not_found(self):
        """Test marking a non-existent node dirty."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
        result = scheduler.mark_node_dirty(key)

        assert result is False

    def test_process_graph(self):
        """Test that process_graph can be called without error."""
        scheduler = self._make_scheduler()
        scheduler.process_graph()

    def test_on_artifact_state_changed_valid(self):
        """Test that callback updates node to VALID."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)
        scheduler.graph.add_node(node)

        scheduler.on_artifact_state_changed(key, "valid")

        assert node.state == NodeState.VALID

    def test_on_artifact_state_changed_processing(self):
        """Test that callback updates node to PROCESSING."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)
        scheduler.graph.add_node(node)

        scheduler.on_artifact_state_changed(key, "processing")

        assert node.state == NodeState.PROCESSING

    def test_on_artifact_state_changed_error(self):
        """Test that callback updates node to ERROR."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)
        scheduler.graph.add_node(node)

        scheduler.on_artifact_state_changed(key, "error")

        assert node.state == NodeState.ERROR

    def test_on_artifact_state_changed_unknown_state(self):
        """Test that callback handles unknown state gracefully."""
        scheduler = self._make_scheduler()

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)
        scheduler.graph.add_node(node)

        scheduler.on_artifact_state_changed(key, "unknown")

        assert node.state == NodeState.DIRTY

    def test_artifact_manager_syncs_dag_state(self):
        """Test that DAG manages node state directly."""
        mock_store = MagicMock()
        scheduler = self._make_scheduler()
        manager = ArtifactManager(mock_store)

        key = ArtifactKey.for_workpiece(WP_UID_1)
        node = ArtifactNode(key=key, state=NodeState.DIRTY)
        scheduler.graph.add_node(node)

        assert node.state == NodeState.DIRTY

        node.state = NodeState.PROCESSING
        assert node.state == NodeState.PROCESSING

        mock_handle = MagicMock(spec=BaseArtifactHandle)
        manager.cache_handle(key, mock_handle, 1)
        node.state = NodeState.VALID
        assert node.state == NodeState.VALID


class TestDagSchedulerJobGeneration:
    """Tests for job generation in the DagScheduler."""

    STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440101"
    STEP_UID_2 = "550e8400-e29b-41d4-a716-446655440102"

    def _make_scheduler(self, artifact_manager=None):
        """Helper to create a scheduler with mocked dependencies."""
        task_manager = MagicMock()
        if artifact_manager is None:
            artifact_manager = MagicMock()
        machine = MagicMock()
        machine.to_dict.return_value = {"name": "test-machine"}
        doc = MagicMock()
        doc.to_dict.return_value = {"layers": []}
        scheduler = DagScheduler(task_manager, artifact_manager, machine)
        scheduler.set_doc(doc)
        scheduler.set_generation_id(1)

        # Patch _job_stage with a real instance in the test helper
        # This is necessary for handle_task_event to work as expected
        real_job_stage = JobPipelineStage(
            task_manager, artifact_manager, machine, scheduler
        )
        real_job_stage.validate_dependencies = MagicMock(return_value=True)
        real_job_stage.collect_step_handles = MagicMock(return_value={})

        workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
        step_stage = MagicMock(spec=StepPipelineStage)

        # Create a real JobPipelineStage instance, passing in mocks for its
        # dependencies. The scheduler instance itself is passed to handle
        # potential circular dependencies.
        real_job_stage = JobPipelineStage(
            task_manager, artifact_manager, machine, scheduler
        )
        real_job_stage.validate_dependencies = MagicMock(return_value=True)
        real_job_stage.collect_step_handles = MagicMock(return_value={})

        workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
        workpiece_stage.generation_finished = MagicMock()
        workpiece_stage.generation_starting = MagicMock()
        step_stage = MagicMock(spec=StepPipelineStage)
        step_stage.assembly_starting = MagicMock()

        step_stage.validate_dependencies.return_value = True
        step_stage.validate_geometry_match.return_value = True
        workpiece_stage.validate_for_launch.return_value = True
        workpiece_stage.prepare_task_settings.return_value = ({}, {})

        scheduler.set_workpiece_stage(workpiece_stage)
        scheduler.set_step_stage(step_stage)
        scheduler.set_job_stage(real_job_stage)

        return scheduler

    def _create_step_ops_handle(self):
        """Create a mock StepOpsArtifactHandle."""
        handle = MagicMock(spec=StepOpsArtifactHandle)
        handle.to_dict.return_value = {
            "shm_name": "test_shm",
            "handle_class_name": "StepOpsArtifactHandle",
            "artifact_type_name": "StepOpsArtifact",
            "time_estimate": 10.0,
        }
        return handle

    def test_generate_job_no_steps(self):
        """Test that generate_job with no steps calls on_done with None."""
        scheduler = self._make_scheduler()
        callback = MagicMock()
        scheduler.generate_job(step_uids=[], on_done=callback)
        callback.assert_called_once_with(None, None)

    def test_generate_job_missing_step_handle(self):
        """Test that generate_job fails when step handles are missing."""
        scheduler = self._make_scheduler()
        job_stage = cast(MagicMock, scheduler._job_stage)
        job_stage.collect_step_handles.return_value = None

        callback = MagicMock()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], on_done=callback)

        assert callback.called
        args = callback.call_args[0]
        assert args[0] is None
        assert isinstance(args[1], RuntimeError)

    def test_generate_job_triggers_task(self):
        """Test that generate_job launches a task when deps are ready."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        am = scheduler._artifact_manager
        am.get_step_ops_handle.return_value = step_handle  # type: ignore

        callback = MagicMock()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], on_done=callback)

        tm = scheduler._task_manager
        tm.run_process.assert_called_once()  # type: ignore
        call_kwargs = tm.run_process.call_args.kwargs  # type: ignore
        assert "job_description_dict" in call_kwargs
        assert call_kwargs["creator_tag"] == "job"

    def test_generate_job_with_multiple_steps(self):
        """Test that generate_job collects handles from all steps."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        job_stage = cast(MagicMock, scheduler._job_stage)
        job_stage.collect_step_handles.return_value = {
            self.STEP_UID_1: step_handle.to_dict.return_value,
            self.STEP_UID_2: step_handle.to_dict.return_value,
        }

        callback = MagicMock()
        scheduler.generate_job(
            step_uids=[self.STEP_UID_1, self.STEP_UID_2],
            on_done=callback,
        )

        tm = scheduler._task_manager
        tm.run_process.assert_called_once()  # type: ignore
        job_desc = tm.run_process.call_args.kwargs[  # type: ignore
            "job_description_dict"
        ]
        assert len(job_desc["step_artifact_handles_by_uid"]) == 2

    def test_job_generation_finished_signal(self):
        """Test that job_generation_finished signal is emitted."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        am = scheduler._artifact_manager
        am.get_step_ops_handle.return_value = step_handle  # type: ignore
        am.get_job_handle.return_value = MagicMock()  # type: ignore

        mock_signal = MagicMock()
        assert scheduler._job_stage is not None
        scheduler._job_stage.job_generation_finished.connect(mock_signal)

        callback = MagicMock()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], on_done=callback)

        task = MagicMock()
        task.get_status.return_value = "completed"
        task.result.return_value = None

        tm = scheduler._task_manager
        call_kwargs = tm.run_process.call_args.kwargs  # type: ignore
        call_kwargs["when_done"](task)

        mock_signal.assert_called_once()

    def test_job_generation_failed_signal(self):
        """Test that job_generation_failed signal is emitted on failure."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        am = scheduler._artifact_manager
        am.get_step_ops_handle.return_value = step_handle  # type: ignore

        mock_signal = MagicMock()
        assert scheduler._job_stage is not None
        scheduler._job_stage.job_generation_failed.connect(mock_signal)

        callback = MagicMock()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], on_done=callback)

        task = MagicMock()
        task.get_status.return_value = "failed"
        task.result.side_effect = RuntimeError("Task failed")

        tm = scheduler._task_manager
        call_kwargs = tm.run_process.call_args.kwargs  # type: ignore
        call_kwargs["when_done"](task)

        mock_signal.assert_called_once()

    def test_job_task_event_artifact_created(self):
        """Test that artifact_created event commits the job handle."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        am = scheduler._artifact_manager
        am.get_step_ops_handle.return_value = step_handle  # type: ignore

        job_key = ArtifactKey.for_job()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], job_key=job_key)

        job_handle = MagicMock(spec=JobArtifactHandle)
        job_handle.to_dict.return_value = {
            "shm_name": "test_job_shm",
            "handle_class_name": "JobArtifactHandle",
            "artifact_type_name": "JobArtifact",
            "time_estimate": 0,
            "distance": 0,
        }
        am.adopt_artifact.return_value = job_handle  # type: ignore

        ledger_entry = MagicMock()
        ledger_entry.generation_id = 1
        am.get_ledger_entry.return_value = ledger_entry  # type: ignore

        event_data = {
            "handle_dict": job_handle.to_dict.return_value,
            "generation_id": 1,
            "job_key": {"id": job_key.id, "group": job_key.group},
        }

        # This calls the real JobPipelineStage.handle_task_event
        job_stage = cast(JobPipelineStage, scheduler._job_stage)
        job_stage.handle_task_event(
            MagicMock(), "artifact_created", event_data, job_key, 1
        )

        # Assert on the artifact_manager passed to the JobPipelineStage
        am.cache_handle.assert_called_once_with(  # type: ignore
            job_key, job_handle, 1
        )

    def test_job_task_event_ignores_unknown_event(self):
        """Test that unknown events are ignored."""
        scheduler = self._make_scheduler()

        step_handle = self._create_step_ops_handle()
        am = scheduler._artifact_manager
        am.get_step_ops_handle.return_value = step_handle  # type: ignore

        job_key = ArtifactKey.for_job()
        scheduler.generate_job(step_uids=[self.STEP_UID_1], job_key=job_key)

        tm = scheduler._task_manager
        call_kwargs = tm.run_process.call_args.kwargs  # type: ignore
        call_kwargs["when_event"](MagicMock(), "unknown_event", {})

        am.cache_handle.assert_not_called()  # type: ignore


class TestDagSchedulerContextTaskTracking(unittest.TestCase):
    """Tests for Step 4: Task tracking in GenerationContext."""

    STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440101"

    def _make_scheduler(self, artifact_manager=None):
        """Helper to create a scheduler with mocked dependencies."""
        task_manager = MagicMock()
        if artifact_manager is None:
            artifact_manager = MagicMock()
        machine = MagicMock()
        machine.to_dict.return_value = {"name": "test-machine"}
        doc = MagicMock()
        doc.to_dict.return_value = {"layers": []}
        scheduler = DagScheduler(task_manager, artifact_manager, machine)
        scheduler.set_doc(doc)
        scheduler.set_generation_id(1)

        real_job_stage = JobPipelineStage(
            task_manager, artifact_manager, machine, scheduler
        )
        real_job_stage.validate_dependencies = MagicMock(return_value=True)
        real_job_stage.collect_step_handles = MagicMock(return_value={})

        workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
        workpiece_stage.generation_finished = MagicMock()
        workpiece_stage.generation_starting = MagicMock()
        step_stage = MagicMock(spec=StepPipelineStage)
        step_stage.assembly_starting = MagicMock()

        step_stage.validate_dependencies.return_value = True
        step_stage.validate_geometry_match.return_value = True
        workpiece_stage.validate_for_launch.return_value = True
        workpiece_stage.prepare_task_settings.return_value = ({}, {})

        scheduler.set_workpiece_stage(workpiece_stage)
        scheduler.set_step_stage(step_stage)
        scheduler.set_job_stage(real_job_stage)

        return scheduler

    def _create_step_ops_handle(self):
        """Create a mock StepOpsArtifactHandle."""
        handle = MagicMock(spec=StepOpsArtifactHandle)
        handle.to_dict.return_value = {
            "shm_name": "test_shm",
            "handle_class_name": "StepOpsArtifactHandle",
            "artifact_type_name": "StepOpsArtifact",
            "time_estimate": 10.0,
        }
        return handle

    def test_job_task_added_to_context(self):
        """Test that generate_job adds the job key to context.active_tasks."""
        scheduler = self._make_scheduler()
        ctx = GenerationContext(generation_id=1)
        scheduler.set_context(ctx)

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        self.assertEqual(len(ctx.active_tasks), 0)

        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        self.assertEqual(len(ctx.active_tasks), 1)
        job_key = list(ctx.active_tasks)[0]
        self.assertEqual(job_key.group, "job")

    def test_no_context_no_error(self):
        """Test that launching tasks without a context does not raise."""
        scheduler = self._make_scheduler()
        assert scheduler._active_context is None

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        tm = cast(MagicMock, scheduler._task_manager)
        tm.run_process.assert_called_once()

    def test_multiple_tasks_tracked(self):
        """Test that multiple jobs are tracked in context.active_tasks."""
        scheduler = self._make_scheduler()
        ctx = GenerationContext(generation_id=1)
        scheduler.set_context(ctx)

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        scheduler.generate_job(step_uids=[self.STEP_UID_1])
        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        self.assertEqual(len(ctx.active_tasks), 2)


class TestDagSchedulerContextTaskCompletion(unittest.TestCase):
    """Tests for Step 5: Task completion protocol in GenerationContext."""

    STEP_UID_1 = "550e8400-e29b-41d4-a716-446655440101"

    def _make_scheduler(self, artifact_manager=None):
        """Helper to create a scheduler with mocked dependencies."""
        task_manager = MagicMock()
        if artifact_manager is None:
            artifact_manager = MagicMock()
        machine = MagicMock()
        machine.to_dict.return_value = {"name": "test-machine"}
        doc = MagicMock()
        doc.to_dict.return_value = {"layers": []}
        scheduler = DagScheduler(task_manager, artifact_manager, machine)
        scheduler.set_doc(doc)
        scheduler.set_generation_id(1)

        real_job_stage = JobPipelineStage(
            task_manager, artifact_manager, machine, scheduler
        )
        real_job_stage.validate_dependencies = MagicMock(return_value=True)
        real_job_stage.collect_step_handles = MagicMock(return_value={})

        workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
        workpiece_stage.generation_finished = MagicMock()
        workpiece_stage.generation_starting = MagicMock()
        step_stage = MagicMock(spec=StepPipelineStage)
        step_stage.assembly_starting = MagicMock()

        step_stage.validate_dependencies.return_value = True
        step_stage.validate_geometry_match.return_value = True
        workpiece_stage.validate_for_launch.return_value = True
        workpiece_stage.prepare_task_settings.return_value = ({}, {})

        scheduler.set_workpiece_stage(workpiece_stage)
        scheduler.set_step_stage(step_stage)
        scheduler.set_job_stage(real_job_stage)

        return scheduler

    def _create_step_ops_handle(self):
        """Create a mock StepOpsArtifactHandle."""
        handle = MagicMock(spec=StepOpsArtifactHandle)
        handle.to_dict.return_value = {
            "shm_name": "test_shm",
            "handle_class_name": "StepOpsArtifactHandle",
            "artifact_type_name": "StepOpsArtifact",
            "time_estimate": 10.0,
        }
        return handle

    def _create_job_handle(self):
        """Create a mock JobArtifactHandle."""
        handle = MagicMock(spec=JobArtifactHandle)
        handle.to_dict.return_value = {
            "shm_name": "test_shm",
            "handle_class_name": "JobArtifactHandle",
            "artifact_type_name": "JobArtifact",
        }
        return handle

    def test_job_task_did_finish_called_on_success(self):
        """Test that task_did_finish is called when job task completes."""
        scheduler = self._make_scheduler()
        ctx = GenerationContext(generation_id=1)
        scheduler.set_context(ctx)

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        self.assertEqual(len(ctx.active_tasks), 1)

        tm = cast(MagicMock, scheduler._task_manager)
        call_kwargs = tm.run_process.call_args.kwargs
        when_done_cb = call_kwargs["when_done"]

        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = None

        job_handle = self._create_job_handle()
        am.get_job_handle.return_value = job_handle

        when_done_cb(mock_task)

        self.assertEqual(len(ctx.active_tasks), 0)

    def test_job_task_did_finish_called_on_failure(self):
        """Test that task_did_finish is called when job task fails."""
        scheduler = self._make_scheduler()
        ctx = GenerationContext(generation_id=1)
        scheduler.set_context(ctx)

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        self.assertEqual(len(ctx.active_tasks), 1)

        tm = cast(MagicMock, scheduler._task_manager)
        call_kwargs = tm.run_process.call_args.kwargs
        when_done_cb = call_kwargs["when_done"]

        mock_task = MagicMock()
        mock_task.get_status.return_value = "failed"
        mock_task.result.side_effect = RuntimeError("Job failed")

        when_done_cb(mock_task)

        self.assertEqual(len(ctx.active_tasks), 0)

    def test_job_task_did_finish_no_context_no_error(self):
        """Test that task completion without context does not raise."""
        scheduler = self._make_scheduler()
        assert scheduler._active_context is None

        step_handle = self._create_step_ops_handle()
        am = cast(MagicMock, scheduler._artifact_manager)
        am.get_step_ops_handle.return_value = step_handle

        scheduler.generate_job(step_uids=[self.STEP_UID_1])

        tm = cast(MagicMock, scheduler._task_manager)
        call_kwargs = tm.run_process.call_args.kwargs
        when_done_cb = call_kwargs["when_done"]

        mock_task = MagicMock()
        mock_task.get_status.return_value = "completed"
        mock_task.result.return_value = None

        job_handle = self._create_job_handle()
        am.get_job_handle.return_value = job_handle

        try:
            when_done_cb(mock_task)
        except AttributeError:
            self.fail("when_done_cb raised AttributeError with no context")
