from typing import cast
from unittest.mock import MagicMock
from contextlib import contextmanager

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


def create_node_with_state(
    key, state, generation_id=1, manager=None
) -> ArtifactNode:
    """
    Helper to create a node with a mock manager returning the given state.
    """
    if manager is None:
        manager = MagicMock()
        manager.get_state.return_value = state
    return ArtifactNode(
        key=key,
        generation_id=generation_id,
        _artifact_manager=manager,
    )


def _make_scheduler():
    """Helper to create a scheduler with mocked dependencies."""
    task_manager = MagicMock()
    artifact_manager = MagicMock()
    machine = MagicMock()
    scheduler = DagScheduler(task_manager, artifact_manager, machine)

    workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
    workpiece_stage.generation_finished = MagicMock()
    workpiece_stage.generation_starting = MagicMock()
    workpiece_stage.node_state_changed = MagicMock()
    step_stage = MagicMock(spec=StepPipelineStage)
    step_stage.assembly_starting = MagicMock()
    step_stage.node_state_changed = MagicMock()
    job_stage = MagicMock(spec=JobPipelineStage)
    job_stage.job_generation_finished = MagicMock()
    job_stage.job_generation_failed = MagicMock()
    job_stage.node_state_changed = MagicMock()

    step_stage.validate_dependencies.return_value = True
    step_stage.validate_geometry_match.return_value = True
    job_stage.validate_dependencies.return_value = True
    workpiece_stage.validate_for_launch.return_value = True
    workpiece_stage.prepare_task_settings.return_value = ({}, {})

    scheduler.set_workpiece_stage(workpiece_stage)
    scheduler.set_step_stage(step_stage)
    scheduler.set_job_stage(job_stage)

    return scheduler


def test_scheduler_initialization():
    """Test creating a scheduler."""
    scheduler = _make_scheduler()
    assert scheduler.graph is not None
    assert len(scheduler.graph.get_all_nodes()) == 0


def test_get_ready_nodes_empty_graph():
    """Test getting ready nodes from empty graph."""
    scheduler = _make_scheduler()
    ready = scheduler.get_ready_nodes("workpiece")
    assert len(ready) == 0


def test_get_ready_nodes_filters_by_group():
    """Test that get_ready_nodes filters by group."""
    scheduler = _make_scheduler()

    wp_key = ArtifactKey.for_workpiece(WP_UID_1)
    step_key = ArtifactKey.for_step(STEP_UID_1)

    wp_node = create_node_with_state(wp_key, NodeState.DIRTY)
    step_node = create_node_with_state(step_key, NodeState.DIRTY)

    scheduler.graph.add_node(wp_node)
    scheduler.graph.add_node(step_node)

    wp_ready = scheduler.get_ready_nodes("workpiece")
    step_ready = scheduler.get_ready_nodes("step")

    assert len(wp_ready) == 1
    assert len(step_ready) == 1
    assert wp_ready[0] == wp_key
    assert step_ready[0] == step_key


def test_get_ready_nodes_excludes_valid_nodes():
    """Test that VALID nodes are not included in ready list."""
    scheduler = _make_scheduler()

    wp_key = ArtifactKey.for_workpiece(WP_UID_1)
    wp_node = create_node_with_state(wp_key, NodeState.VALID)

    scheduler.graph.add_node(wp_node)

    ready = scheduler.get_ready_nodes("workpiece")
    assert len(ready) == 0


def test_get_ready_nodes_excludes_nodes_with_dirty_deps():
    """Test that nodes with DIRTY dependencies are not ready."""
    scheduler = _make_scheduler()

    wp_key = ArtifactKey.for_workpiece(WP_UID_1)
    step_key = ArtifactKey.for_step(STEP_UID_1)

    wp_node = create_node_with_state(wp_key, NodeState.DIRTY)
    step_node = create_node_with_state(step_key, NodeState.DIRTY)

    step_node.add_dependency(wp_node)

    scheduler.graph.add_node(wp_node)
    scheduler.graph.add_node(step_node)

    ready = scheduler.get_ready_nodes("step")
    assert len(ready) == 0


def test_find_node():
    """Test finding a node by key."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    node = ArtifactNode(key=key)

    scheduler.graph.add_node(node)

    found = scheduler.find_node(key)
    assert found is node


def test_find_node_not_found():
    """Test finding a non-existent node."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
    found = scheduler.find_node(key)

    assert found is None


def test_set_node_state():
    """Test setting the state of a node."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.DIRTY
    node = create_node_with_state(key, NodeState.DIRTY, manager=manager)

    scheduler.graph.add_node(node)

    result = scheduler.set_node_state(key, NodeState.VALID)

    assert result is True
    manager.set_state.assert_called_once_with(key, 1, NodeState.VALID)


def test_set_node_state_not_found():
    """Test setting state on non-existent node."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
    result = scheduler.set_node_state(key, NodeState.VALID)

    assert result is False


def test_mark_node_dirty():
    """Test marking a node dirty."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.VALID
    node = create_node_with_state(key, NodeState.VALID, manager=manager)

    scheduler.graph.add_node(node)

    result = scheduler.mark_node_dirty(key)

    assert result is True
    manager.set_state.assert_called_with(key, 1, NodeState.DIRTY)


def test_mark_node_dirty_propagates():
    """Test that marking a node dirty propagates to dependents."""
    scheduler = _make_scheduler()

    wp_key = ArtifactKey.for_workpiece(WP_UID_1)
    step_key = ArtifactKey.for_step(STEP_UID_1)

    wp_manager = MagicMock()
    wp_manager.get_state.return_value = NodeState.VALID
    step_manager = MagicMock()
    step_manager.get_state.return_value = NodeState.VALID

    wp_node = create_node_with_state(
        wp_key, NodeState.VALID, manager=wp_manager
    )
    step_node = create_node_with_state(
        step_key, NodeState.VALID, manager=step_manager
    )

    step_node.add_dependency(wp_node)

    scheduler.graph.add_node(wp_node)
    scheduler.graph.add_node(step_node)

    scheduler.mark_node_dirty(wp_key)

    wp_manager.set_state.assert_called_with(wp_key, 1, NodeState.DIRTY)
    step_manager.set_state.assert_called_with(step_key, 1, NodeState.DIRTY)


def test_mark_node_dirty_not_found():
    """Test marking a non-existent node dirty."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece("550e8400-e29b-41d4-a716-446655449999")
    result = scheduler.mark_node_dirty(key)

    assert result is False


def test_process_graph():
    """Test that process_graph can be called without error."""
    scheduler = _make_scheduler()
    scheduler.process_graph()


def test_on_artifact_state_changed_valid():
    """Test that callback updates node to VALID."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.DIRTY
    node = create_node_with_state(key, NodeState.DIRTY, manager=manager)
    scheduler.graph.add_node(node)

    scheduler.on_artifact_state_changed(key, "valid")

    manager.set_state.assert_called_once_with(key, 1, NodeState.VALID)


def test_on_artifact_state_changed_processing():
    """Test that callback updates node to PROCESSING."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.DIRTY
    node = create_node_with_state(key, NodeState.DIRTY, manager=manager)
    scheduler.graph.add_node(node)

    scheduler.on_artifact_state_changed(key, "processing")

    manager.set_state.assert_called_once_with(key, 1, NodeState.PROCESSING)


def test_on_artifact_state_changed_error():
    """Test that callback updates node to ERROR."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.DIRTY
    node = create_node_with_state(key, NodeState.DIRTY, manager=manager)
    scheduler.graph.add_node(node)

    scheduler.on_artifact_state_changed(key, "error")

    manager.set_state.assert_called_once_with(key, 1, NodeState.ERROR)


def test_on_artifact_state_changed_unknown_state():
    """Test that callback handles unknown state gracefully."""
    scheduler = _make_scheduler()

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager = MagicMock()
    manager.get_state.return_value = NodeState.DIRTY
    node = create_node_with_state(key, NodeState.DIRTY, manager=manager)
    scheduler.graph.add_node(node)

    scheduler.on_artifact_state_changed(key, "unknown")

    manager.set_state.assert_not_called()


def test_artifact_manager_syncs_dag_state():
    """Test that DAG manages node state via ArtifactManager."""
    mock_store = MagicMock()
    mock_store._refcounts = {}
    scheduler = _make_scheduler()
    manager = ArtifactManager(mock_store)

    key = ArtifactKey.for_workpiece(WP_UID_1)
    manager.declare_generation({key}, 1)
    node = ArtifactNode(
        key=key,
        generation_id=1,
        _artifact_manager=manager,
    )
    scheduler.graph.add_node(node)

    assert node.state == NodeState.DIRTY

    node.state = NodeState.PROCESSING
    assert node.state == NodeState.PROCESSING

    mock_handle = MagicMock(spec=BaseArtifactHandle)
    mock_handle.shm_name = "test_shm"
    manager.cache_handle(key, mock_handle, 1)
    assert node.state == NodeState.VALID


STEP_UID_JOB_1 = "550e8400-e29b-41d4-a716-446655440101"
STEP_UID_JOB_2 = "550e8400-e29b-41d4-a716-446655440102"


def _make_scheduler_for_job(artifact_manager=None):
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

    real_job_stage = JobPipelineStage(task_manager, artifact_manager, machine)
    real_job_stage.validate_dependencies = MagicMock(return_value=True)
    real_job_stage.collect_step_handles = MagicMock(return_value={})

    workpiece_stage = MagicMock(spec=WorkPiecePipelineStage)
    workpiece_stage.generation_finished = MagicMock()
    workpiece_stage.generation_starting = MagicMock()
    workpiece_stage.node_state_changed = MagicMock()
    step_stage = MagicMock(spec=StepPipelineStage)
    step_stage.assembly_starting = MagicMock()
    step_stage.node_state_changed = MagicMock()

    step_stage.validate_dependencies.return_value = True
    step_stage.validate_geometry_match.return_value = True
    workpiece_stage.validate_for_launch.return_value = True
    workpiece_stage.prepare_task_settings.return_value = ({}, {})

    scheduler.set_workpiece_stage(workpiece_stage)
    scheduler.set_step_stage(step_stage)
    scheduler.set_job_stage(real_job_stage)

    return scheduler


def _create_step_ops_handle():
    """Create a mock StepOpsArtifactHandle."""
    handle = MagicMock(spec=StepOpsArtifactHandle)
    handle.to_dict.return_value = {
        "shm_name": "test_shm",
        "handle_class_name": "StepOpsArtifactHandle",
        "artifact_type_name": "StepOpsArtifact",
        "time_estimate": 10.0,
    }
    return handle


def test_generate_job_no_steps():
    """Test that generate_job with no steps calls on_done with None."""
    scheduler = _make_scheduler_for_job()
    callback = MagicMock()
    scheduler.generate_job(step_uids=[], on_done=callback)
    callback.assert_called_once_with(None, None)


def test_generate_job_missing_step_handle():
    """Test that generate_job fails when step handles are missing."""
    scheduler = _make_scheduler_for_job()
    job_stage = cast(MagicMock, scheduler._job_stage)
    job_stage.collect_step_handles.return_value = None

    callback = MagicMock()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], on_done=callback)

    assert callback.called
    args = callback.call_args[0]
    assert args[0] is None
    assert isinstance(args[1], RuntimeError)


def test_generate_job_triggers_task():
    """Test that generate_job launches a task when deps are ready."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    callback = MagicMock()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], on_done=callback)

    tm = cast(MagicMock, scheduler._task_manager)
    tm.run_process.assert_called_once()
    call_kwargs = tm.run_process.call_args.kwargs
    assert "job_description_dict" in call_kwargs
    assert call_kwargs["creator_tag"] == "job"


def test_generate_job_with_multiple_steps():
    """Test that generate_job collects handles from all steps."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    job_stage = cast(MagicMock, scheduler._job_stage)
    job_stage.collect_step_handles.return_value = {
        STEP_UID_JOB_1: step_handle.to_dict.return_value,
        STEP_UID_JOB_2: step_handle.to_dict.return_value,
    }

    callback = MagicMock()
    scheduler.generate_job(
        step_uids=[STEP_UID_JOB_1, STEP_UID_JOB_2],
        on_done=callback,
    )

    tm = cast(MagicMock, scheduler._task_manager)
    tm.run_process.assert_called_once()
    job_desc = tm.run_process.call_args.kwargs["job_description_dict"]
    assert len(job_desc["step_artifact_handles_by_uid"]) == 2


def test_job_generation_finished_signal():
    """Test that job_generation_finished signal is emitted."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle
    am.get_job_handle.return_value = MagicMock()

    mock_signal = MagicMock()
    assert scheduler._job_stage is not None
    scheduler._job_stage.job_generation_finished.connect(mock_signal)

    callback = MagicMock()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], on_done=callback)

    task = MagicMock()
    task.get_status.return_value = "completed"
    task.result.return_value = None

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    call_kwargs["when_done"](task)

    mock_signal.assert_called_once()


def test_job_generation_failed_signal():
    """Test that job_generation_failed signal is emitted on failure."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    mock_signal = MagicMock()
    assert scheduler._job_stage is not None
    scheduler._job_stage.job_generation_failed.connect(mock_signal)

    callback = MagicMock()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], on_done=callback)

    task = MagicMock()
    task.get_status.return_value = "failed"
    task.result.side_effect = RuntimeError("Task failed")

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    call_kwargs["when_done"](task)

    mock_signal.assert_called_once()


def test_job_task_event_artifact_created():
    """Test that artifact_created event commits the job handle."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    job_key = ArtifactKey.for_job()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], job_key=job_key)

    job_handle = MagicMock(spec=JobArtifactHandle)
    job_handle.to_dict.return_value = {
        "shm_name": "test_job_shm",
        "handle_class_name": "JobArtifactHandle",
        "artifact_type_name": "JobArtifact",
        "time_estimate": 0,
        "distance": 0,
    }

    # Mock the safe_adoption context manager
    @contextmanager
    def mock_safe_adoption(key, handle_dict):
        yield job_handle

    am.safe_adoption.side_effect = mock_safe_adoption

    ledger_entry = MagicMock()
    ledger_entry.generation_id = 1
    am.get_ledger_entry.return_value = ledger_entry

    event_data = {
        "handle_dict": job_handle.to_dict.return_value,
        "generation_id": 1,
        "job_key": {"id": job_key.id, "group": job_key.group},
    }

    job_stage = cast(JobPipelineStage, scheduler._job_stage)
    job_stage.handle_task_event(
        MagicMock(), "artifact_created", event_data, job_key, 1
    )

    am.cache_handle.assert_called_once_with(job_key, job_handle, 1)


def test_job_task_event_ignores_unknown_event():
    """Test that unknown events are ignored."""
    scheduler = _make_scheduler_for_job()

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    job_key = ArtifactKey.for_job()
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1], job_key=job_key)

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    call_kwargs["when_event"](MagicMock(), "unknown_event", {})

    am.cache_handle.assert_not_called()


def test_job_task_added_to_context():
    """Test that generate_job adds the job key to context.active_tasks."""
    scheduler = _make_scheduler_for_job()
    ctx = GenerationContext(generation_id=1)
    scheduler.set_context(ctx)

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    assert len(ctx.active_tasks) == 0

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    assert len(ctx.active_tasks) == 1
    job_key = list(ctx.active_tasks)[0]
    assert job_key.group == "job"


def test_no_context_no_error():
    """Test that launching tasks without a context does not raise."""
    scheduler = _make_scheduler_for_job()
    assert scheduler._active_context is None

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    tm = cast(MagicMock, scheduler._task_manager)
    tm.run_process.assert_called_once()


def test_multiple_tasks_tracked():
    """Test that multiple jobs are tracked in context.active_tasks."""
    scheduler = _make_scheduler_for_job()
    ctx = GenerationContext(generation_id=1)
    scheduler.set_context(ctx)

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])
    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    assert len(ctx.active_tasks) == 2


def _create_job_handle():
    """Create a mock JobArtifactHandle."""
    handle = MagicMock(spec=JobArtifactHandle)
    handle.to_dict.return_value = {
        "shm_name": "test_shm",
        "handle_class_name": "JobArtifactHandle",
        "artifact_type_name": "JobArtifact",
    }
    return handle


def test_job_task_did_finish_called_on_success():
    """Test that task_did_finish is called when job task completes."""
    scheduler = _make_scheduler_for_job()
    ctx = GenerationContext(generation_id=1)
    scheduler.set_context(ctx)

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    assert len(ctx.active_tasks) == 1

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    when_done_cb = call_kwargs["when_done"]

    mock_task = MagicMock()
    mock_task.get_status.return_value = "completed"
    mock_task.result.return_value = None

    job_handle = _create_job_handle()
    am.get_job_handle.return_value = job_handle

    when_done_cb(mock_task)

    assert len(ctx.active_tasks) == 0


def test_job_task_did_finish_called_on_failure():
    """Test that task_did_finish is called when job task fails."""
    scheduler = _make_scheduler_for_job()
    ctx = GenerationContext(generation_id=1)
    scheduler.set_context(ctx)

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    assert len(ctx.active_tasks) == 1

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    when_done_cb = call_kwargs["when_done"]

    mock_task = MagicMock()
    mock_task.get_status.return_value = "failed"
    mock_task.result.side_effect = RuntimeError("Job failed")

    when_done_cb(mock_task)

    assert len(ctx.active_tasks) == 0


def test_job_task_did_finish_no_context_no_error():
    """Test that task completion without context does not raise."""
    scheduler = _make_scheduler_for_job()
    assert scheduler._active_context is None

    step_handle = _create_step_ops_handle()
    am = cast(MagicMock, scheduler._artifact_manager)
    am.get_step_ops_handle.return_value = step_handle

    scheduler.generate_job(step_uids=[STEP_UID_JOB_1])

    tm = cast(MagicMock, scheduler._task_manager)
    call_kwargs = tm.run_process.call_args.kwargs
    when_done_cb = call_kwargs["when_done"]

    mock_task = MagicMock()
    mock_task.get_status.return_value = "completed"
    mock_task.result.return_value = None

    job_handle = _create_job_handle()
    am.get_job_handle.return_value = job_handle

    try:
        when_done_cb(mock_task)
    except AttributeError:
        assert False, "when_done_cb raised AttributeError with no context"
