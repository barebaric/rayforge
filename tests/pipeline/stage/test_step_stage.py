import pytest
from unittest.mock import MagicMock, PropertyMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.step_stage import StepPipelineStage
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager that captures event callbacks."""
    mock_mgr = MagicMock()
    mock_mgr.created_tasks = []

    class MockTask:
        def __init__(self, target, args, kwargs):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.when_done = kwargs.get("when_done")
            self.when_event = kwargs.get("when_event")
            self.key = kwargs.get("key")
            self.id = id(self)

    def run_process_mock(target_func, *args, **kwargs):
        task = MockTask(target_func, args, kwargs)
        mock_mgr.created_tasks.append(task)
        return task

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    return mock_mgr


@pytest.fixture
def mock_artifact_manager():
    """Provides a mock ArtifactManager."""
    manager = MagicMock()
    manager.get_workpiece_handle.return_value = WorkPieceArtifactHandle(
        is_scalable=True,
        source_coordinate_system_name="MILLIMETER_SPACE",
        source_dimensions=(10, 10),
        generation_size=(10, 10),
        shm_name="dummy_wp_shm",
        handle_class_name="WorkPieceArtifactHandle",
        artifact_type_name="WorkPieceArtifact",
    )
    manager.has_step_render_handle.return_value = False
    return manager


@pytest.fixture
def mock_doc_and_step():
    """Provides a mock Doc object with some structure."""
    doc = MagicMock(spec=Doc)
    layer = MagicMock(spec=Layer)
    # Use MagicMock for Step to allow mocking the read-only 'layer' property
    step = MagicMock(spec=Step)
    step.uid = "step1"
    step.per_step_transformers_dicts = []
    step.visible = True

    # Mock the read-only 'layer' property to return our mock layer
    type(step).layer = PropertyMock(return_value=layer)

    wp_mock = WorkPiece(name="wp1")
    wp_mock.uid = "wp1"
    wp_mock.set_size(10, 10)

    layer.workflow.steps = [step]
    layer.all_workpieces = [wp_mock]
    doc.layers = [layer]
    return doc, step


def _complete_step_task(task, time=42.0, gen_id=1):
    """Helper to simulate the completion of a step assembly task."""
    mock_task_obj = MagicMock()
    mock_task_obj.key = task.key
    mock_task_obj.id = task.id
    mock_task_obj.get_status.return_value = "completed"

    if task.when_event:
        render_handle = StepRenderArtifactHandle(
            shm_name="dummy_render",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
        )
        render_event = {
            "handle_dict": render_handle.to_dict(),
            "generation_id": gen_id,
        }
        task.when_event(mock_task_obj, "render_artifact_ready", render_event)

        ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=time,
        )
        ops_event = {
            "handle_dict": ops_handle.to_dict(),
            "generation_id": gen_id,
        }
        task.when_event(mock_task_obj, "ops_artifact_ready", ops_event)

        time_event = {"time_estimate": time, "generation_id": gen_id}
        task.when_event(mock_task_obj, "time_estimate_ready", time_event)

    mock_task_obj.result.return_value = gen_id
    if task.when_done:
        task.when_done(mock_task_obj)


@pytest.mark.usefixtures("context_initializer")
class TestStepPipelineStage:
    def test_instantiation(self, mock_task_mgr, mock_artifact_manager):
        """Test that StepPipelineStage can be created."""
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is mock_artifact_manager

    def test_reconcile_triggers_assembly_for_missing_artifact(
        self, mock_task_mgr, mock_artifact_manager, mock_doc_and_step
    ):
        """
        Tests that reconcile() starts a task if a step artifact is missing.
        """
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager)

        # Act
        stage.reconcile(doc)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_step_artifact_in_subprocess

    def test_mark_stale_and_trigger_starts_assembly(
        self, mock_task_mgr, mock_artifact_manager, mock_doc_and_step
    ):
        """Tests that explicitly marking a step as stale triggers assembly."""
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager)

        # Act
        stage.mark_stale_and_trigger(step)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        assert len(mock_task_mgr.created_tasks) == 1

    def test_assembly_flow_success(
        self, mock_task_mgr, mock_artifact_manager, mock_doc_and_step
    ):
        """
        Tests the full successful flow: triggering, receiving the render
        event, and then receiving the final time estimate.
        """
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager)

        render_signal_handler = MagicMock()
        time_signal_handler = MagicMock()
        stage.render_artifact_ready.connect(render_signal_handler)
        stage.time_estimate_ready.connect(time_signal_handler)

        # Act
        stage.mark_stale_and_trigger(step)

        # Assert a task was created
        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[0]

        # Simulate Phase 1: Artifact and Time Events
        mock_task_obj = MagicMock()
        mock_task_obj.id = task.id
        gen_id = 1
        render_handle = StepRenderArtifactHandle(
            shm_name="render_shm",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
        )
        mock_artifact_manager.adopt_artifact.return_value = render_handle
        task.when_event(
            mock_task_obj,
            "render_artifact_ready",
            {"handle_dict": render_handle.to_dict(), "generation_id": gen_id},
        )

        ops_handle = StepOpsArtifactHandle(
            shm_name="ops_shm",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=None,
        )
        mock_artifact_manager.adopt_artifact.return_value = ops_handle
        task.when_event(
            mock_task_obj,
            "ops_artifact_ready",
            {"handle_dict": ops_handle.to_dict(), "generation_id": gen_id},
        )

        task.when_event(
            mock_task_obj,
            "time_estimate_ready",
            {"time_estimate": 42.5, "generation_id": gen_id},
        )

        # Assert event phase worked
        mock_artifact_manager.put_step_render_handle.assert_called_once_with(
            step.uid, render_handle
        )
        mock_artifact_manager.put_step_ops_handle.assert_called_once_with(
            step.uid, ops_handle
        )
        render_signal_handler.assert_called_once_with(stage, step=step)
        time_signal_handler.assert_called_once_with(
            stage, step=step, time=42.5
        )
        assert stage.get_estimate(step.uid) == 42.5

        # Simulate Phase 2: Task Completion
        mock_task_obj.get_status.return_value = "completed"
        mock_task_obj.result.return_value = gen_id
        task.when_done(mock_task_obj)

        # No new signals should fire, just cleanup
        time_signal_handler.assert_called_once()

    def test_invalidate_cleans_up_and_invalidates_job(
        self, mock_task_mgr, mock_artifact_manager, mock_doc_and_step
    ):
        """Tests that invalidating a step cleans up all its artifacts."""
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager)
        stage.mark_stale_and_trigger(step)
        _complete_step_task(mock_task_mgr.created_tasks[0])
        mock_task_mgr.created_tasks.clear()

        # Act
        stage.invalidate(step.uid)

        # Assert
        mock_artifact_manager.pop_step_ops_handle.assert_called_with(step.uid)
        mock_artifact_manager.pop_step_render_handle.assert_called_with(
            step.uid
        )
        mock_artifact_manager.invalidate_for_job.assert_called()
