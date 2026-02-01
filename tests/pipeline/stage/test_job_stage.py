import pytest
from unittest.mock import MagicMock, ANY

from rayforge.core.doc import Doc
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.job_stage import JobPipelineStage, JobKey
from rayforge.pipeline.stage.job_runner import (
    make_job_artifact_in_subprocess,
)
from rayforge.pipeline.artifact import JobArtifactHandle, StepOpsArtifactHandle
from rayforge.pipeline.artifact.cache import ArtifactCache
from rayforge.context import get_context
from rayforge.pipeline.steps import create_contour_step


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
def artifact_cache():
    """Provides a real ArtifactCache instance for testing."""
    cache = ArtifactCache()
    yield cache
    cache.shutdown()


@pytest.fixture
def real_doc_with_step(context_initializer, test_machine_and_config):
    """
    Provides a real Doc object with a layer and a step, with the context
    correctly configured with a test machine.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None
    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)
    return doc


@pytest.mark.usefixtures("context_initializer")
class TestJobPipelineStage:
    def test_instantiation(self, mock_task_mgr, artifact_cache):
        """Test that JobPipelineStage can be created."""
        stage = JobPipelineStage(mock_task_mgr, artifact_cache)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is artifact_cache

    def test_generate_job_triggers_task(
        self, mock_task_mgr, artifact_cache, real_doc_with_step
    ):
        """Test that generate_job starts a task if steps are available."""
        # Pre-populate the cache with a required StepOps handle for the job
        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_cache.put_step_ops_handle(step.uid, step_ops_handle)

        stage = JobPipelineStage(mock_task_mgr, artifact_cache)
        stage.generate_job(real_doc_with_step)

        mock_task_mgr.run_process.assert_called_once()
        call_args = mock_task_mgr.run_process.call_args
        assert call_args[0][0] is make_job_artifact_in_subprocess
        assert call_args.kwargs["key"] == JobKey

    def test_event_and_completion_flow(
        self,
        mock_task_mgr,
        artifact_cache,
        real_doc_with_step,
        monkeypatch,
    ):
        """Tests the full event->completion flow for job generation."""
        # Arrange
        mock_store = MagicMock()
        monkeypatch.setattr(get_context(), "artifact_store", mock_store)

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_cache.put_step_ops_handle(step.uid, step_ops_handle)

        stage = JobPipelineStage(mock_task_mgr, artifact_cache)
        mock_finished_signal = MagicMock()
        stage.generation_finished.connect(mock_finished_signal)
        stage.generate_job(real_doc_with_step)

        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[0]

        # Use the actual task object and patch methods onto it, because
        # the stage checks identity (self._active_task is task).
        task.get_status = MagicMock(return_value="completed")
        task.result = MagicMock(return_value=None)

        # 1. Simulate the event
        handle = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=123.0,
            distance=456.0,
        )
        event_data = {"handle_dict": handle.to_dict()}

        # Pass the task object itself to satisfy identity check
        task.when_event(task, "artifact_created", event_data)

        # Assert event was processed correctly
        mock_store.adopt.assert_called_once_with(handle)
        assert artifact_cache.get_job_handle() == handle

        # 2. Simulate task completion
        task.when_done(task)

        # Assert completion was processed
        mock_finished_signal.assert_called_once_with(
            stage, handle=handle, task_status="completed"
        )

    def test_completion_with_failure(
        self,
        mock_task_mgr,
        artifact_cache,
        real_doc_with_step,
        monkeypatch,
    ):
        """Tests that a failed task correctly cleans up the cached handle."""
        # Arrange
        mock_store = MagicMock()
        monkeypatch.setattr(get_context(), "artifact_store", mock_store)

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_cache.put_step_ops_handle(step.uid, step_ops_handle)

        stage = JobPipelineStage(mock_task_mgr, artifact_cache)
        mock_failed_signal = MagicMock()
        stage.generation_failed.connect(mock_failed_signal)
        stage.generate_job(real_doc_with_step)

        task = mock_task_mgr.created_tasks[0]

        # Patch methods onto the actual task object
        task.get_status = MagicMock(return_value="failed")
        task.result = MagicMock(side_effect=RuntimeError("Task failed"))

        # Simulate event, which places a handle in the cache
        handle = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )
        # Pass the task object itself
        task.when_event(
            task,
            "artifact_created",
            {"handle_dict": handle.to_dict()},
        )

        # Assert handle was cached
        assert artifact_cache.get_job_handle() == handle

        # Act
        task.when_done(task)

        # Assert cleanup
        assert artifact_cache.get_job_handle() is None
        mock_failed_signal.assert_called_once_with(
            stage, task_status="failed", error=ANY
        )
