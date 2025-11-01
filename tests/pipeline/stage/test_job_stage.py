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
        self, mock_task_mgr, artifact_cache, real_doc_with_step, monkeypatch
    ):
        """Test that generate_job starts a task if steps are available."""
        # Arrange: Mock the artifact store to allow acquisition
        mock_store = MagicMock()
        mock_store.acquire.return_value = True
        monkeypatch.setattr(
            "rayforge.pipeline.stage.job_stage.get_context",
            lambda: MagicMock(artifact_store=mock_store),
        )

        # Pre-populate the cache with a required StepOps handle
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

        # Assert
        mock_store.acquire.assert_called_once_with(step_ops_handle)
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
        mock_store.acquire.return_value = True
        # Patch the get_context() call inside the stage module
        monkeypatch.setattr(
            "rayforge.pipeline.stage.job_stage.get_context",
            lambda: MagicMock(artifact_store=mock_store),
        )

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

        # Act 1: Generate the job
        stage.generate_job(real_doc_with_step)

        # Assert 1: Task was created and dependency acquired
        mock_store.acquire.assert_called_once_with(step_ops_handle)
        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[0]

        mock_task_obj = MagicMock()
        mock_task_obj.key = task.key
        mock_task_obj.get_status.return_value = "completed"
        mock_task_obj.result.return_value = None  # Runner returns None

        # Act 2: Simulate the event from the runner
        handle = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=123.0,
            distance=456.0,
        )
        event_data = {"handle_dict": handle.to_dict()}
        task.when_event(mock_task_obj, "artifact_created", event_data)

        # Assert 2: Event was processed correctly
        mock_store.adopt.assert_called_once_with(handle)
        assert artifact_cache.get_job_handle() == handle

        # Act 3: Simulate task completion
        task.when_done(mock_task_obj)

        # Assert 3: Completion was processed and dependency released
        mock_finished_signal.assert_called_once_with(
            stage, handle=handle, task_status="completed"
        )
        mock_store.release.assert_called_once_with(step_ops_handle)

    def test_completion_with_failure(
        self,
        mock_task_mgr,
        artifact_cache,
        real_doc_with_step,
        monkeypatch,
    ):
        """
        Tests that a failed task correctly cleans up and releases handles.
        """
        # Arrange
        mock_store = MagicMock()
        mock_store.acquire.return_value = True
        monkeypatch.setattr(
            "rayforge.pipeline.stage.job_stage.get_context",
            lambda: MagicMock(artifact_store=mock_store),
        )

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
        mock_task_obj = MagicMock()
        mock_task_obj.key = task.key
        mock_task_obj.get_status.return_value = "failed"
        mock_task_obj.result.side_effect = RuntimeError("Task failed")

        # Act
        task.when_done(mock_task_obj)

        # Assert cleanup
        assert artifact_cache.get_job_handle() is None
        mock_failed_signal.assert_called_once_with(
            stage, task_status="failed", error=ANY
        )
        # Assert dependency handle was still released even on failure
        mock_store.release.assert_called_once_with(step_ops_handle)
