import pytest
from unittest.mock import MagicMock, ANY

from rayforge.core.doc import Doc
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.job_stage import JobPipelineStage
from rayforge.pipeline.stage.job_runner import (
    make_job_artifact_in_subprocess,
)
from rayforge.pipeline.artifact import (
    JobArtifactHandle,
    ArtifactManager,
    StepOpsArtifactHandle,
)
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
def artifact_manager():
    """Provides a real ArtifactManager instance for testing."""
    from rayforge.pipeline.artifact.store import ArtifactStore

    mock_store = MagicMock(spec=ArtifactStore)
    manager = ArtifactManager(mock_store)
    yield manager
    manager.shutdown()


@pytest.fixture
def real_doc_with_step(context_initializer, test_machine_and_config):
    """
    Provides a real Doc object with a layer and a step, with
    context correctly configured with a test machine.
    """
    doc = Doc()
    layer = doc.active_layer
    assert layer.workflow is not None
    step = create_contour_step(context_initializer)
    layer.workflow.add_step(step)
    return doc


@pytest.mark.usefixtures("context_initializer")
class TestJobPipelineStage:
    def test_instantiation(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test that JobPipelineStage can be created."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is artifact_manager
        assert stage._machine is machine

    def test_generate_job_triggers_task(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test that generate_job starts a task if steps are available."""
        machine, _ = test_machine_and_config
        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_manager.put_step_ops_handle(step.uid, step_ops_handle)

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        stage.generate_job(real_doc_with_step)

        mock_task_mgr.run_process.assert_called_once()
        call_args = mock_task_mgr.run_process.call_args
        assert call_args[0][0] is make_job_artifact_in_subprocess

        # Verify artifact_store is passed as second positional argument
        assert call_args[0][1] is artifact_manager._store

        # Access the nested job_description_dict
        job_desc = call_args.kwargs["job_description_dict"]
        assert job_desc["step_artifact_handles_by_uid"] == {
            step.uid: step_ops_handle.to_dict()
        }

    def test_event_and_completion_flow(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Tests that full event->completion flow for job generation."""
        machine, _ = test_machine_and_config
        handle = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        mock_finished_signal = MagicMock()
        stage.generation_finished.connect(mock_finished_signal)
        stage.generate_job(real_doc_with_step)

        task = mock_task_mgr.created_tasks[0]

        task.get_status = MagicMock(return_value="completed")
        task.result = MagicMock(return_value=None)

        event_data = {"handle_dict": handle.to_dict()}

        # Directly monkeypatch the method on the instance
        artifact_manager.adopt_artifact = MagicMock(return_value=handle)

        task.when_event(task, "artifact_created", event_data)

        # Assert handle was cached after adopt_artifact
        assert artifact_manager.get_job_handle() == handle

        task.when_done(task)

        # Signal is emitted when task completes
        mock_finished_signal.assert_called_once_with(
            stage, handle=handle, task_status="completed"
        )

        # The handle SHOULD persist on success
        assert artifact_manager.get_job_handle() == handle

    def test_completion_with_failure(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Tests that a failed task correctly cleans up the cached handle."""
        machine, _ = test_machine_and_config
        handle = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        mock_failed_signal = MagicMock()
        stage.generation_failed.connect(mock_failed_signal)
        stage.generate_job(real_doc_with_step)

        task = mock_task_mgr.created_tasks[0]

        task.get_status = MagicMock(return_value="failed")
        task.result = MagicMock(side_effect=RuntimeError("Task failed"))

        event_data = {"handle_dict": handle.to_dict()}

        # Directly monkeypatch the method on the instance
        artifact_manager.adopt_artifact = MagicMock(return_value=handle)

        task.when_event(task, "artifact_created", event_data)

        # Assert handle was cached after adopt_artifact (simulating it was set
        # during event)
        task.when_done(task)

        # Signal is emitted when task fails
        mock_failed_signal.assert_called_once_with(
            stage, error=ANY, task_status="failed"
        )

        # On failure, the handle MUST be cleared
        assert artifact_manager.get_job_handle() is None
