import pytest
from unittest.mock import MagicMock, ANY

from rayforge.core.doc import Doc
from rayforge.pipeline.artifact.lifecycle import (
    ArtifactLifecycle,
    LedgerEntry,
)
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.job_stage import JobPipelineStage
from rayforge.pipeline.stage.job_runner import (
    make_job_artifact_in_subprocess,
    JobDescription,
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

        artifact_manager._ledger[artifact_manager.JOB_KEY] = LedgerEntry(
            state=ArtifactLifecycle.MISSING,
        )

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        job_desc = JobDescription(
            step_artifact_handles_by_uid={step.uid: step_ops_handle.to_dict()},
            machine_dict=machine.to_dict(),
            doc_dict=real_doc_with_step.to_dict(),
        )

        stage.generate_job(job_desc)

        mock_task_mgr.run_process.assert_called_once()
        call_args = mock_task_mgr.run_process.call_args
        assert call_args[0][0] is make_job_artifact_in_subprocess

        # Verify artifact_store is passed as second positional argument
        assert call_args[0][1] is artifact_manager._store

        # Access the nested job_description_dict
        job_desc_dict = call_args.kwargs["job_description_dict"]
        assert job_desc_dict == job_desc.__dict__
        assert job_desc_dict["step_artifact_handles_by_uid"] == {
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

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )

        artifact_manager._ledger[artifact_manager.JOB_KEY] = LedgerEntry(
            state=ArtifactLifecycle.MISSING,
        )

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        mock_finished_signal = MagicMock()
        stage.generation_finished.connect(mock_finished_signal)

        job_desc = JobDescription(
            step_artifact_handles_by_uid={step.uid: step_ops_handle.to_dict()},
            machine_dict=machine.to_dict(),
            doc_dict=real_doc_with_step.to_dict(),
        )

        stage.generate_job(job_desc)

        task = mock_task_mgr.created_tasks[0]

        task.get_status = MagicMock(return_value="completed")
        task.result = MagicMock(return_value=None)

        event_data = {
            "handle_dict": handle.to_dict(),
            "generation_id": 123,
        }

        # Directly monkeypatch the method on the instance
        artifact_manager.adopt_artifact = MagicMock(return_value=handle)
        artifact_manager.commit = MagicMock()

        task.when_event(task, "artifact_created", event_data)

        # Manually set the job handle to simulate commit behavior
        def commit_side_effect(key, h, gen_id):
            artifact_manager._ledger[key].handle = h
            artifact_manager._ledger[key].state = ArtifactLifecycle.READY

        # Directly monkeypatch the method on the instance
        artifact_manager.adopt_artifact = MagicMock(return_value=handle)
        artifact_manager.commit = MagicMock(side_effect=commit_side_effect)

        task.when_event(task, "artifact_created", event_data)

        # Assert commit was called with correct params
        artifact_manager.commit.assert_called_once_with(
            artifact_manager.JOB_KEY, handle, 123
        )

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
        _ = JobArtifactHandle(
            shm_name="dummy_job_shm",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )

        artifact_manager._ledger[artifact_manager.JOB_KEY] = LedgerEntry(
            state=ArtifactLifecycle.MISSING,
        )

        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)
        mock_failed_signal = MagicMock()
        stage.generation_failed.connect(mock_failed_signal)

        job_desc = JobDescription(
            step_artifact_handles_by_uid={step.uid: step_ops_handle.to_dict()},
            machine_dict=machine.to_dict(),
            doc_dict=real_doc_with_step.to_dict(),
        )

        stage.generate_job(job_desc)

        task = mock_task_mgr.created_tasks[0]

        task.get_status = MagicMock(return_value="failed")
        task.result = MagicMock(side_effect=RuntimeError("Task failed"))

        # For failure case, the artifact_created event should not be called
        # since the task failed before creating the artifact
        task.when_done(task)

        # Signal is emitted when task fails
        mock_failed_signal.assert_called_once_with(
            stage, error=ANY, task_status="failed"
        )

        # On failure, the handle MUST be cleared
        assert artifact_manager.get_job_handle() is None

    def test_adopt_job_artifact(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test adopting a job artifact."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        handle = JobArtifactHandle(
            shm_name="test_job",
            handle_class_name="JobArtifactHandle",
            artifact_type_name="JobArtifact",
            time_estimate=0,
            distance=0,
        )

        artifact_manager.adopt_artifact = MagicMock(return_value=handle)

        data = {"handle_dict": handle.to_dict()}
        adopted = stage._adopt_job_artifact(data)

        assert adopted is handle
        artifact_manager.adopt_artifact.assert_called_once()

    def test_adopt_job_artifact_wrong_type(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test that adopting wrong artifact type raises TypeError."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        wrong_handle = MagicMock()
        wrong_handle.__class__.__name__ = "WrongHandle"

        artifact_manager.adopt_artifact = MagicMock(return_value=wrong_handle)

        data = {"handle_dict": {}}

        with pytest.raises(TypeError, match="Expected a JobArtifactHandle"):
            stage._adopt_job_artifact(data)
