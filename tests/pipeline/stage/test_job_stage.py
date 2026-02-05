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

    def test_validate_job_generation_state_busy(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test validation when job generation is already in progress."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        # Simulate busy state
        mock_task = MagicMock()
        stage._active_task = mock_task

        callback = MagicMock()
        error = stage._validate_job_generation_state(callback)

        assert error is not None
        assert "already in progress" in str(error)
        callback.assert_called_once()

    def test_validate_job_generation_state_no_machine(
        self, mock_task_mgr, artifact_manager
    ):
        """Test validation when no machine is configured."""
        stage = JobPipelineStage(
            mock_task_mgr,
            artifact_manager,
            None,  # type: ignore[arg-type]
        )

        callback = MagicMock()
        error = stage._validate_job_generation_state(callback)

        assert error is not None
        assert "No machine is configured" in str(error)
        callback.assert_called_once()

    def test_validate_job_generation_state_success(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test validation when state is valid."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        callback = MagicMock()
        error = stage._validate_job_generation_state(callback)

        assert error is None
        callback.assert_not_called()

    def test_collect_step_handles(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test collecting step handles from document layers."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_manager.put_step_ops_handle(step.uid, step_ops_handle)

        step_handles = stage._collect_step_handles(real_doc_with_step)

        assert step.uid in step_handles
        assert step_handles[step.uid] == step_ops_handle.to_dict()

    def test_collect_step_handles_invisible_step(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test that invisible steps are not collected."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        step = real_doc_with_step.active_layer.workflow.steps[0]
        step.visible = False

        step_ops_handle = StepOpsArtifactHandle(
            shm_name="dummy_step_ops",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=10.0,
        )
        artifact_manager.put_step_ops_handle(step.uid, step_ops_handle)

        step_handles = stage._collect_step_handles(real_doc_with_step)

        assert step.uid not in step_handles

    def test_collect_step_handles_no_handle(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test that steps without handles are skipped."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        step_handles = stage._collect_step_handles(real_doc_with_step)

        assert len(step_handles) == 0

    def test_create_adoption_event(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test creation of adoption event."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        event = stage._create_adoption_event()

        assert event is not None
        assert not event.is_set()

    def test_create_job_description(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test creation of job description."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        step_handles = {"step1": {"shm_name": "test"}}
        job_desc = stage._create_job_description(
            step_handles, machine, real_doc_with_step
        )

        assert job_desc.step_artifact_handles_by_uid == step_handles
        assert "machine_dict" in job_desc.__dict__
        assert "doc_dict" in job_desc.__dict__

    def test_is_task_active(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test checking if task is active."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        mock_task = MagicMock()
        stage._active_task = mock_task

        assert stage._is_task_active(mock_task) is True

        other_task = MagicMock()
        assert stage._is_task_active(other_task) is False

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

    def test_complete_adoption_handshake(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test completing the adoption handshake."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        event = stage._create_adoption_event()
        stage._adoption_event = event

        assert not event.is_set()

        stage._complete_adoption_handshake()

        assert event.is_set()

    def test_complete_adoption_handshake_no_event(
        self, mock_task_mgr, artifact_manager, test_machine_and_config
    ):
        """Test handshake completion when no event exists."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        stage._adoption_event = None

        # Should not raise an error
        stage._complete_adoption_handshake()

    def test_handle_artifact_created_inactive_task(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test handling artifact_created from inactive task."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        stage._adoption_event = stage._create_adoption_event()

        other_task = MagicMock()
        data = {"handle_dict": {}}

        stage._handle_artifact_created(other_task, data)

        # Event should still be set to unblock worker
        assert stage._adoption_event.is_set()

    def test_handle_artifact_created_error(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test handling artifact_created with error."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        stage.generate_job(real_doc_with_step)
        task = mock_task_mgr.created_tasks[0]

        artifact_manager.adopt_artifact = MagicMock(
            side_effect=RuntimeError("Adoption failed")
        )

        data = {"handle_dict": {}}

        stage._handle_artifact_created(task, data)

        # Event should still be set to unblock worker
        assert stage._adoption_event is not None
        assert stage._adoption_event.is_set()

    def test_generate_job_busy_state(
        self,
        mock_task_mgr,
        artifact_manager,
        real_doc_with_step,
        test_machine_and_config,
    ):
        """Test generate_job when already busy."""
        machine, _ = test_machine_and_config
        stage = JobPipelineStage(mock_task_mgr, artifact_manager, machine)

        # Set busy state
        mock_task = MagicMock()
        stage._active_task = mock_task

        callback = MagicMock()
        stage.generate_job(real_doc_with_step, callback)

        callback.assert_called_once()
        assert "already in progress" in str(callback.call_args[0][1])

    def test_generate_job_no_machine(
        self, mock_task_mgr, artifact_manager, real_doc_with_step
    ):
        """Test generate_job when no machine is configured."""
        stage = JobPipelineStage(
            mock_task_mgr,
            artifact_manager,
            None,  # type: ignore[arg-type]
        )

        callback = MagicMock()
        stage.generate_job(real_doc_with_step, callback)

        callback.assert_called_once()
        assert "No machine is configured" in str(callback.call_args[0][1])
