import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.step import StepGeneratorStage
from rayforge.pipeline.artifact import StepArtifactHandle
from rayforge.machine.models.machine import Machine, Laser


@pytest.fixture(autouse=True)
def setup_real_config(mocker):
    """
    This fixture is now autouse=True for this module, ensuring config is
    always patched correctly.
    """
    test_laser = Laser()
    test_machine = Machine()
    test_machine.max_cut_speed = 5000
    test_machine.max_travel_speed = 10000
    test_machine.acceleration = 1000
    test_machine.add_head(test_laser)

    class TestConfig:
        machine = test_machine

    test_config = TestConfig()
    mocker.patch("rayforge.pipeline.stage.step.config.config", test_config)
    mocker.patch("builtins._", lambda s: s, create=True)
    return test_config


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
def mock_artifact_cache():
    """Provides a mock ArtifactCache."""
    cache = MagicMock()
    cache.get_workpiece_handle.return_value = MagicMock()  # Dependency is met
    return cache


@pytest.fixture
def mock_doc_and_step():
    """Provides a mock Doc object with some structure."""
    doc = MagicMock(spec=Doc)
    layer = MagicMock(spec=Layer)
    step = MagicMock(spec=Step)
    step.uid = "step1"
    step.layer = layer
    step.per_step_transformers_dicts = []  # Add the missing attribute

    # Give the mock workpiece a UID
    wp_mock = MagicMock(spec=WorkPiece)
    wp_mock.uid = "wp1"

    layer.workflow.steps = [step]
    layer.all_workpieces = [wp_mock]
    doc.layers = [layer]
    return doc, step


class TestStepGeneratorStage:
    def test_instantiation(self, mock_task_mgr, mock_artifact_cache):
        """Test that StepGeneratorStage can be created."""
        stage = StepGeneratorStage(mock_task_mgr, mock_artifact_cache)
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_cache is mock_artifact_cache

    def test_assembly_flow_success(
        self, mock_task_mgr, mock_artifact_cache, mock_doc_and_step
    ):
        """
        Tests the full successful flow: triggering, receiving the render
        event, and then receiving the final time estimate.
        """
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepGeneratorStage(mock_task_mgr, mock_artifact_cache)

        render_signal_handler = MagicMock()
        time_signal_handler = MagicMock()
        stage.render_artifact_ready.connect(render_signal_handler)
        stage.time_estimate_ready.connect(time_signal_handler)

        # Act
        stage.mark_stale_and_trigger(step)

        # Assert a task was created
        assert len(mock_task_mgr.created_tasks) == 1
        task = mock_task_mgr.created_tasks[0]

        # --- Simulate Phase 1: Render Artifact Ready Event ---
        mock_task_obj = MagicMock()
        render_handle = StepArtifactHandle(
            shm_name="render_shm",
            handle_class_name="StepArtifactHandle",
            artifact_type_name="StepArtifact",
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=None,
            time_estimate=None,
            array_metadata={},
        )
        event_data = {
            "handle_dict": render_handle.to_dict(),
            "generation_id": 1,  # Matches the stage's internal ID
        }
        task.when_event(mock_task_obj, "render_artifact_ready", event_data)

        # Assert render phase worked
        mock_artifact_cache.put_step_handle.assert_called_once()
        render_signal_handler.assert_called_once_with(stage, step=step)

        # --- Simulate Phase 2: Task Completion with Time Estimate ---
        mock_task_obj.get_status.return_value = "completed"
        mock_task_obj.result.return_value = (42.5, 1)  # (time, gen_id)
        task.when_done(mock_task_obj)

        # Assert time phase worked
        time_signal_handler.assert_called_once_with(
            stage, step=step, time=42.5
        )
        assert stage.get_estimate(step.uid) == 42.5
