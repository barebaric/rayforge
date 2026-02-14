import uuid
import pytest
from unittest.mock import MagicMock, PropertyMock
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.stage.base import PipelineStage
from rayforge.pipeline.stage.step_stage import StepPipelineStage


@pytest.fixture
def mock_task_mgr():
    """Provides a mock TaskManager."""
    mock_mgr = MagicMock()
    return mock_mgr


@pytest.fixture
def mock_artifact_manager():
    """Provides a mock ArtifactManager."""
    manager = MagicMock()
    manager._ledger = {}
    manager._dependencies = {}
    return manager


@pytest.fixture
def mock_machine():
    """Provides a mock Machine instance."""
    machine = MagicMock()
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000
    machine.acceleration = 1000
    return machine


@pytest.fixture
def mock_scheduler():
    """Provides a mock DagScheduler."""
    scheduler = MagicMock()
    scheduler.step_render_artifact_ready = MagicMock()
    scheduler.step_render_artifact_ready.connect = MagicMock()
    scheduler.step_time_estimate_ready = MagicMock()
    scheduler.step_time_estimate_ready.connect = MagicMock()
    scheduler.step_generation_finished = MagicMock()
    scheduler.step_generation_finished.connect = MagicMock()
    scheduler.process_graph = MagicMock()
    scheduler.mark_node_dirty = MagicMock()
    return scheduler


@pytest.fixture
def mock_doc_and_step():
    """Provides a mock Doc object with some structure."""
    doc = MagicMock(spec=Doc)
    layer = MagicMock(spec=Layer)
    step = MagicMock(spec=Step)
    step.uid = str(uuid.uuid4())
    step.per_step_transformers_dicts = []
    step.visible = True

    type(step).layer = PropertyMock(return_value=layer)

    wp_mock = WorkPiece(name="wp1")
    wp_mock.uid = str(uuid.uuid4())
    wp_mock.set_size(10, 10)

    layer.workflow.steps = [step]
    layer.all_workpieces = [wp_mock]
    doc.layers = [layer]
    return doc, step


@pytest.mark.usefixtures("context_initializer")
class TestStepPipelineStage:
    def test_instantiation(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_scheduler,
    ):
        """Test that StepPipelineStage can be created."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine, mock_scheduler
        )
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is mock_artifact_manager
        assert stage._machine is mock_machine

    def test_mark_stale_and_trigger_starts_assembly(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
        mock_scheduler,
    ):
        """Tests that explicitly marking a step as stale triggers assembly."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine, mock_scheduler
        )

        stage.mark_stale_and_trigger(step, 1)

        mock_scheduler.mark_node_dirty.assert_called_once()
        mock_scheduler.process_graph.assert_called_once()

    def test_signal_forwarding(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
        mock_scheduler,
    ):
        """
        Tests that signals are emitted correctly from handlers.
        """
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine, mock_scheduler
        )

        render_signal_handler = MagicMock()
        time_signal_handler = MagicMock()
        stage.render_artifact_ready.connect(render_signal_handler)
        stage.time_estimate_ready.connect(time_signal_handler)

        stage.handle_time_estimate_ready(step.uid, step, 42.5)

        time_signal_handler.assert_called_once_with(
            stage, step=step, time=42.5
        )
        assert stage.get_estimate(step.uid) == 42.5

    def test_invalidate_cleans_up_artifacts(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
        mock_scheduler,
    ):
        """Tests that invalidating a step cleans up all its artifacts."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine, mock_scheduler
        )

        stage.invalidate(step.uid)

        mock_artifact_manager.pop_step_render_handle.assert_called_with(
            step.uid
        )
        mock_artifact_manager.invalidate_for_step.assert_called_with(
            ArtifactKey.for_step(step.uid)
        )

    def test_time_cache(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
        mock_scheduler,
    ):
        """Tests that time estimates are cached correctly."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine, mock_scheduler
        )

        assert stage.get_estimate(step.uid) is None

        stage.handle_time_estimate_ready(step.uid, step, 42.5)

        assert stage.get_estimate(step.uid) == 42.5
