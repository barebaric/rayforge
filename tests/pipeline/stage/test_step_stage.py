import pytest
from unittest.mock import MagicMock, Mock, PropertyMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.artifact import (
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.artifact.lifecycle import ArtifactLifecycle
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
            self._when_done_callback = kwargs.get("when_done")
            self._when_event_callback = kwargs.get("when_event")
            self.key = kwargs.get("key")
            self.id = id(self)

        def when_done(self, task):
            if self._when_done_callback:
                self._when_done_callback(task)

        def when_event(self, task, event_name, data):
            if self._when_event_callback:
                self._when_event_callback(task, event_name, data)

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
    manager.get_all_step_render_uids.return_value = set()
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
    def test_instantiation(
        self, mock_task_mgr, mock_artifact_manager, mock_machine
    ):
        """Test that StepPipelineStage can be created."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        assert isinstance(stage, PipelineStage)
        assert stage._task_manager is mock_task_mgr
        assert stage._artifact_manager is mock_artifact_manager
        assert stage._machine is mock_machine

    def test_reconcile_triggers_assembly_for_missing_artifact(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """
        Tests that reconcile() starts a task if a step artifact is missing.
        """
        # Arrange
        doc, step = mock_doc_and_step
        from rayforge.pipeline.artifact.lifecycle import (
            ArtifactLifecycle,
            LedgerEntry,
        )

        ledger_key = ("step", step.uid)
        mock_artifact_manager._ledger[ledger_key] = LedgerEntry(
            state=ArtifactLifecycle.MISSING
        )
        mock_artifact_manager.query_work_for_stage.return_value = [ledger_key]

        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        # Act
        stage.reconcile(doc, 1)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_step_artifact_in_subprocess

    def test_mark_stale_and_trigger_starts_assembly(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests that explicitly marking a step as stale triggers assembly."""
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        # Act
        stage.mark_stale_and_trigger(step)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        assert len(mock_task_mgr.created_tasks) == 1

    def test_assembly_flow_success(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """
        Tests the full successful flow: triggering, receiving the render
        event, and then receiving the final time estimate.
        """
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

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
        # Use the actual generation_id set by mark_stale_and_trigger
        gen_id = stage._current_generation_id
        # Set up mock to return None for non-existent ledger entries
        mock_artifact_manager._get_ledger_entry.return_value = None
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
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests that invalidating a step cleans up all its artifacts."""
        # Arrange
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        stage.mark_stale_and_trigger(step)
        _complete_step_task(mock_task_mgr.created_tasks[0])
        mock_task_mgr.created_tasks.clear()

        # Act
        stage.invalidate(step.uid)

        # Assert
        mock_artifact_manager.pop_step_render_handle.assert_called_with(
            step.uid
        )
        mock_artifact_manager.invalidate_for_job.assert_called()

    def test_validate_assembly_dependencies_missing_layer(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests validation fails when step has no layer."""
        doc, step = mock_doc_and_step
        type(step).layer = PropertyMock(return_value=None)
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        result = stage._validate_assembly_dependencies(step)

        assert result is False

    def test_validate_assembly_dependencies_active_task(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests validation fails when task is already active."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        ledger_key = ("step", step.uid)
        mock_artifact_manager._ledger[ledger_key] = Mock(
            state=ArtifactLifecycle.PENDING
        )

        result = stage._validate_assembly_dependencies(step)

        assert result is False

    def test_validate_assembly_dependencies_no_machine(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_doc_and_step,
    ):
        """Tests validation fails when no machine is configured."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(mock_task_mgr, mock_artifact_manager, None)

        result = stage._validate_assembly_dependencies(step)

        assert result is False

    def test_validate_assembly_dependencies_success(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests validation succeeds when all dependencies are met."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        result = stage._validate_assembly_dependencies(step)

        assert result is True

    def test_validate_handle_geometry_match_scalable(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Tests geometry validation passes for scalable handles."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        handle = WorkPieceArtifactHandle(
            is_scalable=True,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
            shm_name="dummy_wp_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
        )
        wp = WorkPiece(name="wp1")
        wp.set_size(20, 20)

        result = stage._validate_handle_geometry_match(handle, wp)

        assert result is True

    def test_validate_handle_geometry_match_non_scalable_match(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Tests geometry validation passes for matching non-scalable."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        handle = WorkPieceArtifactHandle(
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
            shm_name="dummy_wp_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
        )
        wp = WorkPiece(name="wp1")
        wp.set_size(10, 10)

        result = stage._validate_handle_geometry_match(handle, wp)

        assert result is True

    def test_validate_handle_geometry_match_non_scalable_mismatch(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Tests geometry validation fails for mismatched non-scalable."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        handle = WorkPieceArtifactHandle(
            is_scalable=False,
            source_coordinate_system_name="MILLIMETER_SPACE",
            source_dimensions=(10, 10),
            generation_size=(10, 10),
            shm_name="dummy_wp_shm",
            handle_class_name="WorkPieceArtifactHandle",
            artifact_type_name="WorkPieceArtifact",
        )
        wp = WorkPiece(name="wp1")
        wp.set_size(20, 20)

        result = stage._validate_handle_geometry_match(handle, wp)

        assert result is False

    def test_collect_assembly_info_missing_handle(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests assembly info collection returns None when handle missing."""
        doc, step = mock_doc_and_step
        mock_artifact_manager.get_workpiece_handle.return_value = None
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        result, retained = stage._collect_assembly_info(step)

        assert result is None

    def test_collect_assembly_info_geometry_mismatch(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests assembly info returns None on geometry mismatch."""
        doc, step = mock_doc_and_step
        mock_artifact_manager.get_workpiece_handle.return_value = (
            WorkPieceArtifactHandle(
                is_scalable=False,
                source_coordinate_system_name="MILLIMETER_SPACE",
                source_dimensions=(10, 10),
                generation_size=(10, 10),
                shm_name="dummy_wp_shm",
                handle_class_name="WorkPieceArtifactHandle",
                artifact_type_name="WorkPieceArtifact",
            )
        )
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        step.layer.all_workpieces[0].set_size(20, 20)

        result, retained = stage._collect_assembly_info(step)

        assert result is None

    def test_collect_assembly_info_success(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests successful assembly info collection."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        result, retained = stage._collect_assembly_info(step)

        assert result is not None
        assert len(result) == 1
        assert "artifact_handle_dict" in result[0]
        assert "world_transform_list" in result[0]
        assert "workpiece_dict" in result[0]

    def test_handle_render_artifact_ready(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests render artifact ready event handling."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        render_handle = StepRenderArtifactHandle(
            shm_name="render_shm",
            handle_class_name="StepRenderArtifactHandle",
            artifact_type_name="StepRenderArtifact",
        )
        mock_artifact_manager.adopt_artifact.return_value = render_handle

        signal_handler = MagicMock()
        stage.render_artifact_ready.connect(signal_handler)

        stage._handle_render_artifact_ready(
            step.uid, step, render_handle.to_dict()
        )

        mock_artifact_manager.put_step_render_handle.assert_called_once_with(
            step.uid, render_handle
        )
        signal_handler.assert_called_once_with(stage, step=step)

    def test_handle_ops_artifact_ready(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
    ):
        """Tests ops artifact ready event handling."""
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )
        ops_handle = StepOpsArtifactHandle(
            shm_name="ops_shm",
            handle_class_name="StepOpsArtifactHandle",
            artifact_type_name="StepOpsArtifact",
            time_estimate=None,
        )
        mock_artifact_manager.adopt_artifact.return_value = ops_handle

        stage._handle_ops_artifact_ready("step1", ops_handle.to_dict())

        mock_artifact_manager.adopt_artifact.assert_called_once()

    def test_handle_time_estimate_ready(
        self,
        mock_task_mgr,
        mock_artifact_manager,
        mock_machine,
        mock_doc_and_step,
    ):
        """Tests time estimate ready event handling."""
        doc, step = mock_doc_and_step
        stage = StepPipelineStage(
            mock_task_mgr, mock_artifact_manager, mock_machine
        )

        signal_handler = MagicMock()
        stage.time_estimate_ready.connect(signal_handler)

        stage._handle_time_estimate_ready(step.uid, step, 42.5)

        assert stage.get_estimate(step.uid) == 42.5
        signal_handler.assert_called_once_with(stage, step=step, time=42.5)
