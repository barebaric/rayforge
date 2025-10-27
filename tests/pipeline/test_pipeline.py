import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
from rayforge.shared.tasker.task import Task
from rayforge.image import SVG_RENDERER
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    VertexData,
    WorkPieceArtifactHandle,
    StepRenderArtifactHandle,
    StepOpsArtifactHandle,
)
from rayforge.context import get_context
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)


@pytest.fixture
def mock_task_mgr():
    """
    Creates a MagicMock for the TaskManager that executes scheduled calls
    synchronously for predictable testing.
    """
    mock_mgr = MagicMock()
    created_tasks_info = []

    class MockTask:
        def __init__(self, target, args, kwargs):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.when_done = kwargs.get("when_done")
            self.when_event = kwargs.get("when_event")
            self.key = kwargs.get("key")

    def run_process_mock(target_func, *args, **kwargs):
        task = MockTask(target_func, args, kwargs)
        created_tasks_info.append(task)
        return task

    # Make scheduled calls run immediately
    def schedule_sync(callback, *args, **kwargs):
        callback(*args, **kwargs)

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.schedule_on_main_thread = MagicMock(side_effect=schedule_sync)
    mock_mgr.created_tasks = created_tasks_info
    return mock_mgr


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    workpiece = WorkPiece(name="real_workpiece.svg")
    # Importer will set size and pos, we simulate it in the setup helper.
    return workpiece


@pytest.fixture
def doc():
    d = Doc()
    # Get the active layer (the first workpiece layer) and clear its steps
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipeline:
    # This data is used by multiple tests to create the ImportSource.
    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def _setup_doc_with_workpiece(self, doc, workpiece):
        """Helper to correctly link a workpiece to a source within a doc."""
        source = ImportSource(
            Path(workpiece.name),
            original_data=self.svg_data,
            renderer=SVG_RENDERER,
        )
        doc.add_import_source(source)
        workpiece.import_source_uid = source.uid
        # Simulate importer setting the size and pos
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def _complete_all_tasks(
        self, mock_task_mgr, workpiece_handle, step_time=42.0
    ):
        """
        Helper to find and complete all outstanding tasks to bring the
        pipeline to an idle state. Simulates the new event-driven flow.
        """
        processed_keys = set()
        while True:
            tasks_to_process = [
                t
                for t in mock_task_mgr.created_tasks
                if t.key not in processed_keys
            ]
            if not tasks_to_process:
                break

            for task in tasks_to_process:
                mock_task_obj = MagicMock(spec=Task)
                mock_task_obj.key = task.key
                mock_task_obj.get_status.return_value = "completed"

                if task.target is make_workpiece_artifact_in_subprocess:
                    result = (workpiece_handle.to_dict(), 1)
                    mock_task_obj.result.return_value = result
                    if task.when_done:
                        task.when_done(mock_task_obj)
                elif task.target is make_step_artifact_in_subprocess:
                    if task.when_event:
                        # 1. Simulate render artifact event
                        render_handle = StepRenderArtifactHandle(
                            shm_name="dummy_render",
                            handle_class_name="StepRenderArtifactHandle",
                            artifact_type_name="StepRenderArtifact",
                        )
                        render_event = {
                            "handle_dict": render_handle.to_dict(),
                            "generation_id": 1,
                        }
                        task.when_event(
                            mock_task_obj,
                            "render_artifact_ready",
                            render_event,
                        )

                        # 2. Simulate ops artifact event
                        ops_handle = StepOpsArtifactHandle(
                            shm_name="dummy_ops",
                            handle_class_name="StepOpsArtifactHandle",
                            artifact_type_name="StepOpsArtifact",
                            time_estimate=step_time,
                        )
                        ops_event = {
                            "handle_dict": ops_handle.to_dict(),
                            "generation_id": 1,
                        }
                        task.when_event(
                            mock_task_obj, "ops_artifact_ready", ops_event
                        )

                    # 3. Simulate final result (time estimate)
                    result = (step_time, 1)
                    mock_task_obj.result.return_value = result
                    if task.when_done:
                        task.when_done(mock_task_obj)

                processed_keys.add(task.key)

        mock_task_mgr.created_tasks.clear()

    def test_reconcile_all_triggers_ops_generation(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        # Act
        Pipeline(doc, mock_task_mgr)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_workpiece_artifact_in_subprocess

    def test_generation_success_emits_signals_and_caches_result(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        task_to_complete = mock_task_mgr.created_tasks[0]

        # Act
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(1, 1, 0)

        vertex_data = VertexData(
            powered_vertices=np.array([[0, 0, 0], [1, 1, 0]]),
            powered_colors=np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
        )

        expected_artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
            vertex_data=vertex_data,
        )
        handle = get_context().artifact_store.put(expected_artifact)
        expected_result_tuple = (handle.to_dict(), 1)

        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task_to_complete.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = expected_result_tuple

        try:
            task_to_complete.when_done(mock_finished_task)
            cached_ops = pipeline.get_ops(step, real_workpiece)
            assert cached_ops is not None
            assert len(cached_ops) == 2
        finally:
            get_context().artifact_store.release(handle)

    def test_generation_cancellation_is_handled(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        task_to_cancel = mock_task_mgr.created_tasks[0]

        # Act
        mock_cancelled_task = MagicMock(spec=Task)
        mock_cancelled_task.key = task_to_cancel.key
        mock_cancelled_task.get_status.return_value = "cancelled"
        task_to_cancel.when_done(mock_cancelled_task)

        # Assert
        assert pipeline.get_ops(step, real_workpiece) is None

    def test_step_change_triggers_full_regeneration(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            # Act
            # Change a property on the step. This would normally fire a signal.
            step.power = 0.5
            # Manually call the signal handler to simulate the pipeline's
            # correct reaction to a step update.
            pipeline._on_descendant_updated(sender=step, origin=step)

            # Assert
            tasks = mock_task_mgr.created_tasks
            workpiece_tasks = [
                t
                for t in tasks
                if t.target is make_workpiece_artifact_in_subprocess
            ]
            assert len(workpiece_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_workpiece_transform_change_triggers_step_assembly(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        _ = Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            # Act
            real_workpiece.pos = (50, 50)
            # This change fires a signal that the pipeline handles,
            # so we don't need to do anything else to trigger it.

            # Assert
            tasks = mock_task_mgr.created_tasks
            assembly_tasks = [
                t
                for t in tasks
                if t.target is make_step_artifact_in_subprocess
            ]
            assert len(assembly_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_multipass_change_triggers_step_assembly(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            # Act
            step.per_step_transformers_dicts = []
            pipeline._on_job_assembly_invalidated(sender=doc)

            # Assert
            tasks = mock_task_mgr.created_tasks
            assembly_tasks = [
                t
                for t in tasks
                if t.target is make_step_artifact_in_subprocess
            ]
            assert len(assembly_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_workpiece_size_change_triggers_regeneration(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        _ = Pipeline(doc, mock_task_mgr)

        initial_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(initial_artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            # Act
            real_workpiece.set_size(10, 10)
            # This change fires a signal that the pipeline handles.

            # Assert
            tasks = mock_task_mgr.created_tasks
            workpiece_tasks = [
                t
                for t in tasks
                if t.target is make_workpiece_artifact_in_subprocess
            ]
            assert len(workpiece_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_shutdown_releases_all_artifacts(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Simulate completion of a task to populate the cache
        task = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Verify handle is in cache
            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is not None
            )

            # Act
            pipeline.shutdown()

            # Assert
            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is None
            )
        finally:
            # handle should already be released by shutdown
            pass

    def test_doc_property_getter(self, doc, mock_task_mgr):
        # Arrange
        pipeline = Pipeline(doc, mock_task_mgr)

        # Act & Assert
        assert pipeline.doc is doc

    def test_doc_property_setter_with_same_doc(self, doc, mock_task_mgr):
        # Arrange
        pipeline = Pipeline(doc, mock_task_mgr)

        # Act - setting the same document should not cause issues
        pipeline.doc = doc

        # Assert
        assert pipeline.doc is doc

    def test_doc_property_setter_with_different_doc(self, doc, mock_task_mgr):
        # Arrange
        pipeline = Pipeline(doc, mock_task_mgr)
        new_doc = Doc()

        # Act
        pipeline.doc = new_doc

        # Assert
        assert pipeline.doc is new_doc

    def test_is_busy_property(self, doc, real_workpiece, mock_task_mgr):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Initial state - should be busy with one task
        assert pipeline.is_busy is True

        # Complete the task
        task = mock_task_mgr.created_tasks[0]
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (None, 1)

        # Act
        task.when_done(mock_finished_task)

        # Assert - should not be busy anymore
        assert pipeline.is_busy is False

    def test_pause_resume_functionality(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.reset_mock()  # Reset after initialization

        # Act - pause the pipeline
        pipeline.pause()
        assert pipeline.is_paused is True

        # Try to trigger regeneration - should not happen while paused
        real_workpiece.set_size(20, 20)
        mock_task_mgr.run_process.assert_not_called()

        # Resume the pipeline
        pipeline.resume()
        assert pipeline.is_paused is False

        # Assert - reconciliation should happen after resume
        mock_task_mgr.run_process.assert_called()

    def test_paused_context_manager(self, doc, real_workpiece, mock_task_mgr):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.reset_mock()  # Reset after initialization

        # Act - use context manager
        with pipeline.paused():
            assert pipeline.is_paused is True
            # Try to trigger regeneration - should not happen while paused
            real_workpiece.set_size(20, 20)
            mock_task_mgr.run_process.assert_not_called()

        # Assert - should be resumed after context
        assert pipeline.is_paused is False
        # Reconciliation should happen after resume
        mock_task_mgr.run_process.assert_called()

    def test_is_paused_property(self, doc, mock_task_mgr):
        # Arrange
        pipeline = Pipeline(doc, mock_task_mgr)

        # Initial state
        assert pipeline.is_paused is False

        # After pause
        pipeline.pause()
        assert pipeline.is_paused is True

        # After resume
        pipeline.resume()
        assert pipeline.is_paused is False

    def test_get_estimated_time_returns_none(
        self, doc, real_workpiece, mock_task_mgr
    ):
        """
        Tests that the refactored get_estimated_time now correctly
        returns None as it's no longer per-workpiece.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        # Act
        result = pipeline.get_estimated_time(step, real_workpiece)

        # Assert
        assert result is None

    def test_preview_time_updated_signal_is_correct(
        self, doc, real_workpiece, mock_task_mgr
    ):
        """
        Tests the new end-to-end time estimation by checking the final
        signal received by the UI.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Create a dummy workpiece artifact to allow the pipeline to proceed
        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)

        mock_handler = MagicMock()
        pipeline.job_time_updated.connect(mock_handler)

        # Act
        try:
            # Complete all tasks, simulating a time of 55.5s for the step
            self._complete_all_tasks(mock_task_mgr, wp_handle, step_time=55.5)

            # Assert
            # The handler is called multiple times (e.g., initially with None)
            # We check the final call to see if it received the correct value.
            mock_handler.assert_called()
            last_call_args, last_call_kwargs = mock_handler.call_args_list[-1]
            assert last_call_kwargs.get("total_seconds") == 55.5
        finally:
            get_context().artifact_store.release(wp_handle)

    def test_get_artifact_handle(self, doc, real_workpiece, mock_task_mgr):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Act & Assert - No handle initially
        assert (
            pipeline.get_artifact_handle(step.uid, real_workpiece.uid) is None
        )

        # Simulate a completed task
        task = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Act & Assert - Should return the handle
            retrieved_handle = pipeline.get_artifact_handle(
                step.uid, real_workpiece.uid
            )
            assert retrieved_handle is not None
            assert isinstance(retrieved_handle, WorkPieceArtifactHandle)
            assert retrieved_handle.generation_size == real_workpiece.size
        finally:
            get_context().artifact_store.release(handle)

    def test_get_scaled_ops(self, doc, real_workpiece, mock_task_mgr):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Act & Assert - No ops initially
        world_transform = real_workpiece.get_world_transform()
        assert (
            pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )
            is None
        )

        # Simulate a completed task with scalable artifact
        task = mock_task_mgr.created_tasks[0]
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(10, 10, 0)

        artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Act
            scaled_ops = pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )

            # Assert
            assert scaled_ops is not None
            assert len(scaled_ops) == 2  # MoveTo + LineTo
        finally:
            get_context().artifact_store.release(handle)

    def test_get_scaled_ops_with_stale_non_scalable_artifact(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Simulate a completed task with non-scalable artifact at
        # different size
        task = mock_task_mgr.created_tasks[0]
        original_size = (25, 15)  # Different from workpiece size
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,  # Not scalable
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=original_size,
            generation_size=original_size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Act - Try to get scaled ops for different size
            world_transform = real_workpiece.get_world_transform()
            scaled_ops = pipeline.get_scaled_ops(
                step.uid, real_workpiece.uid, world_transform
            )

            # Assert - Should return None for stale non-scalable artifact
            assert scaled_ops is None
        finally:
            get_context().artifact_store.release(handle)

    def test_get_artifact(self, doc, real_workpiece, mock_task_mgr):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Act & Assert - No artifact initially
        assert pipeline.get_artifact(step, real_workpiece) is None

        # Simulate a completed task
        task = mock_task_mgr.created_tasks[0]
        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(10, 10, 0)

        artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Act
            retrieved_artifact = pipeline.get_artifact(step, real_workpiece)

            # Assert
            assert retrieved_artifact is not None
            assert retrieved_artifact.is_scalable is True
            assert len(retrieved_artifact.ops) == 2  # MoveTo + LineTo
            assert retrieved_artifact.source_dimensions == real_workpiece.size
        finally:
            get_context().artifact_store.release(handle)

    def test_get_artifact_with_stale_non_scalable_artifact(
        self, doc, real_workpiece, mock_task_mgr
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step()
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Simulate a completed task with non-scalable artifact at
        # different size
        task = mock_task_mgr.created_tasks[0]
        original_size = (25, 15)  # Different from workpiece size
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,  # Not scalable
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=original_size,
            generation_size=original_size,
        )
        handle = get_context().artifact_store.put(artifact)
        mock_finished_task = MagicMock(spec=Task)
        mock_finished_task.key = task.key
        mock_finished_task.get_status.return_value = "completed"
        mock_finished_task.result.return_value = (handle.to_dict(), 1)

        try:
            task.when_done(mock_finished_task)

            # Act - Try to get artifact for different size
            retrieved_artifact = pipeline.get_artifact(step, real_workpiece)

            # Assert - Should return None for stale non-scalable artifact
            assert retrieved_artifact is None
        finally:
            get_context().artifact_store.release(handle)
