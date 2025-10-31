"""Tests for pipeline state management (pause, resume, busy, shutdown)."""

import pytest
import asyncio
from unittest.mock import MagicMock, ANY
from pathlib import Path
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.steps import create_contour_step
from rayforge.shared.tasker.task import Task


@pytest.fixture
def mock_task_mgr():
    """Creates a MagicMock for the TaskManager that is "event-loop-aware"."""
    mock_mgr = MagicMock()
    created_tasks_info = []

    class MockTask:
        def __init__(self, target, args, kwargs, returned_task_obj):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.when_done = kwargs.get("when_done")
            self.when_event = kwargs.get("when_event")
            self.key = kwargs.get("key")
            self.returned_task_obj = returned_task_obj

    def run_process_mock(target_func, *args, **kwargs):
        mock_returned_task = MagicMock(spec=Task)
        mock_returned_task.key = kwargs.get("key")
        task = MockTask(target_func, args, kwargs, mock_returned_task)
        created_tasks_info.append(task)
        return mock_returned_task

    def schedule_awarely(callback, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(callback, *args, **kwargs)
        except RuntimeError:
            callback(*args, **kwargs)

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.schedule_on_main_thread = MagicMock(side_effect=schedule_awarely)
    mock_mgr.created_tasks = created_tasks_info
    return mock_mgr


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    return WorkPiece(name="real_workpiece.svg")


@pytest.fixture
def doc():
    d = Doc()
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipelineState:
    """Tests for pipeline state management."""

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
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def test_shutdown_releases_all_artifacts(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        task_info = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        task_obj = task_info.returned_task_obj
        task_obj.key = task_info.key
        task_obj.get_status.return_value = "completed"
        task_obj.result.return_value = 1

        try:
            event_data = {"handle_dict": handle.to_dict(), "generation_id": 1}
            if task_info.when_event:
                task_info.when_event(task_obj, "artifact_created", event_data)
            if task_info.when_done:
                task_info.when_done(task_obj)

            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is not None
            )
            pipeline.shutdown()
            assert (
                pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
                is None
            )
        finally:
            pass  # Handle should be released by shutdown

    def test_doc_property_management(self, doc, mock_task_mgr):
        pipeline = Pipeline(doc, mock_task_mgr)
        assert pipeline.doc is doc

        pipeline.doc = doc  # Setting same doc should be a no-op
        assert pipeline.doc is doc

        new_doc = Doc()
        pipeline.doc = new_doc
        assert pipeline.doc is new_doc

    def test_is_busy_property(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        assert pipeline.is_busy is True

        task_info = mock_task_mgr.created_tasks[0]
        task_obj = task_info.returned_task_obj
        task_obj.get_status.return_value = "completed"
        task_obj.result.return_value = 1
        if task_info.when_done:
            task_info.when_done(task_obj)

        assert pipeline.is_busy is False

    def test_pause_resume_functionality(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.reset_mock()

        pipeline.pause()
        assert pipeline.is_paused is True

        real_workpiece.set_size(20, 20)
        mock_task_mgr.run_process.assert_not_called()

        pipeline.resume()
        assert pipeline.is_paused is False
        mock_task_mgr.run_process.assert_called()

    def test_paused_context_manager(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.reset_mock()

        with pipeline.paused():
            assert pipeline.is_paused is True
            real_workpiece.set_size(20, 20)
            mock_task_mgr.run_process.assert_not_called()

        assert pipeline.is_paused is False
        mock_task_mgr.run_process.assert_called()

    def test_rapid_invalidation_does_not_corrupt_busy_state(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Simulates a rapid invalidation that cancels an in-progress task and
        starts a new one. This test verifies that the callback from the old,
        cancelled task does NOT corrupt the stage's internal state by
        prematurely clearing the 'active_tasks' dict.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        mock_artifact_ready_handler = MagicMock()
        mock_processing_state_handler = MagicMock()

        # Capture the actual task object returned by the mocked run_process
        returned_tasks = []
        original_side_effect = mock_task_mgr.run_process.side_effect

        def side_effect_wrapper(*args, **kwargs):
            returned_task = original_side_effect(*args, **kwargs)
            returned_tasks.append(returned_task)
            return returned_task

        mock_task_mgr.run_process.side_effect = side_effect_wrapper

        # Act 1: Create pipeline with an empty doc, so it's idle.
        pipeline = Pipeline(doc=Doc(), task_manager=mock_task_mgr)
        pipeline.workpiece_artifact_ready.connect(mock_artifact_ready_handler)
        pipeline.processing_state_changed.connect(
            mock_processing_state_handler
        )

        assert pipeline.is_busy is False
        mock_task_mgr.run_process.assert_not_called()

        # Act 2: Set the doc property. This triggers reconcile_all() and starts
        # task1.
        pipeline.doc = doc

        # Assert 2: Pipeline is now busy, and the state change signal was
        # fired.
        assert pipeline.is_busy is True
        mock_task_mgr.run_process.assert_called_once()
        assert len(mock_task_mgr.created_tasks) == 1
        task1_info = mock_task_mgr.created_tasks[0]
        # Our side effect should have captured the returned task object
        assert len(returned_tasks) == 1
        task1_object_in_stage = returned_tasks[0]

        mock_processing_state_handler.assert_called_with(
            ANY, is_processing=True
        )

        # Reset mocks for the next phase
        mock_task_mgr.run_process.reset_mock()
        mock_task_mgr.created_tasks.clear()
        returned_tasks.clear()
        mock_processing_state_handler.reset_mock()

        # Act 3: Trigger a second regeneration immediately, cancelling task 1
        # and starting task 2.
        step.power = 0.5
        pipeline._on_descendant_updated(sender=step, origin=step)

        # Assert 3: A new task was created and the pipeline remains busy,
        # without firing redundant state change signals.
        mock_task_mgr.run_process.assert_called_once()
        assert len(mock_task_mgr.created_tasks) == 1
        task2_info = mock_task_mgr.created_tasks[0]
        assert len(returned_tasks) == 1

        # The state should remain busy, so no new signals should have fired.
        mock_processing_state_handler.assert_not_called()
        assert pipeline.is_busy is True

        # Act 4: Simulate the 'when_done' callback of the CANCELLED
        # task (task1) firing. Use the actual task object that was stored
        # in the stage.
        task1_object_in_stage.get_status.return_value = "canceled"
        if task1_info.when_done:
            task1_info.when_done(task1_object_in_stage)

        # The pipeline should remain busy because task2 is still active.
        assert pipeline.is_busy is True, (
            "Pipeline incorrectly became idle after a cancelled task's "
            "callback."
        )
        mock_artifact_ready_handler.assert_not_called()

        # Act 5: Simulate the SUCCESSFUL task (task2) completing.
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            task2_object_in_stage = returned_tasks[0]
            # Use the actual task object that the stage is now holding
            # for task 2
            task2_object_in_stage.get_status.return_value = "completed"
            task2_object_in_stage.result.return_value = 2  # Gen ID for task 2

            if task2_info.when_event:
                task2_info.when_event(
                    task2_object_in_stage,
                    "artifact_created",
                    {"handle_dict": handle.to_dict(), "generation_id": 2},
                )

            # The workpiece task completion will trigger a step task.
            # We must simulate that one finishing as well.
            if task2_info.when_done:
                task2_info.when_done(task2_object_in_stage)

            # Find the newly created step task and complete it.
            step_task_info = next(
                (
                    t
                    for t in mock_task_mgr.created_tasks
                    if t.target is make_step_artifact_in_subprocess
                ),
                None,
            )
            assert step_task_info is not None, "Step task was not created"
            step_task_obj = step_task_info.returned_task_obj
            step_task_obj.get_status.return_value = "completed"
            if step_task_info.when_done:
                step_task_info.when_done(step_task_obj)

            # Assert 5: The final signals were emitted correctly.
            mock_artifact_ready_handler.assert_called_once()
            assert pipeline.is_busy is False

            # The callback should have triggered the state change signal.
            mock_processing_state_handler.assert_called_with(
                ANY, is_processing=False
            )
        finally:
            get_context().artifact_store.release(handle)
