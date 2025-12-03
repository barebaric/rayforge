import pytest
import logging
import asyncio
import threading
from unittest.mock import MagicMock, ANY
from pathlib import Path
from rayforge.shared.tasker.task import Task
from rayforge.image import SVG_RENDERER
from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.geo import Geometry
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.job_runner import make_job_artifact_in_subprocess
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    StepRenderArtifact,
    StepOpsArtifact,
    JobArtifact,
)
from rayforge.context import get_context


logger = logging.getLogger(__name__)


@pytest.fixture
def mock_task_mgr():
    """
    Creates a MagicMock for the TaskManager that executes scheduled tasks
    immediately.
    """
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
        # Add a mock cancel method to the task object returned to the caller
        mock_returned_task = MagicMock(spec=Task)
        mock_returned_task.key = kwargs.get("key")

        task = MockTask(target_func, args, kwargs, mock_returned_task)
        created_tasks_info.append(task)
        return mock_returned_task

    # Execute scheduled callbacks synchronously.
    def schedule_awarely(callback, *args, **kwargs):
        callback(*args, **kwargs)

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.schedule_on_main_thread = MagicMock(side_effect=schedule_awarely)
    mock_mgr.created_tasks = created_tasks_info
    return mock_mgr


@pytest.fixture(autouse=True)
def zero_debounce_delay(monkeypatch):
    monkeypatch.setattr(Pipeline, "RECONCILIATION_DELAY_MS", 0)


@pytest.fixture(autouse=True)
def mock_threading_timer(monkeypatch):
    class SyncTimer:
        def __init__(self, interval, function, args=None, kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args or []
            self.kwargs = kwargs or {}
            self._cancelled = False

        def start(self):
            if not self._cancelled:
                self.function(*self.args, **self.kwargs)

        def cancel(self):
            self._cancelled = True

    monkeypatch.setattr(threading, "Timer", SyncTimer)


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    workpiece = WorkPiece(name="real_workpiece.svg")
    return workpiece


@pytest.fixture
def doc():
    d = Doc()
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipelineState:
    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def _setup_doc_with_workpiece(self, doc, workpiece):
        """Helper to correctly link a workpiece to a source within a doc."""
        source = SourceAsset(
            Path(workpiece.name),
            original_data=self.svg_data,
            renderer=SVG_RENDERER,
        )
        doc.add_asset(source)
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        workpiece.source_segment = gen_config
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def _complete_all_tasks(
        self, mock_task_mgr, workpiece_handle, step_time=42.0
    ):
        """Helper to find and complete all outstanding tasks."""
        processed_keys = set()
        while True:
            tasks_to_process = [
                t
                for t in mock_task_mgr.created_tasks
                if t.key not in processed_keys
            ]
            if not tasks_to_process:
                break

            for task_info in tasks_to_process:
                task_obj = task_info.returned_task_obj
                task_obj.key = task_info.key
                task_obj.get_status.return_value = "completed"
                task_obj.result.return_value = None

                if task_info.target is make_job_artifact_in_subprocess:
                    if task_info.when_event:
                        store = get_context().artifact_store
                        job_artifact = JobArtifact(ops=Ops(), distance=0.0)
                        job_handle = store.put(job_artifact)
                        event_data = {"handle_dict": job_handle.to_dict()}
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                elif task_info.target is make_step_artifact_in_subprocess:
                    gen_id = task_info.args[2]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        store = get_context().artifact_store
                        render_artifact = StepRenderArtifact()
                        render_handle = store.put(render_artifact)
                        ops_artifact = StepOpsArtifact(ops=Ops())
                        ops_handle = store.put(ops_artifact)

                        render_event = {
                            "handle_dict": render_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj,
                            "render_artifact_ready",
                            render_event,
                        )

                        ops_event = {
                            "handle_dict": ops_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "ops_artifact_ready", ops_event
                        )

                        time_event = {
                            "time_estimate": step_time,
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "time_estimate_ready", time_event
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                elif task_info.target is make_workpiece_artifact_in_subprocess:
                    gen_id = task_info.args[6]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        event_data = {
                            "handle_dict": workpiece_handle.to_dict(),
                            "generation_id": gen_id,
                        }
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                    if task_info.when_done:
                        task_info.when_done(task_obj)

                processed_keys.add(task_info.key)

        mock_task_mgr.created_tasks.clear()

    def test_shutdown_releases_all_artifacts(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Simulate completion of a task to populate the cache
        task_info = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "completed"
        task_obj_for_stage.result.return_value = 1

        try:
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": 1,
            }
            task_info.when_event(
                task_obj_for_stage, "artifact_created", event_data
            )
            task_info.when_done(task_obj_for_stage)

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

    def test_is_busy_property(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Initial state - should be busy with one task
        assert pipeline.is_busy is True

        # Complete the task
        task_info = mock_task_mgr.created_tasks[0]
        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "completed"
        task_obj_for_stage.result.return_value = 1

        # Act
        task_info.when_done(task_obj_for_stage)

        # Assert - should not be busy anymore
        assert pipeline.is_busy is False

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
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
        await asyncio.sleep(0)  # Allow debounced task to run

        # Assert - reconciliation should happen after resume
        mock_task_mgr.run_process.assert_called()

    @pytest.mark.asyncio
    async def test_paused_context_manager(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
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
        await asyncio.sleep(0)  # Allow debounced task to run
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
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests that the refactored get_estimated_time now correctly
        returns None as it's no longer per-workpiece.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        # Act
        result = pipeline.get_estimated_time(step, real_workpiece)

        # Assert
        assert result is None

    def test_preview_time_updated_signal_is_correct(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Tests the new end-to-end time estimation by checking the final
        signal received by the UI.
        """
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
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

    @pytest.mark.asyncio
    async def test_rapid_invalidation_does_not_corrupt_busy_state(
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
        await asyncio.sleep(0)

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
        pipeline._on_descendant_updated(
            sender=step, origin=step, parent_of_origin=layer.workflow
        )
        await asyncio.sleep(0)

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

            # Allow the final scheduled state check to run
            await asyncio.sleep(0)

            # Assert 5: The final signals were emitted correctly.
            mock_artifact_ready_handler.assert_called_once()
            assert pipeline.is_busy is False

            # The callback should have triggered the state change signal.
            mock_processing_state_handler.assert_called_with(
                ANY, is_processing=False
            )
        finally:
            get_context().artifact_store.release(handle)
