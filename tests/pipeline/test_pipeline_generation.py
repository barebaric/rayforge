import pytest
import logging
import numpy as np
from unittest.mock import MagicMock, ANY
from pathlib import Path
import asyncio
import threading
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
    VertexData,
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
class TestPipelineGeneration:
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

    def test_reconcile_all_triggers_ops_generation(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        # Act
        Pipeline(doc, mock_task_mgr)

        # Assert
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_workpiece_artifact_in_subprocess

    def test_generation_success_emits_signals_and_caches_result(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        task_info = mock_task_mgr.created_tasks[0]

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
        gen_id = 1

        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "completed"
        task_obj_for_stage.result.return_value = gen_id

        try:
            # Simulate the new two-step flow: event first, then completion
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": gen_id,
            }
            task_info.when_event(
                task_obj_for_stage, "artifact_created", event_data
            )
            task_info.when_done(task_obj_for_stage)

            cached_ops = pipeline.get_ops(step, real_workpiece)
            assert cached_ops is not None
            assert len(cached_ops) == 2
        finally:
            get_context().artifact_store.release(handle)

    def test_generation_cancellation_is_handled(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        task_info = mock_task_mgr.created_tasks[0]

        # Act
        task_obj_for_stage = task_info.returned_task_obj
        task_obj_for_stage.key = task_info.key
        task_obj_for_stage.get_status.return_value = "cancelled"
        task_info.when_done(task_obj_for_stage)

        # Assert
        assert pipeline.get_ops(step, real_workpiece) is None

    @pytest.mark.asyncio
    async def test_step_change_triggers_full_regeneration(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
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
            step.power = 0.5
            pipeline._on_descendant_updated(
                sender=step, origin=step, parent_of_origin=layer.workflow
            )
            await asyncio.sleep(0)  # Allow debounced task to run

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

    @pytest.mark.asyncio
    async def test_workpiece_transform_change_triggers_step_assembly(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        Pipeline(doc, mock_task_mgr)
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
            await asyncio.sleep(0)  # Allow debounced task to run

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

    @pytest.mark.asyncio
    async def test_workpiece_size_change_triggers_regeneration(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        Pipeline(doc, mock_task_mgr)
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
            await asyncio.sleep(0)  # Allow debounced task to run

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

    def test_generate_job_artifact_callback_success(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange: Setup a complete pipeline state
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # First, complete the prerequisite workpiece and step generation
        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        expected_job_handle = None
        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle)
            mock_task_mgr.run_process.reset_mock()
            mock_task_mgr.created_tasks.clear()

            callback_mock = MagicMock()
            store = get_context().artifact_store
            job_artifact = JobArtifact(ops=Ops(), distance=0)
            expected_job_handle = store.put(job_artifact)

            # Act
            pipeline.generate_job_artifact(when_done=callback_mock)

            # Assert a job task was created
            mock_task_mgr.run_process.assert_called_once()
            job_task_info = next(
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_job_artifact_in_subprocess
            )

            # Simulate the job task completing successfully
            job_task_obj = job_task_info.returned_task_obj
            job_task_obj.key = job_task_info.key
            job_task_obj.get_status.return_value = "completed"
            job_task_obj.result.return_value = None

            # 1. Simulate the event that puts the handle in the cache
            job_task_info.when_event(
                job_task_obj,
                "artifact_created",
                {"handle_dict": expected_job_handle.to_dict()},
            )
            # 2. Simulate the final completion callback
            job_task_info.when_done(job_task_obj)

            # Assert
            callback_mock.assert_called_once_with(expected_job_handle, None)
        finally:
            get_context().artifact_store.release(wp_handle)
            if expected_job_handle:
                get_context().artifact_store.release(expected_job_handle)

    def test_generate_job_artifact_callback_failure(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle)
            mock_task_mgr.run_process.reset_mock()
            mock_task_mgr.created_tasks.clear()

            callback_mock = MagicMock()

            # Act
            pipeline.generate_job_artifact(when_done=callback_mock)
            mock_task_mgr.run_process.assert_called_once()

            # Find the when_done callback captured by the mock task manager
            job_task_info = next(
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_job_artifact_in_subprocess
            )
            when_done_callback = job_task_info.when_done

            # Create a realistic mock of a failed task object
            mock_failed_task = job_task_info.returned_task_obj
            mock_failed_task.get_status.return_value = "failed"
            # When result() is called on a failed task, it should raise.
            mock_failed_task.result.side_effect = RuntimeError(
                "Job generation failed."
            )

            # Directly invoke the captured callback with the mock failed task
            when_done_callback(mock_failed_task)

            # Assert the user's callback receives (None, <Error>)
            callback_mock.assert_called_once_with(None, ANY)
            # Further inspect the error argument
            args, kwargs = callback_mock.call_args
            error_arg = args[1]
            assert isinstance(error_arg, RuntimeError)
            assert "Job generation failed" in str(error_arg)
        finally:
            get_context().artifact_store.release(wp_handle)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_success(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        expected_job_handle = None
        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle)
            mock_task_mgr.run_process.reset_mock()
            mock_task_mgr.created_tasks.clear()

            store = get_context().artifact_store
            job_artifact = JobArtifact(ops=Ops(), distance=0)
            expected_job_handle = store.put(job_artifact)

            # Act
            future = asyncio.create_task(
                pipeline.generate_job_artifact_async()
            )
            await asyncio.sleep(0)  # Allow the event loop to run

            # The task should have been created
            mock_task_mgr.run_process.assert_called_once()
            job_task_info = next(
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_job_artifact_in_subprocess
            )

            # Simulate completion
            job_task_obj = job_task_info.returned_task_obj
            job_task_obj.key = job_task_info.key
            job_task_obj.get_status.return_value = "completed"
            job_task_obj.result.return_value = None

            job_task_info.when_event(
                job_task_obj,
                "artifact_created",
                {"handle_dict": expected_job_handle.to_dict()},
            )
            job_task_info.when_done(job_task_obj)

            # Now await the result
            result_handle = await future

            # Assert
            assert result_handle == expected_job_handle
        finally:
            get_context().artifact_store.release(wp_handle)
            if expected_job_handle:
                get_context().artifact_store.release(expected_job_handle)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_failure(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle)
            mock_task_mgr.run_process.reset_mock()
            mock_task_mgr.created_tasks.clear()

            # Act & Assert
            async def _failure_simulator_task():
                await asyncio.sleep(0)
                job_task_info = next(
                    t
                    for t in mock_task_mgr.created_tasks
                    if t.target is make_job_artifact_in_subprocess
                )
                # Simulate task failure by directly invoking the callback
                mock_failed_task = job_task_info.returned_task_obj
                mock_failed_task.get_status.return_value = "failed"
                mock_failed_task.result.side_effect = RuntimeError(
                    "Job failed."
                )
                if job_task_info.when_done:
                    job_task_info.when_done(mock_failed_task)

            # Act & Assert
            with pytest.raises(RuntimeError, match="Job failed."):
                failure_future = asyncio.create_task(_failure_simulator_task())
                # This will start the job and the future will be populated
                # by the _when_done_callback created by the async method.
                await pipeline.generate_job_artifact_async()
                await failure_future  # ensure the simulator ran
        finally:
            get_context().artifact_store.release(wp_handle)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_already_running(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        # Arrange
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle)
            mock_task_mgr.run_process.reset_mock()
            mock_task_mgr.created_tasks.clear()

            # Act
            # Start the first generation, but don't complete it
            future1 = asyncio.create_task(
                pipeline.generate_job_artifact_async()
            )
            await asyncio.sleep(0)  # Allow the event loop to run

            mock_task_mgr.run_process.assert_called_once()

            # Try to start a second one while the first is 'running'
            with pytest.raises(
                RuntimeError, match="Job generation is already in progress."
            ):
                await pipeline.generate_job_artifact_async()

            # Cleanup: complete the first task to avoid leaving it hanging
            job_task_info = next(
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_job_artifact_in_subprocess
            )
            job_task_obj = job_task_info.returned_task_obj
            job_task_obj.key = job_task_info.key
            job_task_obj.get_status.return_value = "completed"
            # Simulate an empty job result for cleanup
            job_task_info.when_done(job_task_obj)
            await future1  # consume the result
        finally:
            get_context().artifact_store.release(wp_handle)

    @pytest.mark.asyncio
    async def test_rapid_step_change_emits_correct_final_signal(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """
        Simulates a user changing a step setting twice in quick succession.
        This test verifies that the pipeline correctly cancels the first task,
        processes the second task, and emits the `workpiece_artifact_ready`
        signal exactly once with the correct, final generation ID.
        """
        # Arrange: Setup doc, workpiece, step, and pipeline
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)

        # Mock the final signal handler to intercept the call
        mock_signal_handler = MagicMock()
        pipeline.workpiece_artifact_ready.connect(mock_signal_handler)

        # Act 1: The initial pipeline creation starts the first task.
        await asyncio.sleep(0)
        mock_task_mgr.run_process.assert_called_once()
        assert len(mock_task_mgr.created_tasks) == 1
        task1_info = mock_task_mgr.created_tasks[0]
        # Generation ID for the first task is 1
        assert task1_info.args[6] == 1
        mock_task_mgr.run_process.reset_mock()
        mock_task_mgr.created_tasks.clear()

        # Act 2: Trigger a second regeneration immediately.
        # This simulates a rapid UI change, cancelling task1 and
        # starting task2.
        step.power = 0.5  # Change a property to trigger invalidation
        pipeline._on_descendant_updated(
            sender=step, origin=step, parent_of_origin=layer.workflow
        )
        await asyncio.sleep(0)

        # Assert 2: A new task was created, and the old one was cancelled.
        mock_task_mgr.run_process.assert_called_once()
        mock_task_mgr.cancel_task.assert_called_once_with(task1_info.key)
        assert len(mock_task_mgr.created_tasks) == 1
        task2_info = mock_task_mgr.created_tasks[0]
        # Generation ID for the second task should be incremented to 2
        assert task2_info.args[6] == 2

        # Act 3: Simulate the CANCELLED task's callback firing.
        # This could happen if the task was already running when cancelled.
        task1_obj = task1_info.returned_task_obj
        task1_obj.key = task1_info.key
        task1_obj.get_status.return_value = "canceled"
        if task1_info.when_done:
            task1_info.when_done(task1_obj)

        # Assert 3: The signal handler should NOT have been called for the
        # cancelled task.
        mock_signal_handler.assert_not_called()

        # Act 4: Simulate the SUCCESSFUL task's (task2) callbacks firing.
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            generation_size=real_workpiece.size,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            task2_obj = task2_info.returned_task_obj
            task2_obj.key = task2_info.key
            task2_obj.get_status.return_value = "completed"
            task2_obj.result.return_value = 2  # Gen ID from task2

            # Simulate the 'artifact_created' event from task2
            if task2_info.when_event:
                event_data = {
                    "handle_dict": handle.to_dict(),
                    "generation_id": 2,
                }
                task2_info.when_event(
                    task2_obj, "artifact_created", event_data
                )

            # Simulate the final 'when_done' callback for task2
            if task2_info.when_done:
                task2_info.when_done(task2_obj)

            # Assert 4: The signal handler was called exactly once with the
            # correct generation ID from the second, successful task.
            mock_signal_handler.assert_called_once()
            call_args, call_kwargs = mock_signal_handler.call_args
            assert call_args[0] is step
            assert call_kwargs.get("workpiece") is real_workpiece
            assert call_kwargs.get("generation_id") == 2

        finally:
            get_context().artifact_store.release(handle)
