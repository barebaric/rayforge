"""Tests for retrieving pipeline artifacts, ops, and time estimates."""

import pytest
import logging
from unittest.mock import MagicMock
from pathlib import Path
import asyncio
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.ops import Ops
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    WorkPieceArtifactHandle,
    StepRenderArtifact,
    StepOpsArtifact,
    JobArtifact,
)
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.stage.job_runner import make_job_artifact_in_subprocess
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.steps import create_contour_step
from rayforge.shared.tasker.task import Task

logger = logging.getLogger(__name__)


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
class TestPipelineArtifacts:
    """Tests for retrieving results from the pipeline."""

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
                if task_info.target is make_workpiece_artifact_in_subprocess:
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
                elif task_info.target is make_step_artifact_in_subprocess:
                    gen_id = task_info.args[2]
                    task_obj.result.return_value = gen_id
                    if task_info.when_event:
                        store = get_context().artifact_store
                        render_handle = store.put(StepRenderArtifact())
                        ops_handle = store.put(StepOpsArtifact(ops=Ops()))
                        task_info.when_event(
                            task_obj,
                            "render_artifact_ready",
                            {
                                "handle_dict": render_handle.to_dict(),
                                "generation_id": gen_id,
                            },
                        )
                        task_info.when_event(
                            task_obj,
                            "ops_artifact_ready",
                            {
                                "handle_dict": ops_handle.to_dict(),
                                "generation_id": gen_id,
                            },
                        )
                        task_info.when_event(
                            task_obj,
                            "time_estimate_ready",
                            {
                                "time_estimate": step_time,
                                "generation_id": gen_id,
                            },
                        )
                elif task_info.target is make_job_artifact_in_subprocess:
                    if task_info.when_event:
                        store = get_context().artifact_store
                        job_handle = store.put(
                            JobArtifact(ops=Ops(), distance=0.0)
                        )
                        event_data = {"handle_dict": job_handle.to_dict()}
                        task_info.when_event(
                            task_obj, "artifact_created", event_data
                        )
                if task_info.when_done:
                    task_info.when_done(task_obj)
                processed_keys.add(task_info.key)

    def test_get_estimated_time_returns_none(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)
        assert pipeline.get_estimated_time(step, real_workpiece) is None

    def test_job_time_updated_signal_is_correct(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=real_workpiece.size,
        )
        wp_handle = get_context().artifact_store.put(wp_artifact)
        mock_handler = MagicMock()
        pipeline.job_time_updated.connect(mock_handler)

        try:
            self._complete_all_tasks(mock_task_mgr, wp_handle, step_time=55.5)
            mock_handler.assert_called()
            _, last_call_kwargs = mock_handler.call_args_list[-1]
            assert last_call_kwargs.get("total_seconds") == 55.5
        finally:
            get_context().artifact_store.release(wp_handle)

    def test_get_artifact_and_handle(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        assert (
            pipeline.get_artifact_handle(step.uid, real_workpiece.uid) is None
        )
        assert pipeline.get_artifact(step, real_workpiece) is None

        task_info = mock_task_mgr.created_tasks[0]
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(artifact)
        task_obj = task_info.returned_task_obj
        task_obj.get_status.return_value = "completed"
        task_obj.result.return_value = 1
        try:
            if task_info.when_event:
                event_data = {
                    "handle_dict": handle.to_dict(),
                    "generation_id": 1,
                }
                task_info.when_event(task_obj, "artifact_created", event_data)
            if task_info.when_done:
                task_info.when_done(task_obj)

            retrieved_handle = pipeline.get_artifact_handle(
                step.uid, real_workpiece.uid
            )
            assert isinstance(retrieved_handle, WorkPieceArtifactHandle)
            retrieved_artifact = pipeline.get_artifact(step, real_workpiece)
            assert retrieved_artifact is not None
        finally:
            get_context().artifact_store.release(handle)

    def test_get_artifact_with_stale_non_scalable_artifact(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        original_size = (25, 15)
        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=original_size,
        )
        handle = get_context().artifact_store.put(artifact)
        task_info = mock_task_mgr.created_tasks[0]
        task_obj = task_info.returned_task_obj
        task_obj.get_status.return_value = "completed"

        try:
            if task_info.when_event:
                event_data = {
                    "handle_dict": handle.to_dict(),
                    "generation_id": 1,
                }
                task_info.when_event(task_obj, "artifact_created", event_data)
            if task_info.when_done:
                task_info.when_done(task_obj)

            assert pipeline.get_artifact(step, real_workpiece) is None
        finally:
            get_context().artifact_store.release(handle)

    def test_generate_job_artifact_callback_success(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=real_workpiece.size,
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

            pipeline.generate_job_artifact(when_done=callback_mock)

            job_task_info = next(t for t in mock_task_mgr.created_tasks)
            job_task_obj = job_task_info.returned_task_obj
            job_task_obj.get_status.return_value = "completed"

            if job_task_info.when_event:
                job_task_info.when_event(
                    job_task_obj,
                    "artifact_created",
                    {"handle_dict": expected_job_handle.to_dict()},
                )
            if job_task_info.when_done:
                job_task_info.when_done(job_task_obj)

            callback_mock.assert_called_once_with(expected_job_handle, None)
        finally:
            get_context().artifact_store.release(wp_handle)
            if expected_job_handle:
                get_context().artifact_store.release(expected_job_handle)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_success(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        wp_artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            generation_size=real_workpiece.size,
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

            future = asyncio.create_task(
                pipeline.generate_job_artifact_async()
            )
            await asyncio.sleep(0)

            job_task_info = next(t for t in mock_task_mgr.created_tasks)
            job_task_obj = job_task_info.returned_task_obj
            job_task_obj.get_status.return_value = "completed"

            if job_task_info.when_event:
                job_task_info.when_event(
                    job_task_obj,
                    "artifact_created",
                    {"handle_dict": expected_job_handle.to_dict()},
                )
            if job_task_info.when_done:
                job_task_info.when_done(job_task_obj)

            result_handle = await future
            assert result_handle == expected_job_handle
        finally:
            get_context().artifact_store.release(wp_handle)
            if expected_job_handle:
                get_context().artifact_store.release(expected_job_handle)
