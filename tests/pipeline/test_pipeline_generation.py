"""Tests for core pipeline generation and regeneration logic."""

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.import_source import ImportSource
from rayforge.core.ops import Ops
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import (
    WorkPieceArtifact,
    WorkPieceViewArtifact,
    WorkPieceArtifactHandle,
)
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.steps import create_contour_step
from rayforge.shared.tasker.task import Task
from rayforge.workbench.elements.workpiece import WorkPieceElement


# Common fixtures and helpers from the original test_pipeline.py
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
    wp = WorkPiece(name="real_workpiece.svg")
    # Give it a UID for mapping
    wp.uid = "wp_uid_1"
    return wp


@pytest.fixture
def doc():
    d = Doc()
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.mark.usefixtures("context_initializer")
class TestPipelineGeneration:
    """Tests for pipeline generation, cancellation, and regeneration."""

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

    def _complete_all_tasks(self, mock_task_mgr, workpiece_handle):
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
                    if task_info.when_event:
                        event_data = {
                            "handle_dict": workpiece_handle.to_dict(),
                            "generation_id": 1,  # Simulate gen 1
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
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        Pipeline(doc, mock_task_mgr)
        mock_task_mgr.run_process.assert_called_once()
        called_func = mock_task_mgr.run_process.call_args[0][0]
        assert called_func is make_workpiece_artifact_in_subprocess

    def test_generation_success_emits_signals_and_caches_result(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        task_info = mock_task_mgr.created_tasks[0]

        expected_ops = Ops()
        expected_ops.move_to(0, 0, 0)
        expected_ops.line_to(1, 1, 0)
        expected_artifact = WorkPieceArtifact(
            ops=expected_ops,
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=real_workpiece.size,
        )
        handle = get_context().artifact_store.put(expected_artifact)
        gen_id = 1
        task_obj = task_info.returned_task_obj
        task_obj.key = task_info.key
        task_obj.get_status.return_value = "completed"
        task_obj.result.return_value = gen_id

        try:
            event_data = {
                "handle_dict": handle.to_dict(),
                "generation_id": gen_id,
            }
            if task_info.when_event:
                task_info.when_event(task_obj, "artifact_created", event_data)
            if task_info.when_done:
                task_info.when_done(task_obj)

            cached_ops = pipeline.get_ops(step, real_workpiece)
            assert cached_ops is not None and len(cached_ops) == 2
        finally:
            get_context().artifact_store.release(handle)

    def test_generation_cancellation_is_handled(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(doc, mock_task_mgr)
        task_info = mock_task_mgr.created_tasks[0]

        task_obj = task_info.returned_task_obj
        task_obj.get_status.return_value = "cancelled"
        if task_info.when_done:
            task_info.when_done(task_obj)

        assert pipeline.get_ops(step, real_workpiece) is None

    def test_step_change_triggers_full_regeneration(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            step.power = 0.5
            pipeline._on_descendant_updated(sender=step, origin=step)

            workpiece_tasks = [
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_workpiece_artifact_in_subprocess
            ]
            assert len(workpiece_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_workpiece_transform_change_triggers_step_assembly(
        self,
        doc,
        real_workpiece,
        mock_task_mgr,
        context_initializer,
        monkeypatch,
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)

        mock_store = MagicMock()
        mock_store.acquire.return_value = True
        mock_context = MagicMock()
        mock_context.artifact_store = mock_store
        monkeypatch.setattr(
            "rayforge.pipeline.stage.step.get_context", lambda: mock_context
        )

        pipeline = Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = cast(
            WorkPieceArtifactHandle, get_context().artifact_store.put(artifact)
        )
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            pipeline._artifact_cache.put_workpiece_handle(
                step.uid, real_workpiece.uid, handle
            )
            mock_task_mgr.run_process.reset_mock()

            real_workpiece.pos = (50, 50)

            assembly_tasks = [
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_step_artifact_in_subprocess
            ]
            assert len(assembly_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_workpiece_size_change_triggers_regeneration(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        Pipeline(doc, mock_task_mgr)

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            self._complete_all_tasks(mock_task_mgr, handle)
            mock_task_mgr.run_process.reset_mock()

            real_workpiece.set_size(10, 10)

            workpiece_tasks = [
                t
                for t in mock_task_mgr.created_tasks
                if t.target is make_workpiece_artifact_in_subprocess
            ]
            assert len(workpiece_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)

    def test_rapid_step_change_emits_correct_final_signal(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = create_contour_step(context_initializer)
        layer.workflow.add_step(step)
        pipeline = Pipeline(doc, mock_task_mgr)

        mock_signal_handler = MagicMock()
        pipeline.workpiece_artifact_ready.connect(mock_signal_handler)

        task1_info = mock_task_mgr.created_tasks[0]
        mock_task_mgr.created_tasks.clear()

        step.power = 0.5
        pipeline._on_descendant_updated(sender=step, origin=step)

        mock_task_mgr.cancel_task.assert_called_once_with(task1_info.key)
        task2_info = mock_task_mgr.created_tasks[0]

        task1_obj = task1_info.returned_task_obj
        task1_obj.get_status.return_value = "canceled"
        if task1_info.when_done:
            task1_info.when_done(task1_obj)
        mock_signal_handler.assert_not_called()

        artifact = WorkPieceArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )
        handle = get_context().artifact_store.put(artifact)
        try:
            task2_obj = task2_info.returned_task_obj
            task2_obj.get_status.return_value = "completed"
            task2_obj.result.return_value = 2

            if task2_info.when_event:
                event_data = {
                    "handle_dict": handle.to_dict(),
                    "generation_id": 2,  # Simulate gen 2 finishing
                }
                task2_info.when_event(
                    task2_obj, "artifact_created", event_data
                )
            if task2_info.when_done:
                task2_info.when_done(task2_obj)

            mock_signal_handler.assert_called_once()
            _, call_kwargs = mock_signal_handler.call_args
            assert call_kwargs.get("generation_id") == 2
        finally:
            get_context().artifact_store.release(handle)

    def test_view_render_flicker_is_resolved(
        self, real_workpiece, context_initializer
    ):
        """
        Tests that the element correctly updates its internal surfaces when
        the set of view artifacts changes, preventing flicker.
        """
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.workpiece_starting = MagicMock()
        mock_pipeline.workpiece_chunk_available = MagicMock()
        mock_pipeline.workpiece_artifact_ready = MagicMock()
        mock_pipeline.view_artifacts_changed = MagicMock()

        mock_canvas = MagicMock()
        mock_canvas.get_style_context.return_value = MagicMock()

        element = WorkPieceElement(real_workpiece, mock_pipeline)
        element.canvas = mock_canvas
        element.parent = mock_canvas

        step_uid = "test-step-uid"
        artifact_store = get_context().artifact_store
        handles_to_release = []

        try:
            # State 1: Progressive chunks are available
            view_chunk_1 = WorkPieceViewArtifact(
                np.full((10, 10, 4), 1, np.uint8), (0, 0, 10, 10)
            )
            vc1_h = artifact_store.put(view_chunk_1, "vc1")
            view_chunk_2 = WorkPieceViewArtifact(
                np.full((10, 10, 4), 2, np.uint8), (10, 0, 10, 10)
            )
            vc2_h = artifact_store.put(view_chunk_2, "vc2")
            handles_to_release.extend([vc1_h, vc2_h])

            mock_pipeline.get_view_artifacts_for_workpiece.return_value = [
                vc1_h,
                vc2_h,
            ]
            element._on_view_artifacts_changed(
                sender=None,
                step_uid=step_uid,
                workpiece_uid=real_workpiece.uid,
            )

            # Assert state 1: Element has 2 drawable surfaces.
            assert len(element._drawable_surfaces.get(step_uid, [])) == 2

            # State 2: The final artifact is ready.
            final_view = WorkPieceViewArtifact(
                np.zeros((50, 50, 4), dtype=np.uint8), (0, 0, 50, 50)
            )
            final_h = artifact_store.put(final_view, "final")
            handles_to_release.append(final_h)

            mock_pipeline.get_view_artifacts_for_workpiece.return_value = [
                final_h
            ]
            element._on_view_artifacts_changed(
                sender=None,
                step_uid=step_uid,
                workpiece_uid=real_workpiece.uid,
            )

            # Assert state 2: The list of drawables now contains only the
            # final one.
            drawable_list = element._drawable_surfaces.get(step_uid, [])
            assert len(drawable_list) == 1
            # Check that the artifact associated with the surface has the
            # same data.
            surface, artifact = drawable_list[0]
            assert artifact.bbox_mm == final_view.bbox_mm
            assert np.array_equal(artifact.bitmap_data, final_view.bitmap_data)

        finally:
            element.remove()
            for handle in handles_to_release:
                artifact_store.release(handle)
