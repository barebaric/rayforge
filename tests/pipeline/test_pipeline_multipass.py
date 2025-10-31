"""Tests for how MultiPassTransformer changes affect the pipeline."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
import asyncio
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops
from rayforge.core.import_source import ImportSource
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.stage.step_runner import (
    make_step_artifact_in_subprocess,
)
from rayforge.pipeline.stage.workpiece_runner import (
    make_workpiece_artifact_in_subprocess,
)
from rayforge.pipeline.steps import create_contour_step
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
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
class TestMultipassRegeneration:
    """Test that changes to multipass transformer trigger re-generation."""

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

    def test_job_assembly_invalidated_signal_connected(self):
        """Test pipeline connects to the job_assembly_invalidated signal."""
        doc = Doc()
        layer = Layer(name="Test Layer")
        step = Step(typelabel="Test Step", name="Test Step")
        workpiece = WorkPiece(name="Test Workpiece")

        doc.add_child(layer)
        assert layer.workflow is not None
        layer.workflow.add_child(step)
        layer.add_child(workpiece)

        task_manager = MagicMock()

        with patch("rayforge.pipeline.pipeline.logger"):
            handler_called = MagicMock()

            def track_handler(sender):
                handler_called()

            pipeline = Pipeline(doc, task_manager)
            pipeline._on_job_assembly_invalidated = track_handler

        doc.job_assembly_invalidated.disconnect(
            pipeline._on_job_assembly_invalidated
        )
        doc.job_assembly_invalidated.connect(track_handler)

        doc.job_assembly_invalidated.send(doc)
        assert handler_called.called

    def test_multipass_change_sends_job_assembly_invalidated(self):
        """Test multipass changes send job_assembly_invalidated signal."""
        doc = Doc()
        layer = Layer(name="Test Layer")
        step = Step(typelabel="Test Step", name="Test Step")
        workpiece = WorkPiece(name="Test Workpiece")

        doc.add_child(layer)
        assert layer.workflow is not None
        layer.workflow.add_child(step)
        layer.add_child(workpiece)

        step.per_step_transformers_dicts = [
            MultiPassTransformer(
                enabled=True, passes=2, z_step_down=0.5
            ).to_dict()
        ]

        doc.job_assembly_invalidated = MagicMock()

        new_multipass = MultiPassTransformer(
            enabled=True, passes=3, z_step_down=1.0
        )
        step.per_step_transformers_dicts = [new_multipass.to_dict()]
        step.per_step_transformer_changed.send(step)
        assert doc.job_assembly_invalidated.send.called

    def test_multipass_change_triggers_step_assembly(
        self, doc, real_workpiece, mock_task_mgr, context_initializer
    ):
        """Test a multipass change triggers step artifact regeneration."""
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
            mock_task_mgr.created_tasks.clear()
            mock_task_mgr.run_process.reset_mock()

            # Act: Change multipass settings and manually fire the signal
            # that the pipeline listens for.
            step.per_step_transformers_dicts = []
            pipeline._on_job_assembly_invalidated(sender=doc)

            # Assert: A new step assembly task was created.
            tasks = mock_task_mgr.created_tasks
            assembly_tasks = [
                t
                for t in tasks
                if t.target is make_step_artifact_in_subprocess
            ]
            assert len(assembly_tasks) == 1
        finally:
            get_context().artifact_store.release(handle)
