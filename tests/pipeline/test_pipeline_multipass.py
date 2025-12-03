"""Test that multipass transformer changes trigger re-generation."""

from unittest.mock import MagicMock, patch
import pytest
import logging
import asyncio
import threading
from pathlib import Path
from rayforge.shared.tasker.task import Task
from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.image import SVG_RENDERER
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.geo import Geometry
from rayforge.core.ops import Ops
from rayforge.pipeline.coord import CoordinateSystem
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
class TestPipelineMultipass:
    """Test that changes to multipass transformer trigger re-generation."""

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

    @pytest.mark.asyncio
    async def test_multipass_change_triggers_step_assembly(
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
            step.per_step_transformers_dicts = []
            pipeline._on_job_assembly_invalidated(sender=doc)
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
