import pytest
import threading
import uuid
from unittest.mock import MagicMock
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.geo import Geometry
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact.key import ArtifactKey
from rayforge.pipeline.context import GenerationContext
from rayforge.pipeline.pipeline import Pipeline
from rayforge.pipeline.steps import create_contour_step
from rayforge.shared.tasker.task import Task
from pathlib import Path


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
        mock_returned_task = MagicMock(spec=Task)
        mock_returned_task.key = kwargs.get("key")
        mock_returned_task.id = id(mock_returned_task)
        mock_returned_task._cancelled = False
        mock_returned_task.is_running.return_value = False

        task = MockTask(target_func, args, kwargs, mock_returned_task)
        created_tasks_info.append(task)
        return mock_returned_task

    def get_task_mock(task_key):
        for task in created_tasks_info:
            if task.key == task_key:
                return task.returned_task_obj
        return None

    def cancel_task_mock(task_key):
        for task in created_tasks_info:
            if task.key == task_key:
                task.returned_task_obj._cancelled = True

    def schedule_awarely(callback, *args, **kwargs):
        callback(*args, **kwargs)

    mock_mgr.run_process = MagicMock(side_effect=run_process_mock)
    mock_mgr.get_task = MagicMock(side_effect=get_task_mock)
    mock_mgr.cancel_task = MagicMock(side_effect=cancel_task_mock)
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
            self._cancelled = False

    monkeypatch.setattr(threading, "Timer", SyncTimer)


@pytest.fixture
def doc():
    d = Doc()
    active_layer = d.active_layer
    assert active_layer.workflow is not None
    active_layer.workflow.set_steps([])
    return d


@pytest.fixture
def real_workpiece():
    """Creates a lightweight WorkPiece with transforms, but no source."""
    workpiece = WorkPiece(name="real_workpiece.svg")
    return workpiece


@pytest.mark.usefixtures("context_initializer")
class TestPipelineContextIntegration:
    """Test suite for Step 2: Context integration into Pipeline lifecycle."""

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
            pristine_geometry=Geometry(),
            vectorization_spec=PassthroughSpec(),
        )
        workpiece.source_segment = gen_config
        workpiece.set_size(50, 30)
        workpiece.pos = 10, 20
        doc.active_layer.add_workpiece(workpiece)
        return doc.active_layer

    def test_reconcile_data_creates_context(self, doc, mock_task_mgr):
        """
        Test that reconcile_data creates a new GenerationContext.
        """
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert pipeline._active_context is not None
        assert pipeline._active_context.generation_id == 1
        assert 1 in pipeline._contexts
        assert pipeline._contexts[1] is pipeline._active_context

    def test_reconcile_data_creates_incrementing_context_ids(
        self, doc, mock_task_mgr, real_workpiece
    ):
        """
        Test that multiple reconcile_data calls create contexts with
        incrementing IDs.
        """
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        initial_id = pipeline._data_generation_id
        assert pipeline._active_context is not None
        assert pipeline._active_context.generation_id == initial_id

        pipeline.pause()
        step = create_contour_step(get_context())
        doc.active_layer.workflow.add_step(step)
        self._setup_doc_with_workpiece(doc, real_workpiece)

        pipeline._reconciliation_timer = MagicMock()
        pipeline._reconciliation_timer.cancel = MagicMock()
        pipeline.resume()

        next_id = pipeline._data_generation_id
        assert next_id >= initial_id

        pipeline.reconcile_data()

        assert pipeline._data_generation_id > next_id
        assert pipeline._active_context is not None
        assert (
            pipeline._active_context.generation_id
            == pipeline._data_generation_id
        )

        gen_ids = list(pipeline._contexts.keys())
        assert gen_ids == sorted(gen_ids)

    def test_scheduler_receives_context(self, doc, mock_task_mgr):
        """
        Test that the scheduler receives the active context.
        """
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert pipeline._scheduler._active_context is not None
        assert pipeline._scheduler._active_context is pipeline._active_context
        assert pipeline._active_context is not None
        assert (
            pipeline._scheduler._generation_id
            == pipeline._active_context.generation_id
        )

    def test_context_generation_id_matches_data_generation_id(
        self, doc, mock_task_mgr, real_workpiece
    ):
        """
        Test that the context's generation_id matches the pipeline's
        _data_generation_id (preserving the old integer ID logic).
        """
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert pipeline._active_context is not None
        assert (
            pipeline._active_context.generation_id
            == pipeline._data_generation_id
        )

        pipeline.pause()
        step = create_contour_step(get_context())
        doc.active_layer.workflow.add_step(step)
        self._setup_doc_with_workpiece(doc, real_workpiece)

        pipeline._reconciliation_timer = MagicMock()
        pipeline._reconciliation_timer.cancel = MagicMock()
        pipeline.resume()

        pipeline.reconcile_data()
        assert pipeline._active_context is not None
        assert (
            pipeline._active_context.generation_id
            == pipeline._data_generation_id
        )

    def test_old_contexts_preserved(self, doc, mock_task_mgr, real_workpiece):
        """
        Test that old contexts are preserved when new ones are created.
        """
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        first_context = pipeline._active_context
        first_gen_id = pipeline._data_generation_id

        pipeline.pause()
        step = create_contour_step(get_context())
        doc.active_layer.workflow.add_step(step)
        self._setup_doc_with_workpiece(doc, real_workpiece)

        pipeline._reconciliation_timer = MagicMock()
        pipeline._reconciliation_timer.cancel = MagicMock()
        pipeline.resume()

        pipeline.reconcile_data()

        assert first_gen_id in pipeline._contexts
        assert pipeline._contexts[first_gen_id] is first_context
        assert pipeline._data_generation_id in pipeline._contexts
        assert (
            pipeline._contexts[pipeline._data_generation_id]
            is pipeline._active_context
        )
        assert first_context is not pipeline._active_context


@pytest.mark.usefixtures("context_initializer")
class TestPipelineBusyState:
    """Test suite for Step 9: Busy state logic with inactive contexts."""

    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def test_is_busy_false_when_no_active_tasks(self, doc, mock_task_mgr):
        """Test that is_busy is False when no tasks are active."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert pipeline._reconciliation_timer is None
        assert not pipeline._scheduler.has_pending_work()
        assert not pipeline.is_busy

    def test_is_busy_true_with_inactive_context_tasks(
        self, doc, mock_task_mgr
    ):
        """Test is_busy is True when inactive context has active tasks."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        old_context = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        old_context.add_task(key)
        pipeline._contexts[1] = old_context
        pipeline._active_context = GenerationContext(generation_id=2)
        pipeline._contexts[2] = pipeline._active_context

        assert not pipeline._scheduler.has_pending_work()
        assert pipeline.is_busy

    def test_is_busy_false_when_inactive_context_empty(
        self, doc, mock_task_mgr
    ):
        """Test is_busy is False when inactive context has no tasks."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        old_context = GenerationContext(generation_id=1)
        pipeline._contexts[1] = old_context
        pipeline._active_context = GenerationContext(generation_id=2)
        pipeline._contexts[2] = pipeline._active_context

        assert not pipeline._scheduler.has_pending_work()
        assert not pipeline.is_busy

    def test_is_busy_true_with_reconciliation_timer(self, doc, mock_task_mgr):
        """Test is_busy is True when reconciliation timer is pending."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        pipeline._reconciliation_timer = MagicMock()

        assert pipeline.is_busy

    def test_is_busy_checks_all_inactive_contexts(self, doc, mock_task_mgr):
        """Test is_busy checks all inactive contexts for active tasks."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        ctx1 = GenerationContext(generation_id=1)
        ctx2 = GenerationContext(generation_id=2)
        ctx3 = GenerationContext(generation_id=3)

        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        ctx2.add_task(key)

        pipeline._contexts[1] = ctx1
        pipeline._contexts[2] = ctx2
        pipeline._contexts[3] = ctx3
        pipeline._active_context = ctx3

        assert pipeline.is_busy

    def test_is_busy_active_context_tasks_ignored(self, doc, mock_task_mgr):
        """Test is_busy ignores active tasks in the active context."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        active_ctx = GenerationContext(generation_id=1)
        key = ArtifactKey.for_workpiece(str(uuid.uuid4()))
        active_ctx.add_task(key)
        pipeline._contexts[1] = active_ctx
        pipeline._active_context = active_ctx

        assert not pipeline._scheduler.has_pending_work()
        assert not pipeline.is_busy
