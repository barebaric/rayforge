import logging
from pathlib import Path

import pytest
from raygeo.geo import Geometry

from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    """Apply zero debounce delay to all tests in this file."""


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
    """Test generation ID and rebuild lifecycle in the new pipeline."""

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

    def test_pipeline_creates_initial_generation(self, doc, mock_task_mgr):
        """Test that pipeline creation sets up an initial generation ID."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert pipeline.data_generation_id >= 0

    def test_rebuild_increments_generation_id(
        self, doc, mock_task_mgr, real_workpiece, contour_step_class
    ):
        """Test that triggering a rebuild increments the generation ID."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        initial_id = pipeline.data_generation_id

        pipeline._intent_ctl.force_rebuild()

        assert pipeline.data_generation_id > initial_id

    def test_generation_id_matches_intent_controller(self, doc, mock_task_mgr):
        """Test that pipeline.data_generation_id matches the
        IntentController's generation_id."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert (
            pipeline.data_generation_id == pipeline._intent_ctl.generation_id
        )

    def test_multiple_rebuilds_produce_incrementing_ids(
        self, doc, mock_task_mgr
    ):
        """Test that multiple rebuilds produce incrementing generation IDs."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        ids = [pipeline.data_generation_id]
        for _ in range(3):
            pipeline._intent_ctl.force_rebuild()
            ids.append(pipeline.data_generation_id)

        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)


@pytest.mark.usefixtures("context_initializer")
class TestPipelineBusyState:
    """Test busy state logic in the new pipeline."""

    def test_is_busy_false_when_idle(self, doc, mock_task_mgr):
        """Test that is_busy is False when no tasks are active."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        assert not pipeline._intent_ctl.is_rebuild_pending
        assert not pipeline.is_busy

    def test_is_busy_true_when_rebuild_pending(self, doc, mock_task_mgr):
        """Test that is_busy is True when a rebuild is pending."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        pipeline._intent_ctl._rebuild_timer = object()
        assert pipeline._intent_ctl.is_rebuild_pending
        assert pipeline.is_busy

    def test_is_busy_false_after_rebuild_completes(self, doc, mock_task_mgr):
        """Test that is_busy returns to False after a rebuild completes."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        pipeline._intent_ctl._rebuild_timer = object()
        assert pipeline.is_busy

        pipeline._intent_ctl._rebuild_timer = None
        assert not pipeline.is_busy

    def test_is_busy_true_when_task_manager_has_tasks(
        self, doc, mock_task_mgr
    ):
        """Test that is_busy is True when the task manager has tasks."""
        pipeline = Pipeline(
            doc=doc,
            task_manager=mock_task_mgr,
            artifact_store=get_context().artifact_store,
            machine=get_context().machine,
        )

        mock_task_mgr.created_tasks.append(object())
        assert pipeline.is_busy

        mock_task_mgr.created_tasks.clear()
        assert not pipeline.is_busy
