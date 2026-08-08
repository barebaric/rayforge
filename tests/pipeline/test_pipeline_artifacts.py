import asyncio
import logging
from pathlib import Path

import pytest
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.step import Step
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import WorkPieceArtifactHandle
from rayforge.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    """Apply zero debounce delay to all tests in this file."""


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
class TestPipelineArtifacts:
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

    @pytest.mark.asyncio
    async def test_get_artifact_handle(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies retrieving a WorkPieceArtifactHandle after
        generation completes."""
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        # Wait for pipeline to settle
        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        handle = pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
        assert handle is not None
        assert isinstance(handle, WorkPieceArtifactHandle)
        assert handle.generation_size == real_workpiece.size

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_get_artifact(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies retrieving the full WorkPieceArtifact after
        generation completes."""
        layer = self._setup_doc_with_workpiece(doc, real_workpiece)
        assert layer.workflow is not None
        step = contour_step_class.create(context_initializer)
        layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        artifact = pipeline.get_artifact(step, real_workpiece)
        assert artifact is not None
        assert artifact.generation_id >= 1

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_get_artifact_handle_none_before_generation(
        self,
        doc,
        real_workpiece,
        mock_task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies get_artifact_handle returns None when no step/workpiece
        has been added yet."""
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        assert (
            pipeline.get_artifact_handle("nonexistent", "nonexistent") is None
        )

    @pytest.mark.asyncio
    async def test_get_artifact_none_before_generation(
        self,
        doc,
        real_workpiece,
        mock_task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies get_artifact returns None when no step/workpiece
        has been added yet."""
        pipeline = Pipeline(
            doc,
            mock_task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )

        step = Step(typelabel="test")
        assert pipeline.get_artifact(step, real_workpiece) is None
