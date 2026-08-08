import asyncio
import logging
from pathlib import Path

import pytest
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.artifact import JobArtifact
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
class TestPipelineGeneration:
    """Test ops generation, signal emission, and caching in the
    raygeo-backed pipeline."""

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
    async def test_generation_success_produces_artifact_handle(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that after generation completes, a workpiece artifact
        handle is available via get_artifact_handle."""
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

        handle = pipeline.get_artifact_handle(step.uid, real_workpiece.uid)
        assert handle is not None

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_step_change_triggers_full_regeneration(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that changing a step power triggers a new rebuild."""
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
        gen1 = pipeline.data_generation_id

        step.set_power(0.5)

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        gen2 = pipeline.data_generation_id
        assert gen2 > gen1, "Step change should trigger a new generation"

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_workpiece_position_change_triggers_regeneration(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that changing a workpiece position triggers a
        regeneration for position-sensitive steps."""
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
        gen1 = pipeline.data_generation_id

        real_workpiece.pos = 50, 50

        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        gen2 = pipeline.data_generation_id
        assert gen2 > gen1, "Position change should trigger a new generation"

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_callback_success(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that generate_job_artifact invokes when_done with
        a valid handle on success."""
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

        result = []

        def when_done(handle, error):
            result.append((handle, error))

        pipeline.generate_job_artifact(when_done=when_done)

        deadline = asyncio.get_running_loop().time() + 10.0
        while not result and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert len(result) == 1
        handle, error = result[0]
        assert error is None
        assert handle is not None

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_success(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that generate_job_artifact_async returns the handle."""
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

        handle = await asyncio.wait_for(
            pipeline.generate_job_artifact_async(), timeout=10
        )
        assert handle is not None

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_generate_job_artifact_async_already_running(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Verifies that a second call returns the cached handle
        without starting a duplicate generation."""
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

        handle1 = await asyncio.wait_for(
            pipeline.generate_job_artifact_async(), timeout=10
        )
        handle2 = await asyncio.wait_for(
            pipeline.generate_job_artifact_async(), timeout=10
        )
        assert handle1 is handle2

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_rapid_step_change_emits_correct_final_generation(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Simulates two rapid property changes on a step, verifying
        that the pipeline settles to a correct final generation."""
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

        # Let the first rebuild start
        await asyncio.sleep(0.1)

        # Rapidly change step power twice
        step.set_power(0.3)
        step.set_power(0.7)

        # Wait for everything to settle
        deadline = asyncio.get_running_loop().time() + 10.0
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

        assert pipeline.is_busy is False

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_job_generation_produces_encoded_output(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Integration test: full pipeline produces an encoded job
        artifact."""
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

        handle = await asyncio.wait_for(
            pipeline.generate_job_artifact_async(), timeout=10
        )
        assert handle is not None

        with pipeline.artifact_store.checkout_handle(handle) as artifact:
            assert artifact is not None
            assert isinstance(artifact, JobArtifact)
            assert artifact.encoded_output is not None
            assert artifact.ops is not None

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)
