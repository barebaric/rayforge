"""
Tests for pipeline invalidation behavior.

In the new raygeo-backed pipeline, invalidation is driven by version
tokens in the :class:`IntentBuilder`.  When a workpiece or step changes,
the :class:`IntentController` schedules a debounced rebuild.  Each
rebuild increments the generation ID.  These tests verify that the
correct changes trigger rebuilds through the public API.
"""

import asyncio
import logging
from pathlib import Path

import pytest
from raygeo.geo import Geometry, Matrix

from rayforge.core.doc import Doc
from rayforge.core.group import Group
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    pass


@pytest.fixture
def doc():
    d = Doc()
    assert d.active_layer.workflow is not None
    d.active_layer.workflow.set_steps([])
    return d


@pytest.fixture
def real_workpiece():
    return WorkPiece(name="real_workpiece.svg")


@pytest.mark.usefixtures("context_initializer")
class TestPipelineInvalidation:
    """Tests that DOM changes trigger pipeline rebuilds."""

    svg_data = b"""
    <svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
    <rect width="50" height="30" />
    </svg>"""

    def _setup_doc_with_workpiece(self, doc, workpiece):
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

    async def _wait_settled(self, pipeline, timeout=10.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while (
            pipeline.is_busy and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_workpiece_geometry_change_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Changing workpiece geometry should trigger a new rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        real_workpiece.set_size(20, 20)
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_workpiece_position_change_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Changing workpiece position should trigger a rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        real_workpiece.pos = 100, 100
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_workpiece_rotation_change_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Changing workpiece rotation should trigger a rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        real_workpiece.angle = 45
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_step_power_change_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Changing step power should trigger a rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        new_power = 0.5 if step.power != 0.5 else 0.3
        step.set_power(new_power)

        await asyncio.sleep(0.2)

        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_adding_workpiece_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Adding a workpiece to the doc should trigger a rebuild."""
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        self._setup_doc_with_workpiece(doc, real_workpiece)
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_group_transform_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Transforming a group containing workpieces should trigger
        a rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)

        group = Group()
        doc.active_layer.add_child(group)
        group.add_child(real_workpiece)

        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        group.matrix = group.matrix @ Matrix.translation(50, 50)
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_layer_transform_triggers_rebuild(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Transforming a layer should trigger a rebuild."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        gen1 = pipeline.data_generation_id

        layer = doc.active_layer
        layer.matrix = layer.matrix @ Matrix.translation(10, 10)
        await self._wait_settled(pipeline)

        assert pipeline.data_generation_id > gen1
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_rapid_multiple_changes_settle_correctly(
        self,
        doc,
        real_workpiece,
        task_mgr,
        context_initializer,
        contour_step_class,
    ):
        """Rapid multiple changes should settle to a correct state."""
        self._setup_doc_with_workpiece(doc, real_workpiece)
        step = contour_step_class.create(context_initializer)
        # Keep the rendered images small: this test exercises invalidation
        # bookkeeping, not raster performance, and the default 50 px/mm
        # makes the 90x70mm workpiece render at 4500x3500px. That much
        # CPU work can exceed the settle timeout on a heavily loaded
        # parallel test run, making the test flaky.
        step.pixels_per_mm = (5, 5)
        doc.active_layer.workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            context_initializer.machine,
        )
        await self._wait_settled(pipeline)
        assert pipeline.is_busy is False, "initial rebuild did not settle"

        for i in range(5):
            step.set_power(0.1 * (i + 1))
            real_workpiece.set_size(50 + i * 10, 30 + i * 10)

        await self._wait_settled(pipeline, timeout=30.0)
        assert pipeline.is_busy is False

        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)
