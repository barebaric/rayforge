"""Regression tests for material-test-grid ops across workpiece resizes.

The canvas stretches a workpiece's ops view by
``current workpiece size / generation_size``, so ``generation_size``
must always describe the coordinate space the ops were generated in —
never the workpiece's live size at the moment the output happens to
arrive on the main thread.
"""

import asyncio
from types import SimpleNamespace

import pytest

from rayforge.core.doc import Doc
from rayforge.core.step_registry import step_registry
from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.pipeline import Pipeline


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    pass


@pytest.fixture
def doc():
    d = Doc()
    assert d.active_layer.workflow is not None
    d.active_layer.workflow.set_steps([])
    return d


async def _settle(pipeline: Pipeline, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while pipeline.is_busy and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.02)
    assert not pipeline.is_busy, "pipeline did not settle"


@pytest.fixture
def material_test_setup(
    doc, task_mgr, test_machine_and_config, context_initializer
):
    """A doc with a material test step and an 80x80 workpiece, plus a
    settled pipeline."""
    _machine, _config = test_machine_and_config
    step_cls = step_registry.get("MaterialTestStep")
    assert step_cls is not None
    step = step_cls.create(context_initializer)

    workpiece = WorkPiece(name="mtg")
    workpiece.set_size(80.0, 80.0)
    doc.active_layer.add_workpiece(workpiece)
    step.workpiece_uids = [workpiece.uid]
    doc.active_layer.workflow.add_step(step)

    pipeline = Pipeline(
        doc,
        task_mgr,
        context_initializer.artifact_store,
        context_initializer.machine,
    )
    return doc, pipeline, step, workpiece


class TestMaterialTestResize:
    @pytest.mark.asyncio
    async def test_resize_regenerates_ops_at_new_size(
        self, material_test_setup
    ):
        """A resize followed by the app's pause/resume sequence must
        regenerate the ops for the new size."""
        _doc, pipeline, step, workpiece = material_test_setup
        await _settle(pipeline)

        artifact = pipeline.get_artifact(step, workpiece)
        assert artifact.source_dimensions == pytest.approx((80.0, 80.0))

        pipeline.pause()
        workpiece.set_size(120.0, 60.0)
        pipeline.resume()
        await _settle(pipeline)

        artifact = pipeline.get_artifact(step, workpiece)
        assert artifact.source_dimensions == pytest.approx((120.0, 60.0))
        rect = artifact.ops.rect()
        assert rect[2] == pytest.approx(120.0, abs=0.5)
        assert rect[3] <= 60.0

        await asyncio.to_thread(pipeline.task_manager.wait_until_settled, 5000)

    @pytest.mark.asyncio
    async def test_late_output_keeps_generation_time_size(
        self, material_test_setup
    ):
        """An output computed before a resize but delivered after it must
        carry the size its ops were generated at, not the workpiece's
        live size."""
        _doc, pipeline, step, workpiece = material_test_setup
        await _settle(pipeline)

        stale = pipeline.get_artifact(step, workpiece)
        stale_output = SimpleNamespace(
            ops=stale.ops,
            is_scalable=stale.is_scalable,
            source_dimensions=stale.source_dimensions,
            warnings=[],
        )

        pipeline.pause()
        workpiece.set_size(120.0, 60.0)
        pipeline._on_wp_output(
            pipeline,
            step=step,
            workpiece=workpiece,
            output=stale_output,
            generation_id=pipeline.data_generation_id,
        )

        stored = pipeline.get_artifact(step, workpiece)
        assert stored.source_dimensions == pytest.approx((80.0, 80.0))
        assert stored.generation_size == pytest.approx((80.0, 80.0))

        pipeline.resume()
        await _settle(pipeline)
        await asyncio.to_thread(pipeline.task_manager.wait_until_settled, 5000)
