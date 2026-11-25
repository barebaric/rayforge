"""End-to-end Epilog encoding through the pipeline.

A machine using ``EpilogZingDriver`` must encode to a PJL/PCL/HPGL
print job: the artifact carries a readable listing for the G-code
panel plus the exact device byte stream the driver uploads.
"""

import asyncio
from pathlib import Path

import pytest
from raydriver.encode import EpilogSpec
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import SVG_RENDERER
from rayforge.machine.driver.epilog import EpilogZingDriver
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Origin
from rayforge.pipeline.epilog_inputs import (
    build_epilog_context,
    epilog_spec,
)
from rayforge.pipeline.intent_builder import IntentBuilder
from rayforge.pipeline.pipeline import Pipeline

SVG_DATA = b"""
<svg width="50mm" height="30mm" xmlns="http://www.w3.org/2000/svg">
<rect width="50" height="30" />
</svg>
"""

JOB_HEADER = b"\x1b%-12345X@PJL JOB NAME="
JOB_FOOTER = b"\x1bE\x1b%-12345X@PJL EOJ \r\n"


@pytest.fixture(autouse=True)
def _zero_debounce(zero_debounce_delay):
    """Apply zero debounce delay to all tests in this file."""


def _setup_doc(doc: Doc) -> None:
    workpiece = WorkPiece(name="rect.svg")
    source = SourceAsset(
        Path(workpiece.name), original_data=SVG_DATA, renderer=SVG_RENDERER
    )
    doc.add_asset(source)
    workpiece.source_segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=Geometry(),
        vectorization_spec=PassthroughSpec(),
    )
    workpiece.set_size(50, 30)
    workpiece.pos = 10, 20
    doc.active_layer.add_workpiece(workpiece)


@pytest.fixture
def epilog_machine(context_initializer):
    from rayforge.machine.models.laser import LaserHead
    from rayforge.machine.models.machine import Machine, Origin

    machine = Machine(context_initializer)
    machine.name = "Epilog"
    machine.auto_connect = False
    machine.add_head(LaserHead())
    machine.set_axis_extents(609.6, 304.8)
    machine.set_origin(Origin.TOP_LEFT)
    context_initializer.machine_mgr.add_machine(machine)
    machine.set_driver(
        EpilogZingDriver,
        {"host": "10.0.0.1", "port": 515, "resolution": "500"},
    )
    return machine


@pytest.mark.usefixtures("context_initializer")
class TestEpilogEncoding:
    @pytest.mark.asyncio
    async def test_artifact_carries_payload_and_listing(
        self, task_mgr, context_initializer, contour_step_class, epilog_machine
    ):
        doc = Doc()
        workflow = doc.active_layer.workflow
        assert workflow is not None
        workflow.set_steps([])
        _setup_doc(doc)
        step = contour_step_class.create(context_initializer)
        workflow.add_step(step)

        pipeline = Pipeline(
            doc,
            task_mgr,
            context_initializer.artifact_store,
            epilog_machine,
        )
        handle = await pipeline.generate_job_artifact_async()

        assert handle is not None
        with context_initializer.artifact_store.checkout_handle(
            handle
        ) as artifact:
            payload = artifact.payload
            assert payload is not None
            assert payload.startswith(JOB_HEADER)
            assert payload.rstrip(b"\x00").endswith(JOB_FOOTER)
            # The readable listing is escaped text, not raw bytes.
            assert "\x1b" not in artifact.machine_code
            assert "@PJL ENTER LANGUAGE=PCL" in artifact.machine_code

        await pipeline.wait_until_idle()
        pipeline.shutdown()


class TestEncoderRouting:
    async def _epilog_machine(self, machine, task_mgr):
        machine.auto_connect = False
        machine.add_head(LaserHead())
        machine.set_driver(
            EpilogZingDriver,
            {"host": "10.0.0.1", "port": 515, "resolution": "400"},
        )
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)
        await asyncio.sleep(0.3)
        await asyncio.to_thread(task_mgr.wait_until_settled, 5000)
        return machine

    @pytest.mark.asyncio
    async def test_context_from_machine_and_doc(self, machine, task_mgr):
        await self._epilog_machine(machine, task_mgr)
        machine.set_axis_extents(609.6, 304.8)
        machine.set_origin(Origin.TOP_LEFT)
        context = build_epilog_context(machine, Doc())
        assert context["resolution"] == 400
        assert context["bed_mm"] == [609.6, 304.8]
        assert context["origin_top_left"] is True
        assert context["rotary"] is False

    @pytest.mark.asyncio
    async def test_rotary_layer_is_flagged(self, machine, task_mgr):
        await self._epilog_machine(machine, task_mgr)
        doc = Doc()
        doc.active_layer.set_rotary_enabled(True)
        assert build_epilog_context(machine, doc)["rotary"] is True

    @pytest.mark.asyncio
    async def test_intent_builder_selects_epilog_spec(self, machine, task_mgr):
        await self._epilog_machine(machine, task_mgr)
        builder = IntentBuilder(machine=machine, generation_id=1)
        assert isinstance(builder._build_encoder(Doc()), EpilogSpec)
        assert isinstance(epilog_spec(machine, Doc()), EpilogSpec)
