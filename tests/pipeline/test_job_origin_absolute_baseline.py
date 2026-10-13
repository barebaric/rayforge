"""
Start From "Absolute Coordinates" is the default and must not change
the output of any existing machine setup.

The expected files were generated on main before Start From existed
(merge base c9ac56606) for the machine setups below. A document with
the default job origin must still produce byte-identical G-code.
"""

from pathlib import Path

import pytest
import pytest_asyncio
from raygeo.ops.axis import Axis

from rayforge.context import get_context
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.machine.models.machine import Origin
from rayforge.machine.models.rotary_module import RotaryMode, RotaryModule
from rayforge.pipeline.artifact import JobArtifact

ASSETS = Path(__file__).parent.parent / "doceditor" / "assets"
DATA = Path(__file__).parent / "data" / "job_origin" / "absolute_baseline"


def _bottom_left(machine, layer):
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))


def _top_right_negative_axes(machine, layer):
    machine.set_origin(Origin.TOP_RIGHT)
    machine.set_reverse_x_axis(True)
    machine.set_reverse_y_axis(True)
    machine.update_wcs_offset("G54", (-120.0, -80.0, 0.0))


def _top_left_workarea_origin(machine, layer):
    machine.set_origin(Origin.TOP_LEFT)
    machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
    machine.set_wcs_origin_is_workarea_origin(True)


def _bottom_right_negative_x_margins(machine, layer):
    machine.set_origin(Origin.BOTTOM_RIGHT)
    machine.set_reverse_x_axis(True)
    machine.set_work_margins(5.0, 0.0, 15.0, 25.0)
    machine.set_wcs_origin_is_workarea_origin(True)


def _layer_with_own_wcs(machine, layer):
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))
    machine.update_wcs_offset("G55", (30.0, 20.0, 0.0))
    layer.set_wcs("G55")


def _rotary_layer(machine, layer):
    machine.set_origin(Origin.BOTTOM_LEFT)
    machine.update_wcs_offset("G54", (100.0, 50.0, 0.0))
    module = RotaryModule()
    module.set_mode(RotaryMode.TRUE_4TH_AXIS)
    module.set_axis(Axis.A)
    machine.add_rotary_module(module)
    layer.set_rotary_enabled(True)
    layer.set_rotary_module_uid(module.uid)
    layer.set_rotary_diameter(25.0)


SETUPS = {
    "bottom_left": _bottom_left,
    "top_right_negative_axes": _top_right_negative_axes,
    "top_left_workarea_origin": _top_left_workarea_origin,
    "bottom_right_negative_x_margins": _bottom_right_negative_x_margins,
    "layer_with_own_wcs": _layer_with_own_wcs,
    "rotary_layer": _rotary_layer,
}


@pytest_asyncio.fixture
async def editor(doc_editor, context_initializer, contour_step_class):
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_axis_extents(400, 400)
    machine.set_active_wcs("G54")

    step = contour_step_class.create(
        context_initializer, name="Contour", optimize=False
    )
    step.set_power(0.5)
    step.set_cut_speed(3000)
    workflow = doc_editor.doc.active_layer.workflow
    assert workflow is not None
    workflow.add_step(step)

    await doc_editor.import_file_from_path(
        ASSETS / "10x10_square.svg",
        mime_type="image/svg+xml",
        vectorization_spec=PassthroughSpec(),
    )
    await doc_editor.wait_until_settled()
    yield doc_editor, machine
    await doc_editor.wait_until_settled()


async def _gcode(editor) -> str:
    await editor.wait_until_settled()
    pipeline = editor.pipeline
    pipeline.invalidate_job_output()
    handle = await pipeline.generate_job_artifact_async()
    assert handle is not None
    with pipeline.artifact_store.checkout_handle(handle) as artifact:
        assert isinstance(artifact, JobArtifact)
        assert artifact.machine_code is not None
        gcode = artifact.machine_code
    await editor.wait_until_settled()
    return gcode


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(SETUPS))
async def test_default_output_matches_main(editor, name):
    doc_editor, machine = editor
    SETUPS[name](machine, doc_editor.doc.active_layer)
    gcode = await _gcode(doc_editor)
    expected = (DATA / f"{name}.gcode").read_text(encoding="utf-8")
    assert gcode == expected
