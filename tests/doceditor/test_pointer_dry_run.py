"""
End-to-end tests for the pointer dry-run send: the encoded job must
trace the toolpath with the pointer dot (coordinates shifted by the
pointer offset through the WCS offsets) at the framing power instead
of job power, while regular sends stay unshifted at job power.
"""

import re

import pytest
from raygeo.geo import Geometry

from rayforge.context import get_context
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.cmd import MachineCmd
from rayforge.machine.models.machine import Origin


def _first_coord(text: str) -> tuple[float, float]:
    """The first X/Y coordinate pair emitted in a G-code text."""
    match = re.search(r"X(-?\d+(?:\.\d+)?) Y(-?\d+(?:\.\d+)?)", text)
    assert match is not None, f"No X/Y coordinate found in:\n{text}"
    return float(match.group(1)), float(match.group(2))


def _add_rect_doc_items(doc_editor, contour_step_class, context):
    step = contour_step_class.create(context, name="cut")
    step.set_power(0.8)
    workflow = doc_editor.doc.active_layer.workflow
    assert workflow is not None
    workflow.add_child(step)

    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()
    wp = WorkPiece(name="rect")
    wp._edited_boundaries = geo
    wp.set_size(50.0, 30.0)
    doc_editor.doc.active_layer.add_child(wp)


@pytest.mark.asyncio
async def test_pointer_dry_run_shifts_and_caps_power(
    context_initializer,
    doc_editor,
    contour_step_class,
    mocker,
):
    """Dry-Run with Pointer traces with the pointer dot at framing
    power; regular sends burn unshifted at job power."""
    machine = get_context().machine
    assert machine is not None
    machine.set_dialect_uid("grbl")
    machine.set_origin(Origin.BOTTOM_LEFT)

    head = machine.get_default_laser_head()
    assert head is not None
    head.max_power = 1000
    head.set_frame_power(0.1)
    head.set_pointer_offset(10.0, 20.0)
    head.set_pointer_offset_enabled(True)
    machine.set_pointer_alignment(True)

    _add_rect_doc_items(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()

    machine_cmd = MachineCmd(doc_editor)
    run_spy = mocker.spy(machine.driver, "run")

    await machine_cmd.send_job(machine)
    await doc_editor.wait_until_settled()
    regular = run_spy.call_args_list[0].args[0].text

    await machine_cmd.send_job(machine, pointer_dry_run=True)
    await doc_editor.wait_until_settled()
    dry_run = run_spy.call_args_list[-1].args[0].text

    await machine_cmd.send_job(machine)
    await doc_editor.wait_until_settled()
    regular_again = run_spy.call_args_list[-1].args[0].text

    # Regular send: job power (0.8 of 1000 W), unshifted coordinates.
    assert "S800" in regular
    reg_x, reg_y = _first_coord(regular)

    # Dry run: framing power (0.1 of 1000 W), coordinates shifted by
    # the pointer offset so the pointer dot traces the toolpath.
    assert "S800" not in dry_run
    assert "S100" in dry_run
    dry_x, dry_y = _first_coord(dry_run)
    assert dry_x == pytest.approx(reg_x - 10.0)
    assert dry_y == pytest.approx(reg_y - 20.0)

    # The follow-up regular send is bit-identical to the first one:
    # the shifted dry-run artifact must not leak into the cache.
    assert regular_again == regular
