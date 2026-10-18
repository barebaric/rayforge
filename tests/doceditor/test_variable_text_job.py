"""
The job counter advances after a finished job (and, if enabled,
after a G-code export), but not after framing, pointer dry runs or a
cancelled job.
"""

import asyncio

import pytest
from raygeo.geo import Geometry

from rayforge.context import get_context
from rayforge.core.variable_text import CounterSettings
from rayforge.machine.cmd import MachineCmd
from tests.core.template_provider import place


def _rect() -> Geometry:
    geo = Geometry()
    geo.move_to(0.0, 0.0)
    geo.line_to(10.0, 0.0)
    geo.line_to(10.0, 10.0)
    geo.line_to(0.0, 10.0)
    geo.close_path()
    return geo


def _setup(doc_editor, contour_step_class, context, copies: int = 1):
    step = contour_step_class.create(context, name="cut")
    workflow = doc_editor.doc.active_layer.workflow
    assert workflow is not None
    workflow.add_child(step)
    for offset in range(copies):
        wp = place(doc_editor.doc, "{counter}", offset=offset)
        wp._edited_boundaries = _rect()
        wp.set_size(10.0, 10.0)
        wp.pos = (offset * 20.0, 0.0)
    doc_editor.doc.variable_text.counter = CounterSettings(current=5)


def _current(doc_editor) -> int:
    return doc_editor.doc.variable_text.counter.current


def _machine_cmd(doc_editor) -> MachineCmd:
    machine_cmd = MachineCmd(doc_editor)
    machine_cmd.job_completed.connect(
        doc_editor.variable_text.on_job_completed
    )
    return machine_cmd


async def _settle(doc_editor):
    await doc_editor.wait_until_settled()
    # Let callbacks scheduled on the main loop run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_send_job_advances_by_copies(
    context_initializer, doc_editor, contour_step_class
):
    _setup(doc_editor, contour_step_class, context_initializer, copies=3)
    await doc_editor.wait_until_settled()
    machine = get_context().machine
    assert machine is not None

    await _machine_cmd(doc_editor).send_job(machine)
    await _settle(doc_editor)

    assert _current(doc_editor) == 8


@pytest.mark.asyncio
async def test_frame_job_does_not_advance(
    context_initializer, doc_editor, contour_step_class
):
    _setup(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()
    machine = get_context().machine
    assert machine is not None

    await _machine_cmd(doc_editor).frame_job(machine)
    await _settle(doc_editor)

    assert _current(doc_editor) == 5


@pytest.mark.asyncio
async def test_pointer_dry_run_does_not_advance(
    context_initializer, doc_editor, contour_step_class
):
    _setup(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()
    machine = get_context().machine
    assert machine is not None

    await _machine_cmd(doc_editor).send_job(machine, pointer_dry_run=True)
    await _settle(doc_editor)

    assert _current(doc_editor) == 5


@pytest.mark.asyncio
async def test_cancelled_job_does_not_advance(
    context_initializer, doc_editor, contour_step_class, mocker
):
    _setup(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()
    machine = get_context().machine
    assert machine is not None
    machine_cmd = _machine_cmd(doc_editor)

    async def run_and_cancel(*args, **kwargs):
        machine_cmd.cancel_job(machine)

    mocker.patch.object(machine.driver, "run", side_effect=run_and_cancel)
    await machine_cmd.send_job(machine)
    await _settle(doc_editor)

    assert _current(doc_editor) == 5

    # The next, uncancelled job advances again.
    mocker.patch.object(machine.driver, "run", return_value=None)
    await machine_cmd.send_job(machine)
    await _settle(doc_editor)
    assert _current(doc_editor) == 6


@pytest.mark.asyncio
async def test_gcode_export_advances_when_enabled(
    context_initializer, doc_editor, contour_step_class, tmp_path
):
    _setup(doc_editor, contour_step_class, context_initializer)
    doc_editor.doc.variable_text.counter = CounterSettings(
        current=5, advance_on_export=True
    )
    await doc_editor.wait_until_settled()

    out = tmp_path / "job.gcode"
    doc_editor.file.export_gcode_to_path(out)
    for _i in range(200):
        await _settle(doc_editor)
        if out.exists() and _current(doc_editor) != 5:
            break
        await asyncio.sleep(0.01)

    assert out.exists()
    assert _current(doc_editor) == 6


@pytest.mark.asyncio
async def test_gcode_export_keeps_counter_by_default(
    context_initializer, doc_editor, contour_step_class, tmp_path
):
    _setup(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()

    out = tmp_path / "job.gcode"
    doc_editor.file.export_gcode_to_path(out)
    for _i in range(200):
        await _settle(doc_editor)
        if out.exists():
            break
        await asyncio.sleep(0.01)
    await _settle(doc_editor)

    assert out.exists()
    assert _current(doc_editor) == 5


@pytest.mark.asyncio
async def test_job_completed_only_after_real_send(
    context_initializer, doc_editor, contour_step_class
):
    _setup(doc_editor, contour_step_class, context_initializer)
    await doc_editor.wait_until_settled()
    machine = get_context().machine
    assert machine is not None
    machine_cmd = MachineCmd(doc_editor)
    completed = []

    def on_completed(sender):
        completed.append(sender)

    machine_cmd.job_completed.connect(on_completed)
    await machine_cmd.frame_job(machine)
    await machine_cmd.send_job(machine, pointer_dry_run=True)
    await _settle(doc_editor)
    assert completed == []

    await machine_cmd.send_job(machine)
    await _settle(doc_editor)
    assert completed == [machine_cmd]
