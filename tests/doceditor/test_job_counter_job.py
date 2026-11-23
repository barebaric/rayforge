import asyncio

import pytest
from raygeo.template import CounterSettings

from rayforge.machine.cmd import MachineCmd
from rayforge.machine.job_status import JobKind, JobOutcome
from tests.core.template_provider import place


def _two_copies(doc_editor):
    place(doc_editor.doc, "{counter}", offset=0)
    place(doc_editor.doc, "{counter}", offset=1)
    doc_editor.job_counter.set_counter(CounterSettings(current=5))


async def _settle(doc_editor):
    await doc_editor.wait_until_settled()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_completed_send_advances_by_copies(doc_editor, machine):
    _two_copies(doc_editor)
    machine_cmd = MachineCmd(doc_editor)
    machine_cmd.job_finished.connect(doc_editor.job_counter.on_job_finished)
    machine_cmd._emit_job_finished(JobKind.SEND, JobOutcome.COMPLETED, machine)
    await _settle(doc_editor)
    assert doc_editor.doc.job_counter.counter.current == 7


@pytest.mark.asyncio
async def test_frame_job_does_not_advance(doc_editor, machine):
    _two_copies(doc_editor)
    machine_cmd = MachineCmd(doc_editor)
    machine_cmd.job_finished.connect(doc_editor.job_counter.on_job_finished)
    machine_cmd._emit_job_finished(
        JobKind.FRAME, JobOutcome.COMPLETED, machine
    )
    await _settle(doc_editor)
    assert doc_editor.doc.job_counter.counter.current == 5


@pytest.mark.asyncio
async def test_cancelled_job_does_not_advance(doc_editor, machine):
    _two_copies(doc_editor)
    machine_cmd = MachineCmd(doc_editor)
    machine_cmd.job_finished.connect(doc_editor.job_counter.on_job_finished)
    machine_cmd._emit_job_finished(JobKind.SEND, JobOutcome.CANCELLED, machine)
    await _settle(doc_editor)
    assert doc_editor.doc.job_counter.counter.current == 5


@pytest.mark.asyncio
async def test_gcode_export_advances_when_enabled(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(
        CounterSettings(current=5, advance_on_export=True)
    )
    doc_editor.file.gcode_exported.send(doc_editor.file)
    await _settle(doc_editor)
    assert doc_editor.doc.job_counter.counter.current == 7


@pytest.mark.asyncio
async def test_gcode_export_keeps_counter_by_default(doc_editor):
    _two_copies(doc_editor)
    doc_editor.file.gcode_exported.send(doc_editor.file)
    await _settle(doc_editor)
    assert doc_editor.doc.job_counter.counter.current == 5
