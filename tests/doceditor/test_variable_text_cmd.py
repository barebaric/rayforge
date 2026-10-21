import pytest

from rayforge.core.variable_text import SerialSettings
from rayforge.doceditor.variable_text_cmd import VariableTextCmd
from tests.core.template_provider import place


@pytest.fixture
def cmd(doc_editor):
    return VariableTextCmd(doc_editor)


def _serial(doc_editor) -> SerialSettings:
    return doc_editor.doc.variable_text.serial


@pytest.mark.asyncio
async def test_editor_exposes_command(doc_editor):
    assert isinstance(doc_editor.variable_text, VariableTextCmd)


@pytest.mark.asyncio
async def test_set_serial_is_undoable(doc_editor, cmd):
    cmd.set_serial(SerialSettings(current=5, step=2))
    assert _serial(doc_editor).current == 5
    assert not doc_editor.is_saved
    doc_editor.history_manager.undo()
    assert _serial(doc_editor) == SerialSettings()


@pytest.mark.asyncio
async def test_next_previous_reset(doc_editor, cmd):
    cmd.set_serial(SerialSettings(current=10, start=3, step=5))
    cmd.next_serial()
    assert _serial(doc_editor).current == 15
    cmd.previous_serial()
    cmd.previous_serial()
    assert _serial(doc_editor).current == 5
    cmd.reset_serial()
    assert _serial(doc_editor).current == 3


@pytest.mark.asyncio
async def test_advance_after_job_uses_copies(doc_editor, cmd):
    doc = doc_editor.doc
    place(doc, "{serial}", offset=0)
    place(doc, "{serial}", offset=3)
    cmd.set_serial(SerialSettings(current=1, step=2))
    assert cmd.advance_after_job() is True
    assert _serial(doc_editor).current == 9


@pytest.mark.asyncio
async def test_advance_skipped_without_serial_users(doc_editor, cmd):
    place(doc_editor.doc, "Hello")
    assert cmd.advance_after_job() is False
    assert _serial(doc_editor).current == 1


@pytest.mark.asyncio
async def test_advance_skipped_when_disabled(doc_editor, cmd):
    place(doc_editor.doc, "{serial}")
    cmd.set_serial(SerialSettings(auto_advance=False))
    assert cmd.advance_after_job() is False
    assert _serial(doc_editor).current == 1


@pytest.mark.asyncio
async def test_advance_after_export_needs_its_own_switch(doc_editor, cmd):
    place(doc_editor.doc, "{serial}")
    assert cmd.advance_after_job(exported=True) is False
    cmd.set_serial(SerialSettings(advance_on_export=True))
    assert cmd.advance_after_job(exported=True) is True
    assert _serial(doc_editor).current == 2


@pytest.mark.asyncio
async def test_advance_is_undoable(doc_editor, cmd):
    place(doc_editor.doc, "{serial}")
    cmd.advance_after_job()
    doc_editor.history_manager.undo()
    assert _serial(doc_editor).current == 1


@pytest.mark.asyncio
async def test_set_variable_offset_is_undoable(doc_editor, cmd):
    wp = place(doc_editor.doc, "{serial}")
    cmd.set_variable_offset([wp], 4)
    assert wp.variable_offset == 4
    doc_editor.history_manager.undo()
    assert wp.variable_offset == 0


@pytest.mark.asyncio
async def test_number_copies_in_reading_order(doc_editor, cmd):
    doc = doc_editor.doc
    wps = []
    # Two rows of two; machine Y grows upwards, so the top row has
    # the larger Y and comes first.
    for x, y in [(30, 0), (0, 0), (30, 50), (0, 50)]:
        wp = place(doc, "{serial}")
        wp.pos = (x, y)
        wps.append(wp)
    cmd.number_copies(wps)
    assert [wp.variable_offset for wp in wps] == [3, 2, 1, 0]
    doc_editor.history_manager.undo()
    assert all(wp.variable_offset == 0 for wp in wps)


@pytest.mark.asyncio
async def test_preview_values(doc_editor, cmd):
    place(doc_editor.doc, "{serial}", offset=2)
    cmd.set_serial(SerialSettings(current=7))
    assert cmd.preview_serial_values() == [7, 8, 9]


@pytest.mark.asyncio
async def test_preview_without_copies_shows_current(doc_editor, cmd):
    cmd.set_serial(SerialSettings(current=7))
    assert cmd.preview_serial_values() == [7]
