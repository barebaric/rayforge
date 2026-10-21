import pytest

from rayforge.core.job_counter import CounterSettings
from rayforge.doceditor.job_counter_cmd import JobCounterCmd
from tests.core.template_provider import place


@pytest.fixture
def cmd(doc_editor):
    return JobCounterCmd(doc_editor)


def _counter(doc_editor) -> CounterSettings:
    return doc_editor.doc.job_counter.counter


@pytest.mark.asyncio
async def test_editor_exposes_command(doc_editor):
    assert isinstance(doc_editor.job_counter, JobCounterCmd)


@pytest.mark.asyncio
async def test_set_counter_is_undoable(doc_editor, cmd):
    cmd.set_counter(CounterSettings(current=5, step=2))
    assert _counter(doc_editor).current == 5
    assert not doc_editor.is_saved
    doc_editor.history_manager.undo()
    assert _counter(doc_editor) == CounterSettings()


@pytest.mark.asyncio
async def test_next_previous_reset(doc_editor, cmd):
    cmd.set_counter(CounterSettings(current=10, start=3, step=5))
    cmd.next_counter()
    assert _counter(doc_editor).current == 15
    cmd.previous_counter()
    cmd.previous_counter()
    assert _counter(doc_editor).current == 5
    cmd.reset_counter()
    assert _counter(doc_editor).current == 3


@pytest.mark.asyncio
async def test_advance_after_job_uses_copies(doc_editor, cmd):
    doc = doc_editor.doc
    place(doc, "{counter}", offset=0)
    place(doc, "{counter}", offset=3)
    cmd.set_counter(CounterSettings(current=1, step=2))
    assert cmd.advance_after_job() is True
    assert _counter(doc_editor).current == 9


@pytest.mark.asyncio
async def test_advance_skipped_without_counter_users(doc_editor, cmd):
    place(doc_editor.doc, "Hello")
    assert cmd.advance_after_job() is False
    assert _counter(doc_editor).current == 1


@pytest.mark.asyncio
async def test_advance_skipped_when_disabled(doc_editor, cmd):
    place(doc_editor.doc, "{counter}")
    cmd.set_counter(CounterSettings(auto_advance=False))
    assert cmd.advance_after_job() is False
    assert _counter(doc_editor).current == 1


@pytest.mark.asyncio
async def test_advance_after_export_needs_its_own_switch(doc_editor, cmd):
    place(doc_editor.doc, "{counter}")
    assert cmd.advance_after_job(exported=True) is False
    cmd.set_counter(CounterSettings(advance_on_export=True))
    assert cmd.advance_after_job(exported=True) is True
    assert _counter(doc_editor).current == 2


@pytest.mark.asyncio
async def test_advance_is_undoable(doc_editor, cmd):
    place(doc_editor.doc, "{counter}")
    cmd.advance_after_job()
    doc_editor.history_manager.undo()
    assert _counter(doc_editor).current == 1


@pytest.mark.asyncio
async def test_number_copies_in_reading_order(doc_editor, cmd):
    doc = doc_editor.doc
    wps = []
    # Two rows of two; machine Y grows upwards, so the top row has
    # the larger Y and comes first.
    for x, y in [(30, 0), (0, 0), (30, 50), (0, 50)]:
        wp = place(doc, "{counter}")
        wp.pos = (x, y)
        wps.append(wp)
    cmd.number_copies(wps)
    assert [wp.variable_offset for wp in wps] == [3, 2, 1, 0]
    doc_editor.history_manager.undo()
    assert all(wp.variable_offset == 0 for wp in wps)


@pytest.mark.asyncio
async def test_preview_values(doc_editor, cmd):
    place(doc_editor.doc, "{counter}", offset=2)
    cmd.set_counter(CounterSettings(current=7))
    assert cmd.preview_counter_values() == [7, 8, 9]


@pytest.mark.asyncio
async def test_preview_without_copies_shows_current(doc_editor, cmd):
    cmd.set_counter(CounterSettings(current=7))
    assert cmd.preview_counter_values() == [7]

