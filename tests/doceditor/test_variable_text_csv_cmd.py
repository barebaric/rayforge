from pathlib import Path

import pytest
import pytest_asyncio

from rayforge.core.variable_text import CsvMerge, SerialSettings
from rayforge.doceditor.variable_text_cmd import VariableTextCmd
from tests.core.template_provider import place

ROWS = "Name,City\nAnn,Gent\nBob,Hasselt\nCas,Tongeren\nDirk,Genk\n"


@pytest_asyncio.fixture
async def cmd(doc_editor):
    yield VariableTextCmd(doc_editor)
    await doc_editor.wait_until_settled()


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "people.csv"
    path.write_text(ROWS, encoding="utf-8")
    return path


def _csv(doc_editor) -> CsvMerge:
    return doc_editor.doc.variable_text.csv


def _notifications(doc_editor) -> list[str]:
    messages: list[str] = []
    doc_editor.notification_requested.connect(
        lambda sender, message, **kw: messages.append(message), weak=False
    )
    return messages


@pytest.mark.asyncio
async def test_load_csv_is_undoable(doc_editor, cmd, csv_file):
    assert cmd.load_csv(str(csv_file))
    merge = _csv(doc_editor)
    assert merge.header == ("Name", "City")
    assert merge.row_count == 4
    assert merge.path == str(csv_file)
    doc_editor.history_manager.undo()
    assert not _csv(doc_editor).is_loaded


@pytest.mark.asyncio
async def test_load_failure_notifies(doc_editor, cmd, tmp_path):
    messages = _notifications(doc_editor)
    assert not cmd.load_csv(str(tmp_path / "missing.csv"))
    bad = tmp_path / "latin1.csv"
    bad.write_bytes("Naam\nJos\xe9\n".encode("latin-1"))
    assert not cmd.load_csv(str(bad))
    assert len(messages) == 2
    assert not _csv(doc_editor).is_loaded


@pytest.mark.asyncio
async def test_reload_keeps_the_current_row(doc_editor, cmd, csv_file):
    cmd.load_csv(str(csv_file))
    cmd.set_csv(_csv(doc_editor).with_current(3))
    csv_file.write_text(ROWS + "Eva,Bree\n", encoding="utf-8")
    assert cmd.reload_csv()
    assert _csv(doc_editor).row_count == 5
    assert _csv(doc_editor).current == 3


@pytest.mark.asyncio
async def test_row_navigation_moves_by_copies(doc_editor, cmd, csv_file):
    place(doc_editor.doc, '{csv["Name"]}', offset=0)
    place(doc_editor.doc, '{csv["Name"]}', offset=1)
    cmd.load_csv(str(csv_file))
    cmd.next_row()
    assert _csv(doc_editor).current == 3
    cmd.previous_row()
    assert _csv(doc_editor).current == 1
    cmd.next_row()
    cmd.first_row()
    assert _csv(doc_editor).current == 1


@pytest.mark.asyncio
async def test_preview_rows(doc_editor, cmd, csv_file):
    place(doc_editor.doc, '{csv["Name"]}', offset=0)
    place(doc_editor.doc, '{csv["Name"]}', offset=2)
    cmd.load_csv(str(csv_file))
    assert cmd.preview_rows() == [1, 2, 3]


@pytest.mark.asyncio
async def test_copies_needed(doc_editor, cmd, csv_file):
    cmd.load_csv(str(csv_file))
    assert cmd.copies_needed() == 4
    cmd.set_csv(_csv(doc_editor).with_current(4))
    assert cmd.copies_needed() == 1


@pytest.mark.asyncio
async def test_job_advances_the_row(doc_editor, cmd, csv_file):
    place(doc_editor.doc, '{csv["Name"]}', offset=0)
    place(doc_editor.doc, '{csv["Name"]}', offset=1)
    cmd.load_csv(str(csv_file))
    assert cmd.advance_after_job()
    assert _csv(doc_editor).current == 3


@pytest.mark.asyncio
async def test_job_does_not_advance_unused_variables(
    doc_editor, cmd, csv_file
):
    place(doc_editor.doc, '{csv["Name"]}')
    cmd.load_csv(str(csv_file))
    cmd.advance_after_job()
    assert doc_editor.doc.variable_text.serial == SerialSettings()
    assert _csv(doc_editor).current == 2


@pytest.mark.asyncio
async def test_serial_only_leaves_the_row_alone(doc_editor, cmd, csv_file):
    place(doc_editor.doc, "{serial}")
    cmd.load_csv(str(csv_file))
    cmd.advance_after_job()
    assert doc_editor.doc.variable_text.serial.current == 2
    assert _csv(doc_editor).current == 1


@pytest.mark.asyncio
async def test_advance_stops_at_the_end_and_notifies(
    doc_editor, cmd, csv_file
):
    messages = _notifications(doc_editor)
    place(doc_editor.doc, '{csv["Name"]}', offset=0)
    place(doc_editor.doc, '{csv["Name"]}', offset=2)
    cmd.load_csv(str(csv_file))
    cmd.advance_after_job()
    assert _csv(doc_editor).current == 4
    assert messages == []
    cmd.advance_after_job()
    assert _csv(doc_editor).current == 5
    assert _csv(doc_editor).is_exhausted
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_job_advance_is_one_undo_step(doc_editor, cmd, csv_file):
    place(doc_editor.doc, '{serial} {csv["Name"]}')
    cmd.load_csv(str(csv_file))
    cmd.advance_after_job()
    assert doc_editor.doc.variable_text.serial.current == 2
    assert _csv(doc_editor).current == 2
    doc_editor.history_manager.undo()
    assert doc_editor.doc.variable_text.serial.current == 1
    assert _csv(doc_editor).current == 1


@pytest.mark.asyncio
async def test_csv_auto_advance_can_be_off(doc_editor, cmd, csv_file):
    place(doc_editor.doc, '{csv["Name"]}')
    cmd.load_csv(str(csv_file))
    cmd.set_csv(
        CsvMerge(
            header=_csv(doc_editor).header,
            rows=_csv(doc_editor).rows,
            auto_advance=False,
        )
    )
    assert not cmd.advance_after_job()
    assert _csv(doc_editor).current == 1
