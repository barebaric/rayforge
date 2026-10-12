# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from pathlib import Path

import pytest
from gi.repository import Gtk

from rayforge.core.variable_text import CsvMerge
from rayforge.doceditor.editor import DocEditor
from rayforge.ui_gtk.doceditor.variable_text_csv_group import format_rows
from rayforge.ui_gtk.doceditor.variable_text_dialog import (
    VariableTextDialog,
)
from tests.core.template_provider import place

ROWS = "Name,City\nAnn,Gent\nBob,Hasselt\nCas,Tongeren\n"
MERGE = CsvMerge(
    header=("Name", "City"),
    rows=(("Ann", "Gent"), ("Bob", "Hasselt"), ("Cas", "Tongeren")),
)


@pytest.fixture
def editor(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


@pytest.fixture
def group(editor):
    parent = Gtk.Window()
    dialog = VariableTextDialog(parent, editor)
    yield dialog.csv_group
    dialog.destroy()


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    path = tmp_path / "people.csv"
    path.write_text(ROWS, encoding="utf-8")
    return path


def _csv(editor) -> CsvMerge:
    return editor.doc.variable_text.csv


def test_format_rows():
    assert format_rows(CsvMerge(), [1]) == "No CSV file loaded"
    assert format_rows(MERGE, [1, 2]) == "1: Ann, 2: Bob"
    assert format_rows(MERGE, [3, 4]) == "3: Cas, 4 (empty)"
    assert format_rows(MERGE, [1, 2, 3, 4]).endswith("… 4 copies")
    assert format_rows(MERGE.with_current(4), [4]) == (
        "All rows have been used"
    )


@pytest.mark.ui
def test_empty_group_is_disabled(group):
    assert group.file_row.get_subtitle() == "No file linked"
    assert not group.reload_button.get_sensitive()
    assert not group.next_button.get_sensitive()
    assert group.preview_row.get_subtitle() == "No CSV file loaded"


@pytest.mark.ui
def test_loading_updates_the_group(editor, group, csv_file):
    place(editor.doc, '{csv["Name"]}', offset=0)
    place(editor.doc, '{csv["Name"]}', offset=1)
    editor.variable_text.load_csv(str(csv_file))
    assert group.file_row.get_subtitle() == str(csv_file)
    assert group.reload_button.get_sensitive()
    assert group.columns_row.get_subtitle() == "Name, City (3 rows)"
    assert group.preview_row.get_subtitle() == "1: Ann, 2: Bob"
    assert group.copies_label.get_label() == "3"


@pytest.mark.ui
def test_next_and_first_buttons(editor, group, csv_file):
    editor.variable_text.load_csv(str(csv_file))
    group.next_button.emit("clicked")
    assert _csv(editor).current == 2
    assert group.current_row.get_int_value() == 2
    group.first_button.emit("clicked")
    assert _csv(editor).current == 1


@pytest.mark.ui
def test_current_row_and_switch_commit(editor, group, csv_file):
    editor.variable_text.load_csv(str(csv_file))
    group.current_row.set_value(3)
    group.current_row.value_changed.send(group.current_row)
    assert _csv(editor).current == 3
    group.auto_advance_row.set_active(False)
    assert _csv(editor).auto_advance is False
    editor.history_manager.undo()
    assert _csv(editor).auto_advance is True
