# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk

from rayforge.core.variable_text import SerialSettings
from rayforge.doceditor.editor import DocEditor
from rayforge.ui_gtk.doceditor.property_providers.variable_text import (
    VariableTextPropertyProvider,
)
from rayforge.ui_gtk.doceditor.variable_text_dialog import (
    VariableTextDialog,
    format_preview,
)
from tests.core.template_provider import place


@pytest.fixture
def editor(ui_context_initializer, ui_task_mgr):
    editor = DocEditor(
        task_manager=ui_task_mgr, context=ui_context_initializer
    )
    yield editor
    editor.cleanup()


@pytest.fixture
def dialog(editor):
    parent = Gtk.Window()
    dlg = VariableTextDialog(parent, editor)
    yield dlg
    dlg.destroy()


def _serial(editor) -> SerialSettings:
    return editor.doc.variable_text.serial


@pytest.mark.ui
def test_format_preview():
    assert format_preview([7]) == "7"
    assert format_preview([7, 8, 9]) == "7, 8, 9"
    assert format_preview(list(range(1, 21))) == "1 … 20 (20 copies)"


@pytest.mark.ui
def test_dialog_shows_current_settings(editor):
    editor.doc.variable_text.serial = SerialSettings(
        current=42, start=10, step=2, auto_advance=False
    )
    dlg = VariableTextDialog(Gtk.Window(), editor)
    assert dlg.current_row.get_int_value() == 42
    assert dlg.start_row.get_int_value() == 10
    assert dlg.step_row.get_int_value() == 2
    assert dlg.auto_advance_row.get_active() is False
    dlg.destroy()


@pytest.mark.ui
def test_editing_current_value_is_undoable(editor, dialog):
    dialog.current_row.set_value(77)
    dialog.current_row.value_changed.send(dialog.current_row)
    assert _serial(editor).current == 77
    editor.history_manager.undo()
    assert _serial(editor).current == 1


@pytest.mark.ui
def test_end_value_only_applies_when_enabled(editor, dialog):
    dialog.end_row.set_value(9)
    dialog.end_row.value_changed.send(dialog.end_row)
    assert _serial(editor).end is None
    dialog.limit_row.set_active(True)
    assert _serial(editor).end == 9
    dialog.limit_row.set_active(False)
    assert _serial(editor).end is None


@pytest.mark.ui
def test_next_previous_reset_buttons(editor, dialog):
    place(editor.doc, "{serial}", offset=1)
    dialog.next_button.emit("clicked")
    assert _serial(editor).current == 3
    dialog.previous_button.emit("clicked")
    assert _serial(editor).current == 1
    editor.doc.variable_text.serial = SerialSettings(current=50, start=4)
    dialog.reset_button.emit("clicked")
    assert _serial(editor).current == 4


@pytest.mark.ui
def test_preview_follows_model(editor, dialog):
    place(editor.doc, "{serial}", offset=2)
    editor.doc.variable_text.serial = SerialSettings(current=5)
    assert dialog.preview_row.get_subtitle() == "5, 6, 7"
    assert dialog.current_row.get_int_value() == 5


@pytest.mark.ui
def test_switches_update_model(editor, dialog):
    dialog.auto_advance_row.set_active(False)
    dialog.export_row.set_active(True)
    assert _serial(editor).auto_advance is False
    assert _serial(editor).advance_on_export is True


@pytest.mark.ui
def test_property_provider_handles_template_workpieces(editor):
    wp = place(editor.doc, "{serial:03d}")
    plain = place(editor.doc, "Hello")
    provider = VariableTextPropertyProvider()
    assert provider.can_handle([wp])
    assert not provider.can_handle([plain])
    assert not provider.can_handle([])


@pytest.mark.ui
def test_property_provider_sets_offset(editor):
    editor.doc.variable_text.serial = SerialSettings(current=10)
    wp = place(editor.doc, "{serial}")
    provider = VariableTextPropertyProvider()
    provider.create_widgets()
    provider.update_widgets(editor, [wp])
    assert "10" in provider.offset_row.get_subtitle()
    provider.offset_row.set_value(3)
    provider.offset_row.value_changed.send(provider.offset_row)
    assert wp.variable_offset == 3
    provider.update_widgets(editor, [wp])
    assert "13" in provider.offset_row.get_subtitle()
