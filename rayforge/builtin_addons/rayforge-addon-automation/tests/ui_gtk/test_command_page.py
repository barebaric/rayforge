# flake8: noqa: E402
"""UI tests for the Command step settings page."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from automation.steps import CommandStep
from automation.widgets.command_page import CommandStepSettingsPage
from gi.repository import Adw, Gtk

from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage


def _text_view(page):
    """The page's machine-code TextView (inside its varset row)."""
    for widget, _var_set in page._varset_widgets:
        row = widget.row_for("command_text")
        if row is not None:
            return row.core_widget
    raise AssertionError("command_text row not found in page")


def _buffer_text(page) -> str:
    buffer = _text_view(page).get_buffer()
    return buffer.get_text(
        buffer.get_start_iter(), buffer.get_end_iter(), False
    )


@pytest.mark.ui
def test_command_page_shows_text(editor, machine):
    step = CommandStep.create(editor.context)
    step.command_text = "M101\nG4 P2"

    page = CommandStepSettingsPage(editor, step)

    assert isinstance(page, StepSettingsPage)
    assert isinstance(page, Adw.PreferencesPage)
    assert _buffer_text(page) == "M101\nG4 P2"


@pytest.mark.ui
def test_command_page_editor_is_flat_and_tall(editor, machine):
    """No expander header: the editor is always unfolded and large."""
    step = CommandStep.create(editor.context)
    page = CommandStepSettingsPage(editor, step)

    row = page._varset_widgets[0][0].row_for("command_text")
    assert row is not None
    assert not isinstance(row, Adw.ExpanderRow)

    def _find_scroller(widget):
        if isinstance(widget, Gtk.ScrolledWindow):
            return widget
        child = (
            widget.get_first_child()
            if hasattr(widget, "get_first_child")
            else None
        )
        while child is not None:
            found = _find_scroller(child)
            if found is not None:
                return found
            child = child.get_next_sibling()
        return None

    scroller = _find_scroller(row.get_child())
    assert scroller is not None
    assert scroller.get_min_content_height() >= 220


@pytest.mark.ui
def test_command_page_edit_updates_step(editor, machine):
    step = CommandStep.create(editor.context)
    page = CommandStepSettingsPage(editor, step)

    widget, _var_set = page._varset_widgets[0]
    widget.set_values({"command_text": "M103"})
    page._on_varset_data_changed(widget, "command_text")

    assert step.command_text == "M103"


@pytest.mark.ui
def test_command_page_model_sync_overrides_buffer(editor, machine):
    step = CommandStep.create(editor.context)
    page = CommandStepSettingsPage(editor, step)

    step.set_command_text("M104")

    assert _buffer_text(page) == "M104"


@pytest.mark.ui
def test_command_page_typing_updates_step(editor, machine):
    """Typing in the editor reaches the step (via the widget debounce)."""
    step = CommandStep.create(editor.context)
    page = CommandStepSettingsPage(editor, step)

    widget, _var_set = page._varset_widgets[0]
    _text_view(page).get_buffer().set_text("M110\nM111")
    widget.flush_pending()

    assert step.command_text == "M110\nM111"
