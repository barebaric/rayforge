"""Tests for per-machine notes in Machine Settings."""

import gc

import pytest
from gi.repository import Gtk

from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.notes_page import NotesPage
from rayforge.ui_gtk.shared.markdown import MarkdownEditorDialog

pytestmark = pytest.mark.ui


def _widgets(widget):
    yield widget
    child = widget.get_first_child()
    while child:
        yield from _widgets(child)
        child = child.get_next_sibling()


def test_notes_edit_button_shows_icon_and_label(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)

    assert not page.edit_button.has_css_class("flat")
    labels = [
        child.get_text()
        for child in _widgets(page.edit_button)
        if isinstance(child, Gtk.Label)
    ]
    assert "Edit My Notes" in labels
    assert page.edit_button.get_margin_top() == 12
    assert any(
        isinstance(child, Gtk.Image) for child in _widgets(page.edit_button)
    )


def test_notes_page_empty_and_device_note_states(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.device_notes = "Read the profile setup guidance."
    page = NotesPage(machine)

    assert page.empty_label.get_visible()
    assert not page.user_notes_view.get_visible()
    assert page.device_group.get_visible()
    assert page.device_notes_view.get_text() == machine.device_notes


def test_notes_content_aligns_with_group_titles(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.user_notes = "# My notes"
    machine.device_notes = "# Device notes"
    page = NotesPage(machine)

    for widget in (
        page.user_notes_view,
        page.device_notes_view,
        page.empty_label,
        page.edit_button,
    ):
        assert widget.get_margin_start() == 0


def test_notes_page_saves_only_user_notes(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.device_notes = "Profile-maintained guidance."
    page = NotesPage(machine)
    page.edit_button.emit("clicked")
    dialog = page._editor_dialog
    assert dialog is not None

    assert isinstance(dialog, MarkdownEditorDialog)
    assert dialog.get_title() == "Edit My Notes"
    dialog.preview_editor.editor.set_text("My setup notes.")
    dialog._on_save()

    assert machine.user_notes == "My setup notes."
    assert machine.device_notes == "Profile-maintained guidance."
    assert page.user_notes_view.get_text() == "My setup notes."
    assert not page.empty_label.get_visible()


def test_notes_page_cancel_does_not_save(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)
    assert not page.device_group.get_visible()
    page.edit_button.emit("clicked")
    dialog = page._editor_dialog
    assert dialog is not None
    dialog.preview_editor.editor.set_text("Uncommitted")
    dialog._on_cancel()

    assert machine.user_notes == ""


def test_notes_editor_is_non_modal(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)

    page.edit_button.emit("clicked")

    dialog = page._editor_dialog
    assert dialog is not None
    assert not dialog.get_modal()


def test_notes_editor_reuses_still_open_dialog(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)

    page.edit_button.emit("clicked")
    dialog = page._editor_dialog
    assert dialog is not None
    page.edit_button.emit("clicked")

    assert page._editor_dialog is dialog


def test_notes_editor_reopens_with_current_notes(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)
    page.edit_button.emit("clicked")
    dialog = page._editor_dialog
    assert dialog is not None
    dialog.preview_editor.editor.set_text("First draft")
    dialog._on_save()

    page.edit_button.emit("clicked")
    reopened = page._editor_dialog
    assert reopened is not None

    assert reopened.get_text() == "First draft"
    assert reopened is not dialog


def test_notes_editor_saves_after_page_lost_its_window(
    ui_context_initializer,
):
    machine = Machine(ui_context_initializer)
    page = NotesPage(machine)
    window = Gtk.Window()
    window.set_child(page)
    page.edit_button.emit("clicked")
    dialog = page._editor_dialog
    assert dialog is not None

    window.destroy()
    del page
    gc.collect()

    dialog.preview_editor.editor.set_text("Saved without the window")
    dialog._on_save()

    assert machine.user_notes == "Saved without the window"
