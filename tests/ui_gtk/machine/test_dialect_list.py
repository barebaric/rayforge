"""UI tests for the G-code dialect list editor's cleanup button."""

import time

import pytest
from gi.repository import Adw, GLib, Gtk

from rayforge.context import get_context
from rayforge.machine.models.dialect import GRBL_DIALECT
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.dialect_list import DialectListEditor

pytestmark = pytest.mark.ui


def _pump(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        while GLib.MainContext.default().iteration(False):
            pass
        time.sleep(0.01)


def _get_dialog_for(window: Gtk.Window) -> Adw.MessageDialog:
    dialogs = [
        toplevel
        for toplevel in Gtk.Window.list_toplevels()
        if isinstance(toplevel, Adw.MessageDialog)
        and toplevel.get_transient_for() is window
    ]
    assert dialogs
    return dialogs[-1]


def _create_editor(machine):
    editor = DialectListEditor(machine=machine)
    window = Adw.Window()
    window.set_content(editor)
    window.present()
    _pump(0.1)
    return editor, window


def _add_custom_dialect(label: str):
    dialect = GRBL_DIALECT.copy_as_custom(label)
    get_context().dialect_mgr.add_dialect(dialect)
    return dialect


def test_cleanup_button_is_the_group_header_suffix(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    editor, window = _create_editor(machine)

    try:
        assert editor.get_header_suffix() is editor.cleanup_button
    finally:
        window.destroy()


def test_cleanup_dialog_lists_unused_dialects_and_deletes_on_confirm(
    ui_context_initializer,
):
    unused_one = _add_custom_dialect("Unused One")
    unused_two = _add_custom_dialect("Unused Two")
    machine = Machine(ui_context_initializer)
    editor, window = _create_editor(machine)

    try:
        editor.cleanup_button.emit("clicked")
        _pump(0.1)
        dialog = _get_dialog_for(window)

        body = dialog.get_body()
        assert unused_one.label in body
        assert unused_two.label in body

        dialog.response("delete")
        _pump(0.1)

        registry_uids = get_context().dialect_mgr._registry.keys()
        assert unused_one.uid.lower() not in registry_uids
        assert unused_two.uid.lower() not in registry_uids
    finally:
        window.destroy()


def test_cleanup_shows_no_unused_dialog_when_all_dialects_in_use(
    ui_context_initializer,
):
    in_use = _add_custom_dialect("In Use")
    machine = Machine(ui_context_initializer)
    machine.dialect_uid = in_use.uid
    get_context().machine_mgr.add_machine(machine)
    editor, window = _create_editor(machine)

    try:
        editor.cleanup_button.emit("clicked")
        _pump(0.1)
        dialog = _get_dialog_for(window)

        assert dialog.get_heading() == "No Unused Dialects"
        assert dialog.get_body() == (
            "Every custom dialect is used by at least one machine."
        )

        dialog.response("ok")
        _pump(0.1)
        assert in_use.uid.lower() in get_context().dialect_mgr._registry
    finally:
        window.destroy()
