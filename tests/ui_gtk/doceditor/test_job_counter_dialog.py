# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from gi.repository import Gtk
from raygeo.template import CounterSettings

from rayforge.ui_gtk.doceditor.job_counter_dialog import (
    JobCounterDialog,
    format_preview,
)


def test_format_preview_short_list():
    assert format_preview([1, 2, 3]) == "1, 2, 3"


@pytest.mark.ui
def test_dialog_reflects_and_edits_counter(ui_context_initializer, doc_editor):
    parent = Gtk.Window()
    dialog = JobCounterDialog(parent, doc_editor)
    doc_editor.job_counter.set_counter(CounterSettings(current=7, step=2))

    assert dialog.current_row.get_int_value() == 7
    assert dialog.step_row.get_int_value() == 2

    dialog.next_button.emit("clicked")
    assert doc_editor.doc.job_counter.counter.current == 9

    dialog.reset_button.emit("clicked")
    assert doc_editor.doc.job_counter.counter.current == 1
    dialog.destroy()
