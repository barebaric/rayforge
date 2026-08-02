# flake8: noqa: E402
"""Tests for the ComboRow generic row."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.rows import ComboRow

_CHOICES = [("A", "a"), ("B", "b")]


def _combo_row(editor, step):
    return ComboRow(editor, step, "mode", "Mode", _CHOICES)


@pytest.mark.ui
def test_selection_commits_value(editor, step):
    row = _combo_row(editor, step)
    row.widget.set_selected(1)
    assert step.mode == "b"


@pytest.mark.ui
def test_syncs_value_from_step(editor, step):
    row = _combo_row(editor, step)
    step.mode = "b"
    step.updated.send(step)
    assert row.widget.get_selected() == 1
