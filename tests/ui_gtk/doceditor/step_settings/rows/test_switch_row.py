# flake8: noqa: E402
"""Tests for the SwitchRow generic row."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.rows import SwitchRow


def _switch_row(editor, step):
    return SwitchRow(editor, step, "enabled", "Enabled")


@pytest.mark.ui
def test_toggle_commits_value(editor, step):
    row = _switch_row(editor, step)
    row.widget.set_active(False)
    assert step.enabled is False


@pytest.mark.ui
def test_syncs_value_from_step(editor, step):
    row = _switch_row(editor, step)
    step.enabled = False
    step.updated.send(step)
    assert row.widget.get_active() is False
