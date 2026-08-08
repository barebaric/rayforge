# flake8: noqa: E402
"""Tests for the HeadRow core row."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.rows import HeadRow


@pytest.mark.ui
def test_selection_emits_head_changed(editor, step, machine):
    events = []
    row = HeadRow(editor, step)
    row.head_changed.connect(
        lambda sender, head_uid: events.append(head_uid),
        weak=False,
    )
    target = machine.heads[0]
    row.widget.set_selected(1)
    assert events == [target.uid]


@pytest.mark.ui
def test_syncs_value_from_step(editor, step, machine):
    row = HeadRow(editor, step)
    target = machine.heads[0]
    step.selected_head_uid = target.uid
    step.updated.send(step)
    assert row.widget.get_selected() == 1
