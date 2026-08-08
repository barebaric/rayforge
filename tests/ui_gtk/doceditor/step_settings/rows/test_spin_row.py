"""Tests for the SpinRow generic row."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


def _spin_row(editor, step):
    return SpinRow(
        editor, step, "count", "Count", None, 1, 10, 1, 0, is_int=True
    )


@pytest.mark.ui
def test_syncs_value_from_step(editor, step):
    row = _spin_row(editor, step)
    assert row.widget.get_value() == 3
    step.count = 7
    step.updated.send(step)
    assert row.widget.get_value() == 7


@pytest.mark.ui
def test_commit_writes_to_step(editor, step):
    row = _spin_row(editor, step)
    row.commit(5)
    assert step.count == 5


@pytest.mark.ui
def test_change_schedules_debounced_commit(editor, step):
    row = _spin_row(editor, step)
    row.widget.get_adjustment().set_value(9)
    assert row._debounce_timer != 0
    row.cleanup()


@pytest.mark.ui
def test_external_update_does_not_clobber_pending_edit(editor, step):
    row = _spin_row(editor, step)
    row.widget.get_adjustment().set_value(9)
    assert row._debounce_timer != 0

    step.power = 0.4
    step.updated.send(step)

    assert row.widget.get_value() == 9
    assert row._debounce_timer != 0
    row.cleanup()


@pytest.mark.ui
def test_set_range(editor, step):
    row = _spin_row(editor, step)
    row.set_range(0, 20)
    adj = row.widget.get_adjustment()
    assert adj.get_lower() == 0
    assert adj.get_upper() == 20
