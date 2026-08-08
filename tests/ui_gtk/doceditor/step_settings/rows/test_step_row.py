"""Tests for the StepRow base wrapper."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw

from rayforge.ui_gtk.doceditor.step_settings.rows import StepRow


class CountRow(StepRow):
    attr = "count"

    def build_widget(self):
        return Adw.SwitchRow(title="Count")


class VisibilityRow(StepRow):
    attr = "power"

    def build_widget(self):
        return Adw.SwitchRow(title="Power")

    def _sync_dependencies(self):
        self.set_visible(self.step.enabled)


@pytest.mark.ui
def test_commit_uses_setter(editor, step):
    row = CountRow(editor, step)
    row.commit(5)
    assert step.count == 5


@pytest.mark.ui
def test_commit_without_setter(editor, step):
    row = CountRow(editor, step)
    row.attr = "weight"
    row.commit(2.0)
    assert step.weight == 2.0


@pytest.mark.ui
def test_commit_skips_unchanged_value(editor, step):
    row = CountRow(editor, step)
    row.commit(3)
    assert step.count == 3


@pytest.mark.ui
def test_dependencies_reapply_on_step_update(editor, step):
    row = VisibilityRow(editor, step)
    step.enabled = False
    step.updated.send(step)
    assert row.widget.get_visible() is False
