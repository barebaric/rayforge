# flake8: noqa: E402
"""Tests for the StepSettingsPage base."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.doceditor.step_settings.rows import CutSpeedRow, SpinRow


@pytest.mark.ui
def test_page_starts_with_identity_section(editor, step):
    page = StepSettingsPage(editor, step)
    assert len(page._sections) >= 1
    assert page._rows


@pytest.mark.ui
def test_add_section_accepts_row_class_and_instance(editor, step):
    page = StepSettingsPage(editor, step)
    page.add_section(
        "Cutting",
        CutSpeedRow,
        SpinRow(editor, step, "count", "Count", None, 1, 10, 1, 0),
    )
    assert len(page._sections) == 2


@pytest.mark.ui
def test_set_step_property_writes_to_step(editor, step):
    page = StepSettingsPage(editor, step)
    page.set_step_property("count", 8)
    assert step.count == 8


@pytest.mark.ui
def test_get_selected_head(editor, step, machine):
    page = StepSettingsPage(editor, step)
    step.selected_head_uid = machine.heads[0].uid
    assert page.get_selected_head() is machine.heads[0]
