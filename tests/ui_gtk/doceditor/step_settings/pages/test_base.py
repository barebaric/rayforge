# flake8: noqa: E402
"""Tests for the StepSettingsPage base."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk

from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage


@pytest.mark.ui
def test_page_starts_with_identity_section(editor, step):
    page = StepSettingsPage(editor, step)
    assert len(page._sections) >= 1
    assert page._rows


@pytest.mark.ui
def test_add_section_accepts_plain_widgets(editor, step):
    page = StepSettingsPage(editor, step)
    label = Gtk.Label(label="hello")
    page.add_section("Cutting", label)
    assert len(page._sections) == 2
    assert page._rows[-1] is label


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
