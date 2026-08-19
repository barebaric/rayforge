# flake8: noqa: E402
"""Tests for the StepSettingsPage base."""

from typing import cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk

from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.shared.pref_rows import SpeedSpinRow


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


@pytest.mark.ui
def test_travel_speed_row_hidden_without_support(editor, step, machine):
    """A machine whose dialect can't emit travel speed hides the row."""
    page = StepSettingsPage(editor, step)
    widget = page.add_varset_section("Motion", step.recipe_varset())
    page._update_machine_bounds()

    assert not machine.supports_travel_speed()
    row = widget.row_for("travel_speed")
    assert row is not None
    assert row.get_visible() is False


@pytest.mark.ui
def test_travel_speed_row_visible_with_support(editor, step, machine):
    """A machine whose dialect emits travel speed shows the row."""
    machine.set_dialect_uid("smoothieware")
    page = StepSettingsPage(editor, step)
    widget = page.add_varset_section("Motion", step.recipe_varset())
    page._update_machine_bounds()

    assert machine.supports_travel_speed()
    row = widget.row_for("travel_speed")
    assert row is not None
    assert row.get_visible() is True


@pytest.mark.ui
def test_travel_speed_warning_suppressed_without_support(
    editor, step, machine
):
    """No warning icon on the hidden travel row for a non-supporting
    machine, even with a travel speed above the machine's max."""
    step.travel_speed = 99999
    step.cut_speed = 100  # below max, so only travel would warn
    page = StepSettingsPage(editor, step)
    widget = page.add_varset_section("Motion", step.recipe_varset())
    page._update_machine_bounds()

    row = widget.row_for("travel_speed")
    assert row is not None
    icon = page._speed_warning_icons.get(cast(SpeedSpinRow, row))
    assert icon is None or icon.get_visible() is False
