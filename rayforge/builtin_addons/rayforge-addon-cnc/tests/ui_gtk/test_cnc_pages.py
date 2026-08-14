# flake8: noqa: E402
"""UI tests for the CNC step settings pages."""

from typing import Any, cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from cnc_essentials.steps.cnc_assembler_step import CncAssemblerStep
from cnc_essentials.widgets.pages import AdaptiveClearPage, ProfileOuterPage
from gi.repository import Adw, Gtk

from rayforge.core.step_registry import step_registry
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead
from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.doceditor.step_settings.pages import StepSettingsPage
from rayforge.ui_gtk.shared.pref_rows import (
    AngleSpinRow,
    LengthSpinRow,
    SpeedSpinRow,
)


def _row(page, key):
    for widget, _ in page._varset_widgets:
        row = widget.row_for(key)
        if row is not None:
            return row
    raise AssertionError(f"row {key} not found in page")


def _profile_step(ui_context) -> Any:
    step_cls = step_registry.get("ProfileOuterStep")
    assert step_cls is not None
    return cast(CncAssemblerStep, step_cls.create(ui_context))


@pytest.mark.ui
def test_profile_outer_page_composes_step_section(
    editor, cnc_machine, ui_context
):
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    assert isinstance(page, StepSettingsPage)

    # Step-specific rows live on the main page.
    for key in ("step_over", "step_length", "wall_margin"):
        _row(page, key)


@pytest.mark.ui
def test_cnc_page_composes_common_sections(editor, cnc_machine, ui_context):
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()
    assert isinstance(cnc_page, StepSettingsPage)

    for key in (
        "selected_head_uid",
        "spindle_rpm",
        "tool_diameter",
        "coolant_method",
        "target_depth",
        "depth_per_pass",
        "safe_z",
        "cut_speed",
        "travel_speed",
        "plunge_speed",
    ):
        _row(cnc_page, key)


@pytest.mark.ui
def test_head_change_commits(editor, cnc_machine, ui_context):
    """Selecting a different spindle head commits via undo."""
    second = SpindleHead()
    second.name = "Spindle 2"
    cnc_machine.add_head(second)

    step = _profile_step(ui_context)
    page = ProfileOuterPage(editor, step)
    cnc_page = page.cnc_page()

    cnc_page._on_head_changed(second.uid)
    assert step.selected_head_uid == second.uid


@pytest.mark.ui
def test_cooling_row_in_spindle_section(editor, cnc_machine, ui_context):
    """The coolant row lives in the Spindle section, always visible."""
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()

    coolant_row = cnc_page.spindle_widget.row_for("coolant_method")
    assert coolant_row is not None
    assert coolant_row.get_visible() is True


@pytest.mark.ui
def test_length_rows_use_user_units(editor, cnc_machine, ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()

    tool = cast(LengthSpinRow, _row(cnc_page, "tool_diameter"))
    assert isinstance(tool, LengthSpinRow)

    step = page.step
    step.tool_diameter = 25.4
    step.updated.send(step)

    assert tool.get_value_in_base_units() == pytest.approx(25.4)
    assert tool.get_value() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.ui
def test_plunge_speed_row_uses_speed_units(editor, cnc_machine, ui_context):
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()
    plunge = cast(SpeedSpinRow, _row(cnc_page, "plunge_speed"))
    assert isinstance(plunge, SpeedSpinRow)


@pytest.mark.ui
def test_deflection_row_uses_angle_spin_row(editor, cnc_machine, ui_context):
    step_cls = step_registry.get("AdaptiveClearStep")
    assert step_cls is not None
    step = cast(CncAssemblerStep, step_cls.create(ui_context))
    page = AdaptiveClearPage(editor, step)

    deflection = cast(AngleSpinRow, _row(page, "max_deflection_deg"))
    assert isinstance(deflection, AngleSpinRow)


@pytest.mark.ui
def test_speed_rows_show_integer_digits(editor, cnc_machine, ui_context):
    """Speed rows must not show fractional digits (500, not 500.00)."""
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()
    for key in ("cut_speed", "travel_speed", "plunge_speed"):
        row = cast(SpeedSpinRow, _row(cnc_page, key))
        assert row.get_digits() == 0, f"{key} shows fractional digits"


@pytest.mark.ui
def test_dialog_uses_profile_outer_page(editor, cnc_machine, ui_context):
    dialog = StepSettingsDialog(editor, _profile_step(ui_context))
    assert type(dialog.general_view).__name__ == "ProfileOuterPage"
    assert [title for title, _, _ in dialog._extra_pages] == ["CNC"]
    assert len(dialog._extra_buttons) == 1
    dialog.close()


@pytest.mark.ui
def test_dialog_initial_cnc_page(editor, cnc_machine, ui_context):
    dialog = StepSettingsDialog(editor, _profile_step(ui_context))
    dialog.set_initial_page("cnc")
    assert dialog._extra_buttons[0].get_active() is True
    assert dialog.btn_step_settings.get_active() is False
    dialog.close()


@pytest.mark.ui
def test_machine_switch_updates_head_dropdown(editor, cnc_machine, ui_context):
    """Switching the active machine rebuilds the spindle dropdown."""
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()

    # The dropdown initially lists "Spindle 1" from cnc_machine.
    head_row = cast(Adw.ComboRow, _row(cnc_page, "selected_head_uid"))
    model = cast(Gtk.StringList, head_row.get_model())
    names = [model.get_string(i) for i in range(model.get_n_items())]
    assert "Spindle 1" in names

    # Switch to a new machine with a differently-named spindle.
    new_machine = Machine(ui_context)
    new_machine.set_axis_extents(300, 200)
    new_spindle = SpindleHead()
    new_spindle.name = "Router Pro"
    new_machine.heads.clear()
    new_machine.add_head(new_spindle)
    ui_context.machine_mgr.add_machine(new_machine)
    ui_context.config.set_machine(new_machine)

    # The dropdown is rebuilt; re-fetch the row and check the new
    # spindle name appears.
    head_row = cast(Adw.ComboRow, _row(cnc_page, "selected_head_uid"))
    model = cast(Gtk.StringList, head_row.get_model())
    names = [model.get_string(i) for i in range(model.get_n_items())]
    assert "Router Pro" in names
    assert "Spindle 1" not in names

    cnc_page._cleanup()


@pytest.mark.ui
def test_machine_switch_preserves_row_order(editor, cnc_machine, ui_context):
    """Rebuilt rows stay in their original position after a machine switch."""
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    cnc_page = page.cnc_page()

    # The Spindle section rows in order: head, rpm, tool, cooling.
    spindle_widget = cnc_page.spindle_widget
    head_row = spindle_widget.row_for("selected_head_uid")
    rpm_row = spindle_widget.row_for("spindle_rpm")
    tool_row = spindle_widget.row_for("tool_diameter")
    coolant_row = spindle_widget.row_for("coolant_method")
    assert head_row is not None
    assert rpm_row is not None
    assert tool_row is not None
    assert coolant_row is not None

    # Switch machine — the head row is rebuilt.
    new_machine = Machine(ui_context)
    new_machine.set_axis_extents(300, 200)
    new_spindle = SpindleHead()
    new_spindle.name = "Router Pro"
    new_machine.heads.clear()
    new_machine.add_head(new_spindle)
    ui_context.machine_mgr.add_machine(new_machine)
    ui_context.config.set_machine(new_machine)

    # Re-fetch the rebuilt head row and verify the order is unchanged.
    new_head_row = spindle_widget.row_for("selected_head_uid")
    assert new_head_row is not None
    assert new_head_row is not head_row  # row was rebuilt

    # The head row should still come before the rpm row, which comes
    # before the tool row, which comes before the coolant row.
    assert new_head_row.get_next_sibling() is rpm_row
    assert rpm_row.get_next_sibling() is tool_row
    assert tool_row.get_next_sibling() is coolant_row

    cnc_page._cleanup()
