# flake8: noqa: E402
"""UI tests for the CNC step settings pages."""

from typing import Any, cast

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from cnc_essentials.steps.cnc_assembler_step import CncAssemblerStep
from cnc_essentials.widgets.pages import AdaptiveClearPage, ProfileOuterPage

from rayforge.core.step_registry import step_registry
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
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
def test_profile_outer_page_composes_common_sections(
    editor, cnc_machine, ui_context
):
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    assert isinstance(page, StepSettingsPage)

    for key in (
        "spindle_rpm",
        "tool_diameter",
        "target_depth",
        "depth_per_pass",
        "safe_z",
        "cut_speed",
        "travel_speed",
        "plunge_speed",
    ):
        _row(page, key)


@pytest.mark.ui
def test_cooling_section_visible_for_spindle_head(
    editor, cnc_machine, ui_context
):
    """CNC pages show the coolant section for a spindle head."""
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    assert page.coolant_section.get_visible() is True


@pytest.mark.ui
def test_cooling_section_hidden_for_laser_head(editor, ui_context):
    """CNC pages hide the coolant section for a laser head."""
    machine = Machine(ui_context)
    machine.set_axis_extents(200, 150)
    head = Laser()
    head.name = "Laser 1"
    head.spot_size_mm = (0.1, 0.2)
    machine.heads.clear()
    machine.add_head(head)
    ui_context.machine_mgr.machines.clear()
    ui_context.machine_mgr.add_machine(machine)
    ui_context.config.set_machine(machine)

    page = ProfileOuterPage(editor, _profile_step(ui_context))
    assert page.coolant_section.get_visible() is False


@pytest.mark.ui
def test_length_rows_use_user_units(editor, cnc_machine, ui_context):
    ui_context.config.unit_preferences["length"] = "in"
    page = ProfileOuterPage(editor, _profile_step(ui_context))

    tool = cast(LengthSpinRow, _row(page, "tool_diameter"))
    assert isinstance(tool, LengthSpinRow)

    step = page.step
    step.tool_diameter = 25.4
    step.updated.send(step)

    assert tool.get_value_in_base_units() == pytest.approx(25.4)
    assert tool.get_value() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.ui
def test_plunge_speed_row_uses_speed_units(editor, cnc_machine, ui_context):
    page = ProfileOuterPage(editor, _profile_step(ui_context))
    plunge = cast(SpeedSpinRow, _row(page, "plunge_speed"))
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
    for key in ("cut_speed", "travel_speed", "plunge_speed"):
        row = cast(SpeedSpinRow, _row(page, key))
        assert row.get_digits() == 0, f"{key} shows fractional digits"
