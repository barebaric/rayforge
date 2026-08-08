"""Tests for the StepSettingsPage base."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead
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
    assert len(page._sections) == 3


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
def test_cooling_section_hidden_for_laser_head(editor, step, machine):
    page = StepSettingsPage(editor, step)
    assert page.coolant_section.get_visible() is False


@pytest.mark.ui
def test_cooling_section_visible_for_spindle_head(editor, step, ui_context):
    machine = Machine(ui_context)
    machine.set_axis_extents(200, 150)
    head = SpindleHead()
    head.name = "Spindle"
    machine.heads.clear()
    machine.add_head(head)
    ui_context.machine_mgr.machines.clear()
    ui_context.machine_mgr.add_machine(machine)
    ui_context.config.set_machine(machine)

    page = StepSettingsPage(editor, step)
    assert page.coolant_section.get_visible() is True
