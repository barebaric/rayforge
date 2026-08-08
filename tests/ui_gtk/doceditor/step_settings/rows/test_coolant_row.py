"""Tests for the CoolantRow step setting."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from raygeo.ops.state import CoolantMode

from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead
from rayforge.ui_gtk.doceditor.step_settings.rows import CoolantRow


@pytest.fixture
def spindle_machine(ui_context):
    """A machine with a single spindle head supporting only flood."""
    machine = Machine(ui_context)
    machine.set_axis_extents(200, 150)

    head = SpindleHead()
    head.name = "Spindle"
    head.cooling_methods = (CoolantMode.FLOOD,)
    machine.heads.clear()
    machine.add_head(head)

    ui_context.machine_mgr.machines.clear()
    ui_context.machine_mgr.add_machine(machine)
    ui_context.config.set_machine(machine)
    return machine


@pytest.mark.ui
def test_selection_commits_coolant_method(editor, step, spindle_machine):
    row = CoolantRow(editor, step)
    row.widget.set_selected(1)
    assert step.coolant_method == CoolantMode.FLOOD


@pytest.mark.ui
def test_syncs_coolant_method_from_step(editor, step, spindle_machine):
    row = CoolantRow(editor, step)
    step.set_coolant_method(CoolantMode.MIST)
    step.updated.send(step)
    assert row.widget.get_selected() == 2


@pytest.mark.ui
def test_warning_hidden_when_off(editor, step, spindle_machine):
    row = CoolantRow(editor, step)
    assert row._warning_icon.get_visible() is False


@pytest.mark.ui
def test_warning_hidden_when_supported(editor, step, spindle_machine):
    row = CoolantRow(editor, step)
    step.set_coolant_method(CoolantMode.FLOOD)
    step.updated.send(step)
    assert row._warning_icon.get_visible() is False


@pytest.mark.ui
def test_warning_shown_when_unsupported(editor, step, spindle_machine):
    row = CoolantRow(editor, step)
    step.set_coolant_method(CoolantMode.MIST)
    step.updated.send(step)
    assert row._warning_icon.get_visible() is True


@pytest.mark.ui
def test_warning_hidden_without_machine(editor, step):
    row = CoolantRow(editor, step)
    step.set_coolant_method(CoolantMode.MIST)
    step.updated.send(step)
    assert row._warning_icon.get_visible() is False
