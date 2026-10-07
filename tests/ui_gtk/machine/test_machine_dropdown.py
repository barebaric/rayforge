"""Regression tests for the machine dropdown's ETA status labels.

A machine's status line is rendered by every bound list item — the
collapsed button face and the rows of an open popup — so all of them
must receive ETA updates, and bind must apply the current ETA
immediately.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, PropertyMock

import pytest
from gi.repository import Gtk, Pango

from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.models.machine import Machine
from rayforge.shared.util.time_format import format_seconds
from rayforge.ui_gtk.machine.machine_dropdown import (
    MachineDropdown,
    MachineListItem,
)


def make_running_machine(context) -> Machine:
    """A machine whose status text includes the ETA when one is set."""
    machine = Machine(context)
    machine.set_axis_extents(200, 200)
    machine.set_device_state(DeviceState(status=DeviceStatus.RUN))
    return machine


def make_fake_list_item(machine: Machine):
    """Builds a list item mimicking what the factory setup created."""
    box = Gtk.Box(spacing=8)
    icon_box = Gtk.Box(valign=Gtk.Align.CENTER)
    text_box = Gtk.Box(
        orientation=Gtk.Orientation.VERTICAL, valign=Gtk.Align.CENTER
    )
    name_label = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
    status_label = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
    text_box.append(name_label)
    text_box.append(status_label)
    box.append(icon_box)
    box.append(text_box)

    item = MachineListItem(machine)
    list_item = MagicMock()
    list_item.get_child.return_value = box
    list_item.get_item.return_value = item
    return list_item, status_label


@pytest.fixture
def dropdown(
    ui_context_initializer, ui_task_mgr, mocker
) -> Iterator[MachineDropdown]:
    context = ui_context_initializer
    mocker.patch.object(
        Machine, "driver", new_callable=PropertyMock, return_value=None
    )
    machine = make_running_machine(context)
    context.machine_mgr.add_machine(machine)
    context.config.set_machine(machine)
    dropdown = MachineDropdown()
    yield dropdown
    context.config.set_machine(None)


@pytest.mark.ui
def test_update_eta_reaches_all_bound_labels(dropdown):
    machine = dropdown.get_selected_item().machine
    button_item, button_label = make_fake_list_item(machine)
    row_item, row_label = make_fake_list_item(machine)

    dropdown._on_factory_setup(None, button_item)
    dropdown._on_factory_bind(None, button_item)
    dropdown._on_factory_setup(None, row_item)
    dropdown._on_factory_bind(None, row_item)

    dropdown.update_eta(42.0)
    assert "42" in button_label.get_text()
    assert "42" in row_label.get_text()

    dropdown.update_eta(None)
    assert "42" not in button_label.get_text()
    assert "42" not in row_label.get_text()


@pytest.mark.ui
def test_unbind_only_removes_its_own_label(dropdown):
    machine = dropdown.get_selected_item().machine
    button_item, button_label = make_fake_list_item(machine)
    row_item, _row_label = make_fake_list_item(machine)

    dropdown._on_factory_setup(None, button_item)
    dropdown._on_factory_bind(None, button_item)
    dropdown._on_factory_setup(None, row_item)
    dropdown._on_factory_bind(None, row_item)

    dropdown._on_factory_unbind(None, row_item)

    dropdown.update_eta(7.0)
    assert "7" in button_label.get_text()

    dropdown._on_factory_unbind(None, button_item)
    assert dropdown._status_labels == {}


@pytest.mark.ui
def test_bind_applies_current_eta(dropdown):
    machine = dropdown.get_selected_item().machine
    dropdown.update_eta(99.0)

    item, label = make_fake_list_item(machine)
    dropdown._on_factory_setup(None, item)
    dropdown._on_factory_bind(None, item)

    assert format_seconds(99.0) in label.get_text()
