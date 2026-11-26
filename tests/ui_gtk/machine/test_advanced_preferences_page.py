"""Tests for the Advanced machine settings page."""

import pytest

from rayforge.machine.driver.driver import ZHomingMode
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.advanced_preferences_page import (
    AdvancedPreferencesPage,
)

pytestmark = pytest.mark.ui


def test_park_rows_reflect_and_update_machine(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    page = AdvancedPreferencesPage(machine)

    assert machine.park_position_after_homing is None
    assert not page.park_after_homing_row.get_active()
    assert not page.park_x_row.get_sensitive()

    page.park_after_homing_row.set_active(True)

    assert machine.park_position_after_homing == (0.0, 0.0)
    assert page.park_x_row.get_sensitive()

    page.park_x_row.get_spin_button().set_value(12.5)

    assert machine.park_position_after_homing == (12.5, 0.0)

    page.park_after_homing_row.set_active(False)

    assert machine.park_position_after_homing is None
    assert not page.park_x_row.get_sensitive()


def test_park_rows_prefill_from_machine(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.set_park_position_after_homing((10.0, 5.0))
    page = AdvancedPreferencesPage(machine)

    assert page.park_after_homing_row.get_active()
    assert page.park_x_row.get_value_in_base_units() == 10.0
    assert page.park_y_row.get_value_in_base_units() == 5.0


def test_z_homing_hidden_without_a_choice(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.driver_name = "GrblSerialDriver"
    page = AdvancedPreferencesPage(machine)

    assert not page.z_homing_row.get_visible()


def test_z_homing_shows_driver_default(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.driver_name = "RuidaRPAAdapter"
    page = AdvancedPreferencesPage(machine)

    assert page.z_homing_row.get_visible()
    assert page.z_homing_row.get_selected() == 0
    assert machine.z_homing_mode is None


def test_z_homing_selection_updates_machine(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.driver_name = "RuidaRPAAdapter"
    page = AdvancedPreferencesPage(machine)

    page.z_homing_row.set_selected(1)

    assert machine.z_homing_mode is ZHomingMode.SWITCH


def test_z_homing_follows_driver_change(ui_context_initializer):
    machine = Machine(ui_context_initializer)
    machine.driver_name = "GrblSerialDriver"
    page = AdvancedPreferencesPage(machine)

    machine.driver_name = "RuidaRPAAdapter"
    machine.changed.send(machine)

    assert page.z_homing_row.get_visible()
    assert machine.z_homing_mode is None
