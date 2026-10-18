"""Tests for the Z homing row in the Advanced machine settings."""

import pytest

from rayforge.machine.driver.driver import ZHomingMode
from rayforge.machine.models.machine import Machine
from rayforge.ui_gtk.machine.advanced_preferences_page import (
    AdvancedPreferencesPage,
)

pytestmark = pytest.mark.ui


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
