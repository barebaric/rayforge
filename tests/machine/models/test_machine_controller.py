"""
Tests for the MachineController class.

This module tests the MachineController which handles:
- Driver lifecycle management (connect/disconnect/shutdown)
- Command execution (jog, home, run_raw, etc.)
- Signal emissions for state changes

The MachineController is the logic layer that owns and manages the driver.
"""

import pytest

from rayforge.machine.models.controller import MachineController
from rayforge.machine.models.machine import Machine


@pytest.mark.usefixtures("context_initializer")
class TestMachineController:
    """Test suite for the MachineController class."""

    def test_controller_initialization(self, context_initializer):
        """Test that MachineController can be initialized."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        controller = MachineController(machine, context_initializer)
        assert controller is not None
        assert controller.machine == machine
        assert controller.context == context_initializer
        assert controller.driver is not None

    def test_controller_driver_property(self, context_initializer):
        """Test that the controller has a driver property."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert controller.driver is not None

    def test_controller_signals_exist(self, context_initializer):
        """Test that controller has all required signals."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert hasattr(controller, "connection_status_changed")
        assert hasattr(controller, "state_changed")
        assert hasattr(controller, "job_finished")
        assert hasattr(controller, "command_status_changed")
        assert hasattr(controller, "wcs_updated")
