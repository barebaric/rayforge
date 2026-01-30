"""
Tests for the Machine data model.

This module tests the Machine class as a data model, focusing on:
- Machine properties and attributes
- Coordinate transformations
- Machine state management

The Machine class delegates driver lifecycle and command logic to
MachineController, so this module focuses on the data model aspects.
"""

import pytest

from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.transport import TransportStatus


@pytest.mark.usefixtures("context_initializer")
class TestMachineModel:
    """Test suite for the Machine data model."""

    def test_machine_initialization(self, context_initializer):
        """Test that a Machine can be initialized with a context."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        assert machine is not None
        assert machine.context == context_initializer
        assert machine.controller is not None
        assert machine.driver is not None

    def test_machine_id_generation(self, context_initializer):
        """Test that each machine gets a unique ID."""
        machine1 = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine1)
        machine2 = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine2)
        assert machine1.id != machine2.id
        assert isinstance(machine1.id, str)

    def test_machine_dimensions(self, context_initializer):
        """Test setting and getting machine dimensions."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        machine.set_dimensions(200, 150)
        assert machine.dimensions == (200, 150)

    def test_machine_origin(self, context_initializer):
        """Test setting and getting machine origin."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        machine.set_origin(Origin.TOP_LEFT)
        assert machine.origin == Origin.TOP_LEFT

        machine.set_origin(Origin.BOTTOM_LEFT)
        assert machine.origin == Origin.BOTTOM_LEFT

    def test_machine_driver_property(self, context_initializer):
        """Test that the driver property delegates to controller."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        assert machine.driver == machine.controller.driver

    def test_machine_connection_status(self, context_initializer):
        """Test machine connection status property."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        assert machine.connection_status == TransportStatus.DISCONNECTED

    def test_machine_wcs_properties(self, context_initializer):
        """Test machine WCS (Work Coordinate System) properties."""
        machine = Machine(context_initializer)
        context_initializer.machine_mgr.add_machine(machine)
        assert machine.machine_space_wcs is not None
        assert isinstance(machine.machine_space_wcs_display_name, str)

    def test_machine_signals_exist(self, context_initializer):
        """Test that machine has all required signals."""
        machine = Machine(context_initializer)
        assert hasattr(machine, "connection_status_changed")
        assert hasattr(machine, "state_changed")
        assert hasattr(machine, "job_finished")
        assert hasattr(machine, "command_status_changed")
        assert hasattr(machine, "wcs_updated")
