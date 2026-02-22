"""
Tests for the MachineManager class.

This module tests the MachineManager which handles:
- Machine lifecycle (loading/saving machine configurations)
- Active machine management
- Machine file persistence

The MachineManager is responsible for managing the collection of
machines and coordinating their lifecycle.
"""

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.machine.models.manager import MachineManager


@pytest.mark.usefixtures("lite_context")
class TestMachineManager:
    """Test suite for the MachineManager class."""

    def test_manager_initialization(self, tmp_path):
        """Test that MachineManager can be initialized."""
        manager = MachineManager(tmp_path)
        assert manager is not None
        assert manager.base_dir == tmp_path
        assert isinstance(manager.machines, dict)

    def test_manager_add_machine(self, lite_context, tmp_path):
        """Test adding a machine to the manager."""
        manager = MachineManager(tmp_path)
        machine = Machine(lite_context)

        manager.add_machine(machine)
        assert len(manager.machines) == 1
        assert manager.machines[machine.id] == machine

    def test_manager_get_machine_by_id(self, lite_context, tmp_path):
        """Test getting a machine by its ID."""
        manager = MachineManager(tmp_path)
        machine = Machine(lite_context)

        manager.add_machine(machine)
        found = manager.get_machine_by_id(machine.id)
        assert found == machine

    def test_manager_get_machine_by_id_not_found(self, tmp_path):
        """Test getting a non-existent machine by ID."""
        manager = MachineManager(tmp_path)
        found = manager.get_machine_by_id("non-existent-id")
        assert found is None

    def test_manager_signals_exist(self, tmp_path):
        """Test that manager has all required signals."""
        manager = MachineManager(tmp_path)
        assert hasattr(manager, "machine_added")
        assert hasattr(manager, "machine_removed")
        assert hasattr(manager, "machine_updated")
