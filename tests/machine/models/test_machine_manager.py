"""
Tests for the MachineManager class.

This module tests the MachineManager which handles:
- Machine lifecycle (loading/saving machine configurations)
- Active machine management
- Machine file persistence

The MachineManager is responsible for managing the collection of
machines and coordinating their lifecycle.
"""

import asyncio

import pytest

from rayforge.machine.models.machine import Machine
from rayforge.machine.models.manager import MachineManager
from rayforge.shared.tasker.manager import TaskManager


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

    def test_has_controller_does_not_lazily_create(
        self, lite_context, tmp_path
    ):
        """has_controller reports existence without lazily creating one."""
        manager = MachineManager(tmp_path)
        machine = Machine(lite_context)
        manager.add_machine(machine)

        # No controller has been instantiated yet.
        assert manager.has_controller(machine.id) is False
        assert machine.id not in manager.controllers

        # It must also be safe (no raise) for unknown ids.
        assert manager.has_controller("non-existent-id") is False

        # Instantiating the controller flips the flag to True.
        manager.get_controller(machine.id)
        assert manager.has_controller(machine.id) is True

    @pytest.mark.asyncio
    async def test_removed_machine_reports_no_controller(
        self, machine: Machine, task_mgr: TaskManager
    ):
        """
        Regression for #280: removing the currently selected machine must
        leave it without a live controller. Accessing ``machine.controller``
        on a removed machine raises ValueError (the source of the crash), so
        ``has_controller`` is what lets the UI safely skip the disconnect of
        the controller's signals.
        """
        manager = machine.context.machine_mgr

        # Force the controller into existence, mirroring the active machine
        # whose laser_power_changed signal the UI has connected to.
        manager.get_controller(machine.id)
        assert machine.has_controller is True

        manager.remove_machine(machine.id)
        # Let the scheduled controller shutdown settle.
        await asyncio.to_thread(task_mgr.wait_until_settled, 2000)

        assert manager.has_controller(machine.id) is False
        assert machine.has_controller is False
        with pytest.raises(ValueError):
            _ = machine.controller
