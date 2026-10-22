"""
Tests for the MachineController class.

This module tests the MachineController which handles:
- Driver lifecycle management (connect/disconnect/shutdown)
- Command execution (jog, home, run_raw, etc.)
- Signal emissions for state changes

The MachineController is the logic layer that owns and manages the driver.
"""

import asyncio

import pytest

import rayforge.machine.driver as driver_module
from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.models.controller import MachineController
from rayforge.machine.models.machine import Machine
from rayforge.shared.tasker import task_mgr


class OptInDriver(NoDeviceDriver):
    """A driver that absorbs setup changes and stays alive."""

    def update_settings(self, **kwargs):
        self.accepted_args = kwargs
        return True


class OptOutDriver(NoDeviceDriver):
    """A driver that uses the default rebuild-on-change behavior."""


driver_module.register_driver(OptInDriver)
driver_module.register_driver(OptOutDriver)


async def wait_until_settled(task_mgr, timeout=2000):
    await asyncio.to_thread(task_mgr.wait_until_settled, timeout)


@pytest.mark.usefixtures("lite_context")
class TestMachineController:
    """Test suite for the MachineController class."""

    def test_controller_initialization(self, lite_context):
        """Test that MachineController can be initialized."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = MachineController(
            machine, lite_context, task_mgr.schedule_on_main_thread
        )
        assert controller is not None
        assert controller.machine == machine
        assert controller.context == lite_context
        assert controller.driver is not None

    def test_controller_driver_property(self, lite_context):
        """Test that the controller has a driver property."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert controller.driver is not None

    def test_controller_signals_exist(self, lite_context):
        """Test that controller has all required signals."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        assert hasattr(controller, "connection_status_changed")
        assert hasattr(controller, "state_changed")
        assert hasattr(controller, "job_finished")
        assert hasattr(controller, "command_status_changed")
        assert hasattr(controller, "wcs_updated")


@pytest.mark.usefixtures("lite_context")
class TestDriverRebuildDebounce:
    """Rapid driver configuration changes must coalesce into a single
    rebuild so the transport is not torn down per keystroke."""

    @pytest.mark.asyncio
    async def test_set_driver_args_burst_rebuilds_once(
        self, machine, mocker, task_mgr
    ):
        await wait_until_settled(task_mgr)
        machine.auto_connect = False
        rebuild_spy = mocker.spy(machine.controller, "rebuild_driver")

        machine.set_driver_args({"port": "/dev/one"})
        machine.set_driver_args({"port": "/dev/one-two"})
        machine.set_driver_args({"port": "/dev/one-two-three"})
        await wait_until_settled(task_mgr)

        assert rebuild_spy.call_count == 1
        assert machine.driver_args == {"port": "/dev/one-two-three"}

    @pytest.mark.asyncio
    async def test_set_driver_change_triggers_single_rebuild(
        self, machine, mocker, task_mgr
    ):
        from rayforge.machine.driver.dummy import NoDeviceDriver

        class TestDriver(NoDeviceDriver):
            pass

        await wait_until_settled(task_mgr)
        machine.auto_connect = False
        rebuild_spy = mocker.spy(machine.controller, "rebuild_driver")

        machine.set_driver(TestDriver, {"port": "/dev/null"})
        await wait_until_settled(task_mgr)

        assert rebuild_spy.call_count == 1
        assert machine.driver_name == "TestDriver"


@pytest.mark.usefixtures("lite_context")
class TestUpdateSettingsOptIn:
    """A driver that absorbs setup changes via update_settings keeps
    running without being torn down; the default (opt-out) still
    requests a rebuild."""

    @pytest.mark.asyncio
    async def test_opt_in_driver_keeps_live_instance(self, machine, task_mgr):
        await wait_until_settled(task_mgr)
        machine.auto_connect = False
        machine.set_driver(OptInDriver, {"port": "/dev/old"})
        await wait_until_settled(task_mgr)
        live_driver = machine.driver

        machine.set_driver_args({"port": "/dev/new"})
        await wait_until_settled(task_mgr)

        assert live_driver.accepted_args == {"port": "/dev/new"}
        assert machine.driver is live_driver
        assert machine.driver_args == {"port": "/dev/new"}

    @pytest.mark.asyncio
    async def test_opt_out_driver_is_rebuilt(self, machine, task_mgr):
        await wait_until_settled(task_mgr)
        machine.auto_connect = False
        machine.set_driver(OptOutDriver, {"port": "/dev/old"})
        await wait_until_settled(task_mgr)
        live_driver = machine.driver

        machine.set_driver_args({"port": "/dev/new"})
        await wait_until_settled(task_mgr)

        assert machine.driver is not live_driver


@pytest.mark.usefixtures("lite_context")
class TestHomingTracking:
    """The controller learns which axes a homing cycle moved by
    comparing reported positions before and after the HOME state."""

    def _make_machine_with_controller(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        controller = machine.controller
        return machine, controller

    def _drive(self, controller, status, pos):
        controller._on_driver_state_changed(
            controller.driver,
            DeviceState(status=status, machine_pos=pos),
        )

    def test_home_to_idle_records_homed_axes(self, lite_context):
        machine, controller = self._make_machine_with_controller(lite_context)
        self._drive(controller, DeviceStatus.IDLE, (-380.0, -380.0, 0.0))
        self._drive(controller, DeviceStatus.HOME, (-390.0, -390.0, 0.0))
        self._drive(controller, DeviceStatus.IDLE, (-397.0, -397.0, -3.0))
        assert machine.homed_positions == (-397.0, -397.0)

    def test_stationary_axis_is_not_recorded(self, lite_context):
        machine, controller = self._make_machine_with_controller(lite_context)
        self._drive(controller, DeviceStatus.IDLE, (-397.0, -10.0, 0.0))
        self._drive(controller, DeviceStatus.HOME, (-397.0, -10.0, 0.0))
        self._drive(controller, DeviceStatus.IDLE, (-397.0, -10.0, 0.0))
        assert machine.homed_positions == (None, None)

    def test_home_to_alarm_records_nothing(self, lite_context):
        machine, controller = self._make_machine_with_controller(lite_context)
        self._drive(controller, DeviceStatus.IDLE, (-380.0, -380.0, 0.0))
        self._drive(controller, DeviceStatus.HOME, (-390.0, -390.0, 0.0))
        self._drive(controller, DeviceStatus.ALARM, (-395.0, -395.0, 0.0))
        assert machine.homed_positions == (None, None)

    def test_reset_clears_homed_positions(self, lite_context):
        machine, controller = self._make_machine_with_controller(lite_context)
        self._drive(controller, DeviceStatus.IDLE, (-380.0, -380.0, 0.0))
        self._drive(controller, DeviceStatus.HOME, (-390.0, -390.0, 0.0))
        self._drive(controller, DeviceStatus.IDLE, (-397.0, -397.0, -3.0))
        assert machine.homed_positions == (-397.0, -397.0)

        controller._reset_status()
        assert machine.homed_positions == (None, None)
        assert controller._homing_start_positions is None
