"""Built-in Epilog device profiles and framing capability gating.

The profiles ship in ``rayforge/resources/devices`` and must load
without a dialect (the driver is not a G-code driver). Framing is
gated by the driver via ``Driver.can_frame()``.
"""

from unittest.mock import Mock

import pytest

from rayforge.config import BUILTIN_DEVICES_DIR
from rayforge.machine.device.profile import DeviceProfile
from rayforge.machine.driver.dummy import NoDeviceDriver
from rayforge.machine.driver.epilog import EpilogZingDriver
from rayforge.machine.models.laser import LaserHead

PROFILES = {
    "epilog-zing-16": (406.0, 305.0),
    "epilog-zing-24": (609.6, 304.8),
}


@pytest.mark.parametrize("name,extents", PROFILES.items())
def test_builtin_profile_loads(name, extents):
    profile = DeviceProfile.from_path(BUILTIN_DEVICES_DIR / name)
    assert profile.machine_config.driver == "EpilogZingDriver"
    assert profile.machine_config.axis_extents == extents
    assert profile.dialect_config == {}
    assert profile.machine_config.has_z is False


@pytest.mark.parametrize("name", PROFILES)
def test_builtin_profile_creates_machine(lite_context, task_mgr, name):
    profile = DeviceProfile.from_path(BUILTIN_DEVICES_DIR / name)
    machine = profile.create_machine(lite_context)
    machine.auto_connect = False
    task_mgr.wait_until_settled(5000)
    driver = machine.controller.driver
    assert isinstance(driver, EpilogZingDriver)
    assert machine.dialect is None
    assert any(isinstance(h, LaserHead) for h in machine.heads)
    assert machine.driver_args["port"] == 515


class TestFramingCapability:
    def test_driver_default_can_frame(self):
        assert NoDeviceDriver.can_frame(Mock()) is True

    def test_epilog_can_frame(self):
        driver = EpilogZingDriver.__new__(EpilogZingDriver)
        assert driver.can_frame() is True

    def test_controller_delegates_to_driver(self):
        from rayforge.machine.models.controller import MachineController

        driver = Mock()
        driver.can_frame.return_value = False
        controller = MachineController.__new__(MachineController)
        controller.__dict__["driver"] = driver
        assert controller.can_frame() is False

        driver.can_frame.return_value = True
        assert controller.can_frame() is True

        controller.__dict__["driver"] = None
        assert controller.can_frame() is False

    def test_machine_requires_laser_and_driver(self, lite_context):
        from rayforge.machine.models.machine import Machine

        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        laser_heads = [h for h in machine.heads if isinstance(h, LaserHead)]
        assert laser_heads
        controller = machine.controller
        assert controller.driver.can_frame() is True
        assert machine.can_frame() is True

        machine.heads.clear()
        assert machine.can_frame() is False


def _settle(task_mgr) -> None:
    """Wait for the controller's debounced driver rebuild to finish."""
    task_mgr.wait_until_settled(5000)


class TestOriginConstraint:
    """The machine enforces the driver's layout instead of letting a
    user pick an origin the protocol cannot encode."""

    def _machine(
        self,
        lite_context,
        task_mgr,
        driver: type = EpilogZingDriver,
        args: dict | None = None,
    ):
        from rayforge.machine.models.machine import Machine

        machine = Machine(lite_context)
        machine.auto_connect = False
        lite_context.machine_mgr.add_machine(machine)
        machine.set_driver(driver, args or {"host": "10.0.0.1"})
        _settle(task_mgr)
        return machine

    def test_driver_declares_top_left(self):
        from rayforge.machine.models.machine import Origin

        assert EpilogZingDriver.supported_origins == (Origin.TOP_LEFT,)

    def test_set_origin_resolves_to_supported(self, lite_context, task_mgr):
        from rayforge.machine.models.machine import Origin

        machine = self._machine(lite_context, task_mgr)
        machine.set_origin(Origin.BOTTOM_LEFT)
        assert machine.origin is Origin.TOP_LEFT

    def test_driver_switch_moves_origin(self, lite_context, task_mgr):
        from rayforge.machine.driver.dummy import NoDeviceDriver
        from rayforge.machine.models.machine import Origin

        machine = self._machine(lite_context, task_mgr)
        machine.set_origin(Origin.TOP_LEFT)
        machine.set_driver(NoDeviceDriver, {})
        _settle(task_mgr)
        machine.set_origin(Origin.TOP_RIGHT)
        assert machine.origin is Origin.TOP_RIGHT

        machine.set_driver(EpilogZingDriver, {"host": "10.0.0.1"})
        _settle(task_mgr)
        assert machine.origin is Origin.TOP_LEFT

    def test_stored_value_is_normalized_on_read(self, lite_context, task_mgr):
        from rayforge.machine.models.machine import Machine, Origin

        machine = Machine(lite_context)
        machine.auto_connect = False
        lite_context.machine_mgr.add_machine(machine)
        # Simulate a config saved under another driver.
        machine._origin = Origin.BOTTOM_LEFT
        machine.set_driver(EpilogZingDriver, {"host": "10.0.0.1"})
        _settle(task_mgr)
        assert machine.origin is Origin.TOP_LEFT
        assert machine.to_dict()["machine"]["origin"] == "top_left"

    def test_supported_origins_defaults_to_all(self, lite_context, task_mgr):
        from rayforge.machine.driver.dummy import NoDeviceDriver
        from rayforge.machine.models.machine import Origin

        machine = self._machine(lite_context, task_mgr, NoDeviceDriver, {})
        assert set(machine.supported_origins) == set(Origin)
