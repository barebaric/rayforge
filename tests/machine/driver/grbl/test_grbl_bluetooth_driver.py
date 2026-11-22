"""Tests for the Bluetooth-backed GRBL driver shell."""

import pytest

from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import DriverMaturity
from rayforge.machine.driver.grbl import GrblBluetoothDriver


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("GrblBluetoothDriver") is GrblBluetoothDriver

    def test_metadata(self):
        assert GrblBluetoothDriver.machine_space_wcs
        assert GrblBluetoothDriver.supports_settings
        assert GrblBluetoothDriver.maturity == DriverMaturity.EXPERIMENTAL
        assert GrblBluetoothDriver.DISCOVERY is None

    def test_setup_vars(self):
        varset = GrblBluetoothDriver.get_setup_vars()
        keys = set(varset.keys())
        assert {
            "address",
            "channel",
            "poll_status_while_running",
            "deadlock_detection",
            "rx_buffer_size_override",
        } <= keys
        assert varset["channel"].default == 1


class TestSetup:
    def test_precheck_rejects_missing_address(self):
        with pytest.raises(Exception, match="[Bb]luetooth address"):
            GrblBluetoothDriver.precheck(channel=1)

    def test_precheck_rejects_invalid_address(self):
        with pytest.raises(Exception, match="[Ii]nvalid Bluetooth address"):
            GrblBluetoothDriver.precheck(address="not-an-address", channel=1)

    def test_precheck_rejects_bad_channel(self):
        with pytest.raises(Exception, match="[Bb]luetooth channel"):
            GrblBluetoothDriver.precheck(
                address="AA:BB:CC:DD:EE:FF", channel=31
            )

    def test_precheck_accepts_address_forms(self):
        GrblBluetoothDriver.precheck(address="aa:bb:cc:dd:ee:ff", channel=1)
        GrblBluetoothDriver.precheck(address="aa-bb-cc-dd-ee-ff", channel=2)

    def test_setup_requires_address(self, context_initializer, machine):
        drv = GrblBluetoothDriver(context_initializer, machine)
        drv.setup(channel=1)
        assert drv.state.error is not None
        assert "Bluetooth address" in drv.state.error.title
        assert drv.resource_uri is None

    def test_setup_builds_bluetooth_session(
        self, context_initializer, machine
    ):
        drv = GrblBluetoothDriver(context_initializer, machine)
        drv.setup(address="aa-bb-cc-dd-ee-ff", channel=2)
        assert drv.did_setup
        assert drv.resource_uri == "bt://AA:BB:CC:DD:EE:FF"

    def test_setup_defaults_to_channel_one(self, context_initializer, machine):
        drv = GrblBluetoothDriver(context_initializer, machine)
        drv.setup(address="AA:BB:CC:DD:EE:FF")
        assert drv.resource_uri == "bt://AA:BB:CC:DD:EE:FF"

    def test_setup_rejects_invalid_address(self, context_initializer, machine):
        drv = GrblBluetoothDriver(context_initializer, machine)
        drv.setup(address="nope")
        assert drv.state.error is not None
        assert drv.resource_uri is None
