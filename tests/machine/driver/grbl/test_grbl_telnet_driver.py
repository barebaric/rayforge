"""Tests for the Rust-backed GrblTelnetDriver shell."""

import pytest

from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.grbl import GrblTelnetDriver


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("GrblTelnetDriver") is GrblTelnetDriver

    def test_metadata(self):
        assert GrblTelnetDriver.machine_space_wcs
        assert GrblTelnetDriver.supports_settings
        assert GrblTelnetDriver.DISCOVERY is None

    def test_setup_vars(self):
        varset = GrblTelnetDriver.get_setup_vars()
        keys = set(varset.keys())
        assert {
            "host",
            "port",
            "poll_status_while_running",
            "deadlock_detection",
        } <= keys


class TestSetup:
    def test_precheck_rejects_invalid_host(self):
        with pytest.raises(Exception, match="[Hh]ostname"):
            GrblTelnetDriver.precheck(host="not a host")

    def test_precheck_accepts_hostname_and_ip(self):
        GrblTelnetDriver.precheck(host="localhost")
        GrblTelnetDriver.precheck(host="192.168.1.10")

    def test_setup_requires_host(self, context_initializer, machine):
        drv = GrblTelnetDriver(context_initializer, machine)
        drv.setup(port=23)
        assert drv.state.error is not None
        assert "Hostname" in drv.state.error.title
        assert drv.resource_uri is None

    def test_setup_builds_telnet_session(self, context_initializer, machine):
        drv = GrblTelnetDriver(context_initializer, machine)
        drv.setup(host="192.0.2.5", port=8023)
        assert drv.did_setup
        assert drv.resource_uri == "tcp://192.0.2.5:8023"

    def test_setup_defaults_to_port_23(self, context_initializer, machine):
        drv = GrblTelnetDriver(context_initializer, machine)
        drv.setup(host="grbl.local")
        assert drv.resource_uri == "tcp://grbl.local:23"
