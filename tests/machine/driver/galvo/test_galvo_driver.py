"""
Tests for GalvoDriver using MockGalvoConnection.

Verifies driver-level commands and state management.
"""

import asyncio
import logging
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from raygeo.ops import Ops

from rayforge.core.doc import Doc
from rayforge.machine.driver.driver import (
    Axis,
    DriverMaturity,
)
from rayforge.machine.driver.galvo.galvo_driver import GalvoDriver
from rayforge.machine.driver.galvo.galvo_encoder import GalvoEncoder
from rayforge.machine.models.laser import Laser, LaserType
from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)


async def wait_for_connection(
    driver: GalvoDriver, timeout: float = 2.0
) -> bool:
    """Wait for driver to establish connection."""
    await driver.connect()
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if driver.is_connected:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest_asyncio.fixture
async def driver(
    lite_context,
) -> AsyncGenerator[GalvoDriver, None]:
    """Provides a configured GalvoDriver with mock connection."""
    machine = Machine(lite_context)
    machine.driver_name = "GalvoDriver"
    lite_context.machine_mgr.add_machine(machine)

    driver = GalvoDriver(lite_context, machine)
    driver._setup_implementation(source="fiber", mock=True)

    yield driver

    await driver.cleanup()
    await machine.shutdown()


class TestGalvoDriver:
    @pytest.mark.asyncio
    async def test_setup_with_valid_config(self, driver):
        """Test that driver setup succeeds with valid configuration."""
        assert driver._controller is not None
        assert driver._source == "fiber"
        assert driver._mock

    @pytest.mark.asyncio
    async def test_connect_to_mock(self, driver):
        """Test that driver can connect to mock controller."""
        assert await wait_for_connection(driver)
        assert driver.is_connected

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_move_to(self, driver):
        """Test that move_to command works."""
        assert await wait_for_connection(driver)

        await driver.move_to(10.0, 20.0)

        controller = driver._controller
        assert controller is not None

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_set_power(self, driver):
        """Test that set_power command works."""
        assert await wait_for_connection(driver)

        test_head = Laser()
        test_head.uid = "test-head-1"
        test_head.tool_number = 0

        await driver.set_power(test_head, 0.5)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_set_power_zero(self, driver):
        """Test that set_power(0) works."""
        assert await wait_for_connection(driver)

        test_head = Laser()
        test_head.uid = "test-head-2"
        test_head.tool_number = 1

        await driver.set_power(test_head, 0.0)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_jog(self, driver):
        """Test that jog command works."""
        assert await wait_for_connection(driver)

        await driver.jog(5000, x=10.0)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_set_hold(self, driver):
        """Test that set_hold stops/resumes."""
        assert await wait_for_connection(driver)

        await driver.set_hold(True)
        await driver.set_hold(False)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_cancel(self, driver):
        """Test that cancel works."""
        assert await wait_for_connection(driver)

        await driver.cancel()

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_clear_alarm(self, driver):
        """Test that clear_alarm works."""
        assert await wait_for_connection(driver)

        await driver.clear_alarm()

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_select_tool_noop(self, driver):
        """Test that select_tool does nothing (no-op)."""
        await driver.connect()

        await driver.select_tool(1)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_read_settings_sends_signal(self, driver):
        """Test that read_settings sends the settings_read signal."""
        await driver.connect()

        settings_received = []

        def on_settings_read(sender, settings):
            settings_received.append(settings)

        driver.settings_read.connect(on_settings_read)

        await driver.read_settings()

        assert len(settings_received) == 1
        assert settings_received[0] == []

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_write_setting_noop(self, driver):
        """Test that write_setting does nothing (no-op)."""
        await driver.connect()

        await driver.write_setting("test_key", "test_value")

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_read_wcs_offsets(self, driver):
        """Test that read_wcs_offsets returns machine offsets."""
        assert await wait_for_connection(driver)

        offsets = await driver.read_wcs_offsets()

        assert "MACHINE" in offsets
        assert offsets["MACHINE"] == (0.0, 0.0, 0.0)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_read_parser_state_returns_none(self, driver):
        """Test that read_parser_state returns None."""
        await driver.connect()

        state = await driver.read_parser_state()

        assert state is None

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_run_probe_cycle_not_supported(self, driver):
        """Test that run_probe_cycle indicates not supported."""
        await driver.connect()

        probe_messages = []

        def on_probe_status_changed(sender, message):
            probe_messages.append(message)

        driver.probe_status_changed.connect(on_probe_status_changed)

        result = await driver.run_probe_cycle(Axis.Z, -10, 100)

        assert result is None
        assert len(probe_messages) > 0
        assert "not supported" in probe_messages[0].lower()

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_run_with_machine_code(self, driver):
        """Test that run method executes encoded commands."""
        doc = Doc()
        ops = Ops()
        ops.move_to(10.0, 20.0)
        ops.line_to(30.0, 40.0)

        encoded = driver._machine.encode_ops(ops, doc)

        assert await wait_for_connection(driver)

        await driver.run(encoded, doc, ops)

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_run_raw_warns_and_finishes(self, driver):
        """Test that run_raw logs warning and sends job_finished signal."""
        await driver.connect()

        finished_events = []

        def on_job_finished(sender):
            finished_events.append(True)

        driver.job_finished.connect(on_job_finished)

        await driver.run_raw("G0 X10")

        assert len(finished_events) == 1

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_connection_status_signals(self, driver):
        """Test that connection status signals are emitted correctly."""
        status_changes = []

        def on_connection_status_changed(sender, status, message):
            status_changes.append((status, message))

        driver.connection_status_changed.connect(on_connection_status_changed)

        assert await wait_for_connection(driver)

        await driver.cleanup()

        assert len(status_changes) >= 2

    def test_get_encoder_returns_correct_type(self, driver):
        """Test that get_encoder returns GalvoEncoder."""
        encoder = driver.get_encoder()
        assert isinstance(encoder, GalvoEncoder)

    def test_machine_space_wcs_properties(self, driver):
        """Test that machine space WCS properties return correct values."""
        assert driver.machine_space_wcs == "MACHINE"
        assert driver.machine_space_wcs_display_name != ""

    def test_can_home_returns_false(self, driver):
        """Test that can_home returns False for galvo."""
        assert not driver.can_home()
        assert not driver.can_home(Axis.X)
        assert not driver.can_home(Axis.Y)

    def test_can_jog_returns_true(self, driver):
        """Test that can_jog returns True."""
        assert driver.can_jog()
        assert driver.can_jog(Axis.X)
        assert driver.can_jog(Axis.Y)

    def test_driver_label_and_subtitle(self, driver):
        """Test that driver has correct label and subtitle."""
        assert "Galvo" in driver.label
        assert "EzCad2" in driver.label
        assert driver.subtitle != ""
        assert driver.reports_granular_progress is False
        assert driver.uses_gcode is False

    def test_driver_maturity(self, driver):
        """Test driver maturity level."""
        assert driver.maturity == DriverMaturity.EXPERIMENTAL

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, driver):
        """Test that cleanup works even when not connected."""
        await driver.cleanup()
        assert not driver.is_connected

    @pytest.mark.asyncio
    async def test_cleanup_resets_controller(self, driver):
        """Test that cleanup properly resets controller."""
        await driver.cleanup()
        assert driver._controller is None

    def test_resource_uri_property(self, driver):
        """Test that resource_uri returns correct format."""
        uri = driver.resource_uri
        assert uri == "usb://galvo/ezcad2"

    def test_setup_vars_include_source_and_mock(self):
        """Test that setup vars include source and mock options."""
        varset = GalvoDriver.get_setup_vars()
        keys = [v.key for v in varset]
        assert "source" in keys
        assert "mock" in keys

    def test_precheck_passes_by_default(self):
        """Test that precheck does not raise."""
        GalvoDriver.precheck()

    def test_supported_wcs_includes_machine(self, driver):
        """Test that supported_wcs includes MACHINE."""
        wcs_list = driver.supported_wcs
        assert "MACHINE" in wcs_list

    @pytest.mark.asyncio
    async def test_reconnect(self, driver):
        """Test driver can reconnect after disconnect."""
        assert await wait_for_connection(driver)
        assert driver.is_connected

        await driver.cleanup()
        assert not driver.is_connected

        driver._setup_implementation(source="fiber", mock=True)
        assert await wait_for_connection(driver)
        assert driver.is_connected

        await driver.cleanup()

    def test_get_laser_capabilities_diode(self, driver):
        """Test that laser capabilities are empty for diode."""
        laser = Laser()
        laser.laser_type = LaserType.DIODE

        result = driver.get_laser_capabilities(laser)

        assert result == ()

    def test_get_laser_capabilities_co2(self, driver):
        """Test that laser capabilities include PWM for CO2."""
        laser = Laser()
        laser.laser_type = LaserType.CO2
        laser.pwm_frequency = 1000
        laser.max_pwm_frequency = 5000
        laser.pulse_width = 50
        laser.min_pulse_width = 5
        laser.max_pulse_width = 500

        result = driver.get_laser_capabilities(laser)

        assert len(result) == 1

    def test_get_laser_capabilities_fiber(self, driver):
        """Test that laser capabilities include PWM for fiber."""
        laser = Laser()
        laser.laser_type = LaserType.FIBER

        result = driver.get_laser_capabilities(laser)

        assert len(result) == 1
        assert result[0].name == "PWM"
