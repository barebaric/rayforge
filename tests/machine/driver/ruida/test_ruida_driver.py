"""
Tests for RuidaDriver using real RuidaSimulator.

This test suite runs against a real RuidaSimulator instance via UDP,
not mocks, ensuring end-to-end protocol compliance.
"""

import logging
import pytest
import pytest_asyncio
import asyncio
from typing import AsyncGenerator

from rayforge.core.doc import Doc
from rayforge.core.ops import Ops
from rayforge.core.varset import IntVar
from rayforge.machine.driver.driver import Axis
from rayforge.machine.driver.ruida.ruida_driver import RuidaDriver
from rayforge.machine.driver.ruida.ruida_encoder import RuidaEncoder
from rayforge.machine.driver.ruida.ruida_simulator import RuidaSimulator
from rayforge.machine.driver.ruida.ruida_transport import RuidaServerTransport
from rayforge.machine.models.laser import Laser, LaserType
from rayforge.machine.models.machine import Machine
from rayforge.machine.transport.transport import TransportStatus
from rayforge.machine.transport.udp_server import UdpServerTransport

logger = logging.getLogger(__name__)


async def wait_for_connection(
    driver: RuidaDriver, timeout: float = 2.0
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
async def ruida_simulator():
    """
    Provides a running RuidaSimulator with UDP transport.

    The simulator runs in a background task and is automatically
    stopped after each test.
    """
    sim = RuidaSimulator()
    host = "127.0.0.1"

    main_udp = UdpServerTransport(host, 0)
    main_transport = RuidaServerTransport(main_udp, magic=0x88)
    jog_transport = UdpServerTransport(host, 0)

    async def handle_main_decoded(sender, data: bytes, addr):
        response = sim.process_commands(data)
        if response == b"\xcc" or not response:
            await main_transport.send_response(b"\xcc", addr)
        else:
            await main_transport.send_response(b"\xcc", addr)
            await main_transport.send_response(response, addr)

    async def handle_jog(sender, data: bytes, addr):
        logger.debug(f"Jog received: {data.hex()} from {addr}")
        response = sim.handle_jog_packet(data)
        if response:
            await jog_transport.send_to(response, addr)

    def on_main_decoded(sender, data: bytes, addr):
        asyncio.create_task(handle_main_decoded(sender, data, addr))

    def on_jog_received(sender, data: bytes, addr):
        asyncio.create_task(handle_jog(sender, data, addr))

    main_transport.decoded_received.connect(on_main_decoded)
    jog_transport.received.connect(on_jog_received)

    await main_transport.connect()
    await jog_transport.connect()

    port = main_udp.port
    jog_port = jog_transport.port

    yield sim, host, port, jog_port

    await main_transport.disconnect()
    await jog_transport.disconnect()


@pytest_asyncio.fixture
async def driver(
    lite_context, ruida_simulator
) -> AsyncGenerator[RuidaDriver, None]:
    """
    Provides a configured RuidaDriver connected to the simulator.

    Uses the host/port from the running simulator fixture.
    """
    sim, host, port, jog_port = ruida_simulator

    machine = Machine(lite_context)
    machine.driver_name = "RuidaDriver"
    lite_context.machine_mgr.add_machine(machine)

    driver = RuidaDriver(lite_context, machine)
    driver._setup_implementation(
        host=host, port=port, jog_port=jog_port, response_port=0
    )

    yield driver

    await driver.cleanup()
    await machine.shutdown()


@pytest.mark.asyncio
async def test_setup_with_valid_config(driver):
    """Test that driver setup succeeds with valid configuration."""
    assert driver.host is not None
    assert driver.port is not None
    assert driver._udp_transport is not None
    assert driver._ruida_transport is not None
    assert driver._client is not None


@pytest.mark.asyncio
async def test_connect_to_simulator(driver, ruida_simulator):
    """Test that driver can connect to the simulator."""
    assert await wait_for_connection(driver)
    assert driver.is_connected

    await driver.cleanup()


@pytest.mark.asyncio
async def test_move_to_updates_position(driver, ruida_simulator):
    """Test that move_to command updates simulator position."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.move_to(10.0, 20.0)
    await asyncio.sleep(0.2)

    assert sim.x == 10000
    assert sim.y == 20000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_move_to_negative_position(driver, ruida_simulator):
    """Test that move_to works with negative coordinates."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.move_to(-5.5, -3.2)
    await asyncio.sleep(0.2)

    assert sim.x == -5500
    assert sim.y == -3200

    await driver.cleanup()


@pytest.mark.asyncio
async def test_home_xy_resets_position(driver, ruida_simulator):
    """Test that home_xy command resets position to zero."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 50000
    sim.y = 100000

    await driver.home()
    await asyncio.sleep(0.2)

    assert sim.x == 0
    assert sim.y == 0

    await driver.cleanup()


@pytest.mark.asyncio
async def test_home_z_axis(driver, ruida_simulator):
    """Test that home command with Z only does not move axes."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.z = 30000

    await driver.home(Axis.Z)
    await asyncio.sleep(0.2)

    assert sim.z == 30000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_home_all_axes(driver, ruida_simulator):
    """Test that home with None moves to origin (soft home)."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 100000
    sim.y = 200000
    sim.z = 50000

    await driver.home(None)
    await asyncio.sleep(0.2)

    assert sim.x == 0
    assert sim.y == 0
    assert sim.z == 50000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_home_xy_only(driver, ruida_simulator):
    """Test that home can target only XY axes."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 100000
    sim.y = 200000
    sim.z = 50000

    await driver.home(Axis.X | Axis.Y)
    await asyncio.sleep(0.2)

    assert sim.x == 0
    assert sim.y == 0
    assert sim.z == 50000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_set_power(driver, ruida_simulator):
    """Test that set_power command works correctly."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    test_head = Laser()
    test_head.uid = "test-head-1"
    test_head.tool_number = 0

    await driver.set_power(test_head, 0.5)
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_set_power_zero(driver, ruida_simulator):
    """Test that set_power(0) disables power."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    test_head = Laser()
    test_head.uid = "test-head-2"
    test_head.tool_number = 1

    await driver.set_power(test_head, 0.0)
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_jog_x_axis(driver, ruida_simulator):
    """Test that jog command works for X axis."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0

    await driver.jog(5000, x=10.0)
    await asyncio.sleep(0.2)

    assert sim.x == 10000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_jog_y_axis(driver, ruida_simulator):
    """Test that jog command works for Y axis."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.y = 0

    await driver.jog(3000, y=5.0)
    await asyncio.sleep(0.2)

    assert sim.y == 5000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_jog_both_axes(driver, ruida_simulator):
    """Test that jog can move both axes simultaneously."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 10000
    sim.y = 20000

    await driver.jog(4000, x=-10.0, y=5.0)
    await asyncio.sleep(0.2)

    assert sim.x == 0
    assert sim.y == 25000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_jog_negative_direction(driver, ruida_simulator):
    """Test that jog works in negative direction."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 100000

    await driver.jog(6000, x=-50.0)
    await asyncio.sleep(0.2)

    assert sim.x == 50000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_set_hold(driver, ruida_simulator):
    """Test that set_hold pauses the process."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    await driver.set_hold(True)
    await asyncio.sleep(0.2)

    await driver.set_hold(False)
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_cancel(driver, ruida_simulator):
    """Test that cancel stops the process."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    await driver.cancel()
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_clear_alarm(driver, ruida_simulator):
    """Test that clear_alarm works."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    await driver.clear_alarm()
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_select_tool_noop(driver):
    """Test that select_tool does nothing (no-op)."""
    await driver.connect()

    await driver.select_tool(1)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_read_settings_sends_signal(driver):
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
async def test_write_setting_noop(driver):
    """Test that write_setting does nothing (no-op)."""
    await driver.connect()

    await driver.write_setting("test_key", "test_value")

    await driver.cleanup()


@pytest.mark.asyncio
async def test_set_wcs_offset_noop(driver):
    """Test that set_wcs_offset does nothing (no-op)."""
    await driver.connect()

    await driver.set_wcs_offset("G54", 10.0, 20.0, 30.0)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_read_wcs_offsets(driver, ruida_simulator):
    """Test that read_wcs_offsets returns ref point offsets."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    offsets = await driver.read_wcs_offsets()

    assert "MACHINE" in offsets
    assert offsets["MACHINE"] == (0.0, 0.0, 0.0)
    assert "REF0" in offsets
    assert "REF1" in offsets

    await driver.cleanup()


@pytest.mark.asyncio
async def test_read_parser_state_returns_none(driver):
    """Test that read_parser_state returns None."""
    await driver.connect()

    state = await driver.read_parser_state()

    assert state is None

    await driver.cleanup()


@pytest.mark.asyncio
async def test_run_probe_cycle_not_supported(driver):
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
async def test_run_with_machine_code(driver, ruida_simulator):
    """Test that run method executes encoded commands on the simulator."""
    sim, host, port, jog_port = ruida_simulator

    doc = Doc()
    ops = Ops()
    ops.move_to(10.0, 20.0)
    ops.line_to(30.0, 40.0)

    encoded = driver._machine.encode_ops(ops, doc)

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.run(encoded, doc)
    await asyncio.sleep(0.2)

    assert sim.x == 30000
    assert sim.y == 40000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_run_raw_warns_and_finishes(driver):
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
async def test_connection_status_signals(driver, ruida_simulator):
    """Test that connection status signals are emitted correctly."""
    status_changes = []

    def on_connection_status_changed(sender, status, message):
        status_changes.append((status, message))

    driver.connection_status_changed.connect(on_connection_status_changed)

    assert await wait_for_connection(driver)

    await driver.cleanup()

    await asyncio.sleep(0.05)

    assert len(status_changes) >= 2


@pytest.mark.asyncio
async def test_unit_conversion_mm_to_um(driver, ruida_simulator):
    """Test that mm values are correctly converted to µm."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.move_to(123.456, 789.012)
    await asyncio.sleep(0.2)

    assert sim.x == 123456
    assert sim.y == 789012

    await driver.cleanup()


@pytest.mark.asyncio
async def test_unit_conversion_small_values(driver, ruida_simulator):
    """Test unit conversion with sub-millimeter values."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.move_to(0.123, 0.456)
    await asyncio.sleep(0.2)

    assert sim.x == 123
    assert sim.y == 456

    await driver.cleanup()


@pytest.mark.asyncio
async def test_multiple_moves_in_sequence(driver, ruida_simulator):
    """Test that multiple move commands work in sequence."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.move_to(10.0, 10.0)
    await asyncio.sleep(0.2)

    await driver.move_to(20.0, 20.0)
    await asyncio.sleep(0.2)

    await driver.move_to(30.0, 30.0)
    await asyncio.sleep(0.2)

    assert sim.x == 30000
    assert sim.y == 30000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_home_then_move(driver, ruida_simulator):
    """Test that home followed by move works correctly."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 100000
    sim.y = 200000

    await driver.home()
    await asyncio.sleep(0.2)

    assert sim.x == 0
    assert sim.y == 0

    await driver.move_to(50.0, 75.0)
    await asyncio.sleep(0.2)

    assert sim.x == 50000
    assert sim.y == 75000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_multiple_jogs_in_sequence(driver, ruida_simulator):
    """Test that multiple jog commands work in sequence."""
    sim, host, port, jog_port = ruida_simulator

    assert await wait_for_connection(driver)

    sim.x = 0
    sim.y = 0

    await driver.jog(5000, x=10.0)
    await asyncio.sleep(0.2)

    await driver.jog(5000, y=10.0)
    await asyncio.sleep(0.2)

    await driver.jog(5000, x=-5.0)
    await asyncio.sleep(0.2)

    assert sim.x == 5000
    assert sim.y == 10000

    await driver.cleanup()


@pytest.mark.asyncio
async def test_power_settings_with_different_heads(driver, ruida_simulator):
    """Test that set_power works with different laser heads."""
    assert await wait_for_connection(driver)

    head1 = Laser()
    head1.uid = "head-1"
    head1.tool_number = 0

    head2 = Laser()
    head2.uid = "head-2"
    head2.tool_number = 1

    await driver.set_power(head1, 0.75)
    await asyncio.sleep(0.2)

    await driver.set_power(head2, 0.50)
    await asyncio.sleep(0.2)

    await driver.cleanup()


@pytest.mark.asyncio
async def test_get_encoder_returns_correct_type(driver):
    """Test that get_encoder returns RuidaEncoder."""
    encoder = driver.get_encoder()

    assert isinstance(encoder, RuidaEncoder)


@pytest.mark.asyncio
async def test_resource_uri_property(driver):
    """Test that resource_uri returns correct format."""
    driver.host = "192.168.1.100"
    driver.port = 50200
    driver.jog_port = 50207

    uri = driver.resource_uri

    assert uri == "udp://192.168.1.100:50200 (jog: 50207)"


@pytest.mark.asyncio
async def test_machine_space_wcs_properties(driver):
    """Test that machine space WCS properties return correct values."""
    wcs = driver.machine_space_wcs
    display_name = driver.machine_space_wcs_display_name

    assert wcs == "MACHINE"
    assert display_name != ""


@pytest.mark.asyncio
async def test_can_home_returns_true(driver):
    """Test that can_home returns True for all axes."""
    assert driver.can_home()
    assert driver.can_home(Axis.X)
    assert driver.can_home(Axis.Y)
    assert driver.can_home(Axis.Z)
    assert driver.can_home(Axis.X | Axis.Y)


@pytest.mark.asyncio
async def test_can_jog_returns_true(driver):
    """Test that can_jog returns True."""
    assert driver.can_jog()
    assert driver.can_jog(Axis.X)
    assert driver.can_jog(Axis.Y)


@pytest.mark.asyncio
async def test_driver_label_and_subtitle(driver):
    """Test that driver has correct label and subtitle."""
    assert "Ruida" in driver.label
    assert "UDP" in driver.label
    assert driver.subtitle != ""
    assert driver.supports_settings is False
    assert driver.reports_granular_progress is False


@pytest.mark.asyncio
async def test_disconnect_when_not_connected(driver):
    """Test that disconnect works even when not connected."""
    await driver.cleanup()

    assert not driver.is_connected


@pytest.mark.asyncio
async def test_cleanup_resets_transport(driver, ruida_simulator):
    """Test that cleanup properly resets transport objects."""
    _, host, port, jog_port = ruida_simulator
    driver._setup_implementation(host=host, port=port, response_port=0)

    await driver.cleanup()

    assert driver._udp_transport is None
    assert driver._ruida_transport is None
    assert driver._client is None


@pytest.mark.asyncio
async def test_reconnect_after_disconnect(driver, ruida_simulator):
    """Test that driver can reconnect after disconnect."""
    _, host, port, jog_port = ruida_simulator
    assert await wait_for_connection(driver)

    assert driver.is_connected

    await driver.cleanup()

    assert not driver.is_connected

    driver._setup_implementation(
        host=host, port=port, jog_port=jog_port, response_port=0
    )
    assert await wait_for_connection(driver)

    assert driver.is_connected

    await driver.cleanup()


@pytest.mark.asyncio
async def test_keepalive_maintains_connection(driver, ruida_simulator):
    """Test that periodic keepalive maintains the connection."""
    assert await wait_for_connection(driver)

    for _ in range(3):
        await asyncio.sleep(driver.KEEPALIVE_INTERVAL + 0.2)
        assert driver.is_connected, (
            "Connection should remain active with keepalive"
        )

    await driver.cleanup()


@pytest.mark.asyncio
async def test_connection_status_on_connect(driver, ruida_simulator):
    """Test that CONNECTED status is emitted after successful connection."""
    status_changes = []

    def on_status(sender, status, message):
        status_changes.append((status, message))

    driver.connection_status_changed.connect(on_status)

    assert await wait_for_connection(driver)

    statuses = [s for s, _ in status_changes]
    assert TransportStatus.CONNECTED in statuses

    await driver.cleanup()


@pytest.mark.asyncio
async def test_keepalive_timeout_behavior(driver, ruida_simulator):
    """Test that driver handles connection lifecycle correctly."""
    assert await wait_for_connection(driver)

    await asyncio.sleep(driver.KEEPALIVE_INTERVAL * 2)

    assert driver.is_connected, (
        "Connection should remain active with keepalive"
    )

    await driver.cleanup()


@pytest.mark.asyncio
async def test_position_polling_updates_state(driver, ruida_simulator):
    """Test that position polling reads current position from controller."""
    sim, host, port, jog_port = ruida_simulator

    sim.x = 50000
    sim.y = 75000

    assert await wait_for_connection(driver)

    await asyncio.sleep(driver.POSITION_POLL_INTERVAL + 0.5)

    assert driver.state.machine_pos[0] == 50.0
    assert driver.state.machine_pos[1] == 75.0

    await driver.cleanup()


@pytest.mark.asyncio
async def test_multiple_keepalive_cycles(driver, ruida_simulator):
    """Test connection stability over multiple keepalive cycles."""
    assert await wait_for_connection(driver)

    cycle_count = 0
    max_cycles = 5

    for i in range(max_cycles):
        await asyncio.sleep(driver.KEEPALIVE_INTERVAL)
        if driver.is_connected:
            cycle_count += 1

    assert cycle_count >= max_cycles - 1, (
        f"Expected {max_cycles - 1}+ successful cycles, got {cycle_count}"
    )

    await driver.cleanup()


def test_get_laser_capabilities_returns_empty_for_diode():
    driver = RuidaDriver.__new__(RuidaDriver)
    laser = Laser()
    laser.laser_type = LaserType.DIODE

    result = driver.get_laser_capabilities(laser)

    assert result == ()


def test_get_laser_capabilities_returns_pwm_for_co2():
    driver = RuidaDriver.__new__(RuidaDriver)
    laser = Laser()
    laser.laser_type = LaserType.CO2
    laser.pwm_frequency = 1000
    laser.max_pwm_frequency = 5000
    laser.pulse_width = 50
    laser.min_pulse_width = 5
    laser.max_pulse_width = 500

    result = driver.get_laser_capabilities(laser)

    assert len(result) == 1
    cap = result[0]
    assert cap.name == "PWM"
    vs = cap.varset
    freq_var = next(v for v in vs if v.key == "frequency")
    assert isinstance(freq_var, IntVar)
    assert freq_var.default == 1000
    assert freq_var.max_val == 5000
    pw_var = next(v for v in vs if v.key == "pulse_width")
    assert isinstance(pw_var, IntVar)
    assert pw_var.default == 50
    assert pw_var.min_val == 5
    assert pw_var.max_val == 500


def test_get_laser_capabilities_returns_pwm_for_fiber():
    driver = RuidaDriver.__new__(RuidaDriver)
    laser = Laser()
    laser.laser_type = LaserType.FIBER

    result = driver.get_laser_capabilities(laser)

    assert len(result) == 1
    assert result[0].name == "PWM"


def test_get_laser_capabilities_co2_with_zero_frequency():
    driver = RuidaDriver.__new__(RuidaDriver)
    laser = Laser()
    laser.laser_type = LaserType.CO2
    laser.pwm_frequency = 0
    laser.max_pwm_frequency = 0
    laser.pulse_width = 0
    laser.min_pulse_width = 0
    laser.max_pwm_frequency = 0

    result = driver.get_laser_capabilities(laser)

    assert len(result) == 1
    assert result[0].name == "PWM"
