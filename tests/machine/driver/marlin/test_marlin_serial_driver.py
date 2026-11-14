import asyncio
import time
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from raydriver.core.transport import MockTransport
from raydriver.marlin import MarlinSession
from raydriver.marlin.emulator import MarlinEmulator
from raygeo.ops import Ops

from rayforge import config
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.varset import VarSet
from rayforge.machine.driver.driver import (
    Axis,
    DeviceStatus,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
)
from rayforge.machine.driver.marlin.marlin_serial import MarlinSerialDriver
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine
from rayforge.machine.transport.serial import SerialPortPermissionError
from rayforge.pipeline.encoder.gcode import GcodeEncoder

FAST_CONFIG = {
    "handshake_timeout": 2.0,
    "handshake_poll_interval": 0.02,
    "status_poll_interval": 0.05,
    "reconnect_delay": 0.2,
    "command_timeout": 2.0,
    "connect_settle": 0.02,
}


def _make_context(tmp_path, monkeypatch):
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)

    ctx = get_context()
    ctx.initialize_lite_context(temp_machine_dir)
    ctx._dialect_mgr = DialectManager(temp_dialect_dir)
    return ctx


@pytest.fixture
def driver(tmp_path, monkeypatch):
    ctx = _make_context(tmp_path, monkeypatch)
    m = Machine(ctx)
    ctx.machine_mgr.add_machine(m)
    yield MarlinSerialDriver(ctx, m)
    context_module._context_instance = None


@pytest_asyncio.fixture
async def async_driver(tmp_path, monkeypatch):
    ctx = _make_context(tmp_path, monkeypatch)
    m = Machine(ctx)
    ctx.machine_mgr.add_machine(m)
    yield MarlinSerialDriver(ctx, m), m
    await m.shutdown()
    context_module._context_instance = None


class TestMarlinSerialDriverProperties:
    def test_driver_properties(self, driver):
        assert driver.label == "Marlin (Serial)"
        assert driver.subtitle == ("Marlin firmware via serial connection")
        assert driver.supports_settings is False
        assert driver.maturity == DriverMaturity.EXPERIMENTAL
        assert driver.reports_granular_progress is True
        assert driver.supports_probing is True

    def test_get_setup_vars(self):
        setup_vars = MarlinSerialDriver.get_setup_vars()
        assert isinstance(setup_vars, VarSet)
        keys = {v.key for v in setup_vars.vars}
        assert "port" in keys
        assert "baudrate" in keys

    def test_create_encoder(self, driver):
        machine = driver._machine
        machine.set_dialect_uid("marlin")
        encoder = MarlinSerialDriver.create_encoder(machine)
        assert isinstance(encoder, GcodeEncoder)

    def test_get_setting_vars(self, driver):
        setting_vars = driver.get_setting_vars()
        assert isinstance(setting_vars, list)
        assert len(setting_vars) == 1
        assert isinstance(setting_vars[0], VarSet)

    def test_precheck_passes(self, mocker):
        target = (
            "rayforge.machine.driver.marlin.marlin_serial"
            ".SerialTransport.check_serial_permissions_globally"
        )
        mocker.patch(target)
        MarlinSerialDriver.precheck()

    def test_precheck_permission_error(self, mocker):
        target = (
            "rayforge.machine.driver.marlin.marlin_serial"
            ".SerialTransport.check_serial_permissions_globally"
        )
        mocker.patch(
            target,
            side_effect=SerialPortPermissionError("no access"),
        )
        with pytest.raises(DriverPrecheckError):
            MarlinSerialDriver.precheck()

    def test_machine_space_wcs(self, driver):
        assert driver.machine_space_wcs == "MACHINE"

    def test_resource_uri_with_port(self, driver, machine):
        machine.set_dialect_uid("marlin")
        driver.setup(port="/dev/ttyUSB0", baudrate=115200)
        assert driver.resource_uri == driver._session.resource_uri

    def test_resource_uri_without_port(self, driver):
        assert driver.resource_uri is None

    def test_can_home(self, driver):
        assert driver.can_home() is True
        assert driver.can_home(Axis.X) is True

    def test_can_jog(self, driver):
        assert driver.can_jog() is True
        assert driver.can_jog(Axis.X) is True

    def test_setup_missing_port_raises(self, driver):
        with pytest.raises(DriverSetupError):
            driver._setup_implementation(port="", baudrate=115200)

    def test_setup_missing_baudrate_raises(self, driver):
        with pytest.raises(DriverSetupError):
            driver._setup_implementation(port="/dev/ttyUSB0", baudrate=None)


class TestMarlinSerialDriverRealSerial:
    """
    Tests that exercise the full driver + session stack against a
    MarlinEmulator: the firmware-faithful stand-in for the
    MarlinSimulator thread the Python implementation was tested
    with.  The session talks to it through a MockTransport, so
    every byte the driver sends is asserted on directly.
    """

    @pytest.fixture
    def mock(self):
        return MockTransport()

    @pytest_asyncio.fixture
    async def simulator(self, mock):
        emulator = MarlinEmulator(mock)
        task = asyncio.ensure_future(emulator.run())

        class _Sim:
            @property
            def position(self):
                return list(emulator.mpos)

            @position.setter
            def position(self, value):
                emulator.mpos = list(value)

        yield _Sim()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest_asyncio.fixture
    async def connected_driver(self, async_driver, mock, simulator):
        d, machine = async_driver
        machine.set_dialect_uid("marlin")
        d.did_setup = True
        d._session = MarlinSession.with_transport(
            mock,
            config=FAST_CONFIG,
            dialect=d._dialect_templates(),
            event_callback=d._on_session_event,
        )

        statuses = []

        def on_status(sender, status, message=None):
            statuses.append(status)

        d.connection_status_changed.connect(on_status)
        await d.connect()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(s.name == "CONNECTED" for s in statuses):
                break
            await asyncio.sleep(0.02)
        assert any(s.name == "CONNECTED" for s in statuses), (
            f"Driver never reached CONNECTED. "
            f"Got: {[s.name for s in statuses]}"
        )
        d.connection_status_changed.disconnect(on_status)

        yield d
        await d.cleanup()

    @pytest.mark.asyncio
    async def test_connection_and_handshake(self, connected_driver, simulator):
        driver = connected_driver
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if driver.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.02)
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_m114_updates_position_via_real_serial(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        simulator.position = [12.5, 3.5, 0.0]
        for _ in range(250):
            await asyncio.sleep(0.02)
            if driver.state.machine_pos[0] is not None:
                break
        assert driver.state.machine_pos[0] is not None
        assert driver.state.machine_pos[1] is not None
        assert driver.state.machine_pos[2] is not None
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_send_g28_and_wait_for_ok(self, connected_driver):
        driver = connected_driver
        response = await asyncio.wait_for(
            driver.execute_interactive_command("G28"), timeout=2.0
        )
        assert response == []

    @pytest.mark.asyncio
    async def test_home_resets_simulator_position(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        simulator.position = [50.0, 100.0, 0.0]
        await asyncio.wait_for(
            driver.execute_interactive_command("G28"), timeout=2.0
        )
        assert simulator.position == [0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_move_to_updates_simulator(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        await asyncio.wait_for(
            driver.execute_interactive_command("G0 X42.5 Y17.3 F1500"),
            timeout=2.0,
        )
        assert simulator.position[0] == pytest.approx(42.5, abs=0.1)
        assert simulator.position[1] == pytest.approx(17.3, abs=0.1)

    @pytest.mark.asyncio
    async def test_run_raw_streams_gcode(
        self, connected_driver, simulator, mock
    ):
        driver = connected_driver
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        gcode = "G0 X10\nG1 X20 Y20\nG0 X0 Y0"
        await asyncio.wait_for(driver.run_raw(gcode), timeout=5.0)
        assert simulator.position[0] == pytest.approx(0.0, abs=0.1)
        assert simulator.position[1] == pytest.approx(0.0, abs=0.1)
        job_finished.assert_called_once()
        sent = b"".join(mock.sent())
        assert b"G0 X10\n" in sent
        assert b"G1 X20 Y20\n" in sent
        assert b"G0 X0 Y0\n" in sent

    @pytest.mark.asyncio
    async def test_cancel_sends_m410_to_device(self, connected_driver, mock):
        driver = connected_driver
        gcode = "\n".join([f"G0 X{i}" for i in range(50)])
        run_task = asyncio.create_task(driver.run_raw(gcode))
        await asyncio.sleep(0.2)
        await driver.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        assert b"M410" in b"".join(mock.sent())
        assert driver._session.job_running is False

    @pytest.mark.asyncio
    async def test_cancel_stops_streaming(self, connected_driver):
        driver = connected_driver
        gcode = "\n".join([f"G0 X{i}" for i in range(50)])
        run_task = asyncio.create_task(driver.run_raw(gcode))
        await asyncio.sleep(0.2)
        await driver.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        assert driver._session.job_running is False

    @pytest.mark.asyncio
    async def test_run_with_encoded_ops(self, connected_driver, mock):
        driver = connected_driver
        machine = driver._machine
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        machine.set_active_wcs("G54")
        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)
        doc = Doc()
        encoded = driver.get_encoder().encode(ops, machine, doc)
        await asyncio.wait_for(driver.run(encoded, doc, ops), timeout=5.0)
        job_finished.assert_called_once()
        assert b"G1" in b"".join(mock.sent())

    @pytest.mark.asyncio
    async def test_connection_status_on_disconnect(self, connected_driver):
        driver = connected_driver
        assert driver._session is not None
        await driver.cleanup()
        await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_multiple_sequential_commands(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        commands = [
            "G0 X10 Y10",
            "G1 X20 Y20",
            "G0 X30 Y30",
            "G28",
        ]
        for cmd in commands:
            await asyncio.wait_for(
                driver.execute_interactive_command(cmd), timeout=2.0
            )
        assert simulator.position == [0.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_empty_run_raw(self, connected_driver):
        driver = connected_driver
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        await driver.run_raw("")
        await driver.run_raw("   \n\n  \n")
        job_finished.assert_not_called()
        assert driver._session.job_running is False

    @pytest.mark.asyncio
    async def test_cleanup_resets_job_state(self, connected_driver):
        driver = connected_driver
        await driver.cleanup()
        assert driver._session is None

    @pytest.mark.asyncio
    async def test_not_implemented_methods(self, connected_driver):
        with pytest.raises(NotImplementedError):
            await connected_driver.read_settings()
        with pytest.raises(NotImplementedError):
            await connected_driver.write_setting("k", "v")
        with pytest.raises(NotImplementedError):
            await connected_driver.read_wcs_offsets()
        with pytest.raises(NotImplementedError):
            await connected_driver.run_probe_cycle(Axis.Z, -10, 100)
        await connected_driver.set_hold(True)
        await connected_driver.set_hold(False)
