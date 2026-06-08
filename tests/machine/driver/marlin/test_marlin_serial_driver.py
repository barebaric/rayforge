import asyncio
import os
import re
import threading
import time
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
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
from rayforge.machine.transport import TransportStatus
from rayforge.machine.transport.serial import SerialPortPermissionError
from rayforge.pipeline.encoder.gcode import GcodeEncoder

HAS_PTY = hasattr(os, "openpty")

_real_asyncio_sleep = asyncio.sleep


async def _fast_sleep(seconds):
    await _real_asyncio_sleep(0)


class MarlinSimulator:
    """
    Simulates a Marlin firmware device over the master end of a
    pty pair.  Reads G-code commands, writes realistic responses.
    """

    def __init__(self, master_fd):
        self.fd = master_fd
        self.running = False
        self.thread = None
        self.position = [0.0, 0.0, 0.0]
        self._buf = b""
        self.boot_sent = threading.Event()
        self._boot_trigger = threading.Event()
        self._set_nonblocking()

    def _set_nonblocking(self):
        import fcntl

        flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def trigger_boot(self):
        self._boot_trigger.set()

    def stop(self):
        self.running = False
        self._boot_trigger.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _run(self):
        self._boot_trigger.wait(timeout=10.0)
        if not self.running:
            return
        self._send_boot()
        self.boot_sent.set()
        while self.running:
            self._process_input()
            time.sleep(0.002)

    def _write(self, data):
        try:
            if isinstance(data, str):
                data = data.encode()
            os.write(self.fd, data)
        except OSError:
            pass

    def _send_boot(self):
        for line in [
            b"start\n",
            b" External Reset\n",
            b"Marlin 2.1.2.7\n",
            b"echo: Last Updated: 2026-01-22 | Author: (test)\n",
            b"echo: Compiled: May  7 2026\n",
            b"echo: Free Memory: 11880  PlannerBufferBytes: 1184\n",
        ]:
            self._write(line)
            time.sleep(0.01)

    def _process_input(self):
        import select

        try:
            ready, _, _ = select.select([self.fd], [], [], 0.01)
        except (OSError, ValueError):
            return
        if not ready:
            return
        try:
            data = os.read(self.fd, 4096)
        except OSError:
            return
        if not data:
            return
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            cmd = line.decode("utf-8", errors="replace").strip()
            if cmd:
                self._handle(cmd)

    def _handle(self, cmd):
        if cmd == "M114":
            x, y, z = self.position
            self._write(
                f"X:{x} Y:{y} Z:{z} "
                f"Count X:{int(x * 100)} "
                f"Y:{int(y * 100)} "
                f"Z:{int(z * 100)}\n"
            )
            self._write("ok\n")
        elif cmd == "G28" or cmd.startswith("G28 "):
            self.position = [0.0, 0.0, 0.0]
            self._write("ok\n")
        elif cmd == "M410":
            self._write("ok\n")
        elif cmd.startswith("G0 ") or cmd.startswith("G1 "):
            xm = re.search(r"X([+-]?\d+\.?\d*)", cmd)
            ym = re.search(r"Y([+-]?\d+\.?\d*)", cmd)
            if xm:
                self.position[0] = float(xm.group(1))
            if ym:
                self.position[1] = float(ym.group(1))
            self._write("ok\n")
        else:
            self._write("ok\n")


@pytest.fixture
def driver(tmp_path, monkeypatch):
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)

    ctx = get_context()
    ctx.initialize_lite_context(temp_machine_dir)
    ctx._dialect_mgr = DialectManager(temp_dialect_dir)

    m = Machine(ctx)
    ctx.machine_mgr.add_machine(m)
    yield MarlinSerialDriver(ctx, m)
    context_module._context_instance = None


@pytest_asyncio.fixture
async def async_driver(tmp_path, monkeypatch):
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    monkeypatch.setattr(config, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config, "MACHINE_DIR", temp_machine_dir)

    ctx = get_context()
    ctx.initialize_lite_context(temp_machine_dir)
    ctx._dialect_mgr = DialectManager(temp_dialect_dir)

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

    def test_resource_uri_with_port(self, driver):
        driver._port = "/dev/ttyUSB0"
        assert driver.resource_uri == "serial:///dev/ttyUSB0"

    def test_resource_uri_without_port(self, driver):
        driver._port = ""
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


@pytest.mark.skipif(not HAS_PTY, reason="Requires Unix pty support")
class TestMarlinSerialDriverRealSerial:
    """
    Tests that exercise the full driver + SerialTransport stack
    over a real virtual serial port (pty pair).  A MarlinSimulator
    thread on the master end responds to G-code commands.
    """

    @pytest.fixture
    def virtual_serial(self):
        master_fd, slave_fd = os.openpty()
        slave_name = os.ttyname(slave_fd)
        os.close(slave_fd)
        yield master_fd, slave_name
        try:
            os.close(master_fd)
        except OSError:
            pass

    @pytest.fixture
    def simulator(self, virtual_serial):
        master_fd, _ = virtual_serial
        sim = MarlinSimulator(master_fd)
        sim.start()
        yield sim
        sim.stop()

    @pytest_asyncio.fixture
    async def connected_driver(
        self,
        async_driver,
        virtual_serial,
        simulator,
        mocker,
    ):
        d, machine = async_driver
        _, slave_name = virtual_serial
        machine.set_dialect_uid("marlin")
        mocker.patch.object(asyncio, "sleep", _fast_sleep)
        d.setup(port=slave_name, baudrate=115200)

        statuses = []

        def on_status(sender, status, message=None):
            statuses.append(status)

        d.connection_status_changed.connect(on_status)
        await d.connect()

        for _ in range(20):
            await _real_asyncio_sleep(0.01)
            if d._transport and d._transport.is_connected:
                break

        simulator.trigger_boot()

        for _ in range(100):
            await _real_asyncio_sleep(0.02)
            if TransportStatus.CONNECTED in statuses:
                break

        assert TransportStatus.CONNECTED in statuses, (
            f"Driver never reached CONNECTED. "
            f"Got: {[s.name for s in statuses]}"
        )
        d.connection_status_changed.disconnect(on_status)

        d._job_running = True
        await _real_asyncio_sleep(0.2)
        d._ok_event.clear()
        d._job_running = False

        yield d

        await d.cleanup()
        await _real_asyncio_sleep(0.05)

    @pytest.mark.asyncio
    async def test_connection_and_handshake(self, connected_driver):
        driver = connected_driver
        assert driver._handshake_event.is_set()
        assert driver._transport is not None
        assert driver._transport.is_connected

    @pytest.mark.asyncio
    async def test_m114_updates_position_via_real_serial(
        self, connected_driver
    ):
        driver = connected_driver
        for _ in range(50):
            await _real_asyncio_sleep(0.02)
            if driver.state.machine_pos[0] is not None:
                break
        assert driver.state.machine_pos[0] is not None
        assert driver.state.machine_pos[1] is not None
        assert driver.state.machine_pos[2] is not None
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_send_g28_and_wait_for_ok(self, connected_driver):
        driver = connected_driver
        driver._job_running = True
        await _real_asyncio_sleep(0.1)
        response = await asyncio.wait_for(
            driver._send_and_wait("G28"), timeout=2.0
        )
        assert response == []
        driver._job_running = False

    @pytest.mark.asyncio
    async def test_home_resets_simulator_position(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        driver._job_running = True
        await _real_asyncio_sleep(0.1)
        simulator.position = [50.0, 100.0, 0.0]
        await asyncio.wait_for(driver._send_and_wait("G28"), timeout=2.0)
        assert simulator.position == [0.0, 0.0, 0.0]
        driver._job_running = False

    @pytest.mark.asyncio
    async def test_move_to_updates_simulator(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        driver._job_running = True
        await _real_asyncio_sleep(0.1)
        await asyncio.wait_for(
            driver._send_and_wait("G0 X42.5 Y17.3 F1500"),
            timeout=2.0,
        )
        assert simulator.position[0] == pytest.approx(42.5, abs=0.1)
        assert simulator.position[1] == pytest.approx(17.3, abs=0.1)
        driver._job_running = False

    @pytest.mark.asyncio
    async def test_run_raw_streams_gcode(self, connected_driver, simulator):
        driver = connected_driver
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        gcode = "G0 X10\nG1 X20 Y20\nG0 X0 Y0"
        await asyncio.wait_for(driver.run_raw(gcode), timeout=5.0)
        assert simulator.position[0] == pytest.approx(0.0, abs=0.1)
        assert simulator.position[1] == pytest.approx(0.0, abs=0.1)
        job_finished.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_sends_m410_to_device(self, connected_driver):
        driver = connected_driver
        driver._job_running = True
        await _real_asyncio_sleep(0.1)
        await driver.cancel()
        await _real_asyncio_sleep(0.1)
        assert driver._is_cancelled is True
        assert driver._job_running is False

    @pytest.mark.asyncio
    async def test_cancel_stops_streaming(self, connected_driver):
        driver = connected_driver
        gcode = "\n".join([f"G0 X{i}" for i in range(50)])
        run_task = asyncio.create_task(driver.run_raw(gcode))
        await _real_asyncio_sleep(0.2)
        await driver.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        assert driver._job_running is False

    @pytest.mark.asyncio
    async def test_run_with_encoded_ops(self, connected_driver):
        driver = connected_driver
        machine = driver._machine
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        machine.set_active_wcs("G54")
        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)
        doc = Doc()
        encoded = machine.encode_ops(ops, doc)
        await asyncio.wait_for(driver.run(encoded, doc, ops), timeout=5.0)
        job_finished.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_status_on_disconnect(self, connected_driver):
        driver = connected_driver
        assert driver._transport is not None
        assert driver._transport.is_connected
        await driver.cleanup()
        await _real_asyncio_sleep(0.05)

    @pytest.mark.asyncio
    async def test_multiple_sequential_commands(
        self, connected_driver, simulator
    ):
        driver = connected_driver
        driver._job_running = True
        await _real_asyncio_sleep(0.1)
        commands = [
            "G0 X10 Y10",
            "G1 X20 Y20",
            "G0 X30 Y30",
            "G28",
        ]
        for cmd in commands:
            await asyncio.wait_for(driver._send_and_wait(cmd), timeout=2.0)
        assert simulator.position == [0.0, 0.0, 0.0]
        driver._job_running = False

    @pytest.mark.asyncio
    async def test_empty_run_raw(self, connected_driver):
        driver = connected_driver
        job_finished = MagicMock()
        driver.job_finished.connect(lambda sender: job_finished(), weak=False)
        await driver.run_raw("")
        await driver.run_raw("   \n\n  \n")
        job_finished.assert_not_called()
        assert driver._job_running is False

    @pytest.mark.asyncio
    async def test_cleanup_resets_job_state(self, connected_driver):
        driver = connected_driver
        driver._job_running = True
        driver._is_cancelled = True
        driver._on_command_done = lambda x: None
        await driver.cleanup()
        assert driver._job_running is False
        assert driver._is_cancelled is False
        assert driver._on_command_done is None

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
