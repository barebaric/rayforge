"""Tests for the Rust-backed MarlinSerialDriver."""

import asyncio
import time

import pytest
import pytest_asyncio
from raydriver.core.transport import MockTransport
from raydriver.marlin import MarlinSession
from raydriver.marlin.emulator import MarlinEmulator
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.core.doc import Doc
from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
)
from rayforge.machine.driver.marlin import MarlinSerialDriver
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


class SignalRecorder:
    """Collects blinker signal emissions from a driver.

    Handlers are held by strong references: blinker drops weak
    receiver references as soon as they are garbage-collected.
    """

    def __init__(self, driver):
        self.events = []
        driver.state_changed.connect(
            lambda sender, state: self.events.append(("state_changed", state)),
            weak=False,
        )
        driver.connection_status_changed.connect(
            lambda sender, status, message=None: self.events.append(
                ("connection_status_changed", {"status": status})
            ),
            weak=False,
        )
        driver.job_finished.connect(
            lambda sender: self.events.append(("job_finished", None)),
            weak=False,
        )

    def names(self):
        return [name for name, _ in self.events]

    async def wait_for(self, name: str, pred=None, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event_name, payload in self.events:
                if event_name != name:
                    continue
                if pred is None or pred(payload):
                    return payload
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"event {name!r} not observed; got {self.names()}"
        )

    async def wait_connected(self, timeout: float = 5.0):
        return await self.wait_for(
            "connection_status_changed",
            pred=lambda p: p["status"].name == "CONNECTED",
            timeout=timeout,
        )


def make_driver(context, machine) -> tuple[MarlinSerialDriver, MockTransport]:
    """Build a MarlinSerialDriver around an emulator-backed
    session, mirroring what ``_setup_implementation`` does for a
    real serial port."""
    drv = MarlinSerialDriver(context, machine)
    drv.did_setup = True
    mock = MockTransport()
    session = MarlinSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    drv._session = session
    return drv, mock


@pytest_asyncio.fixture
async def connected_driver(context_initializer, machine):
    """A driver connected to a responsive Marlin emulator.

    Yields (driver, mock, emulator, recorder).
    """
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        yield drv, mock, emulator, recorder
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("MarlinSerialDriver") is MarlinSerialDriver

    def test_metadata(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        assert drv.machine_space_wcs == "MACHINE"
        assert drv.supports_settings is False
        assert MarlinSerialDriver.maturity == DriverMaturity.EXPERIMENTAL
        assert MarlinSerialDriver.reports_granular_progress is True
        assert MarlinSerialDriver.supports_probing is True

    def test_discovery_spec(self):
        spec = MarlinSerialDriver.DISCOVERY
        assert spec is not None
        assert spec.serial is not None

    def test_setup_vars(self):
        varset = MarlinSerialDriver.get_setup_vars()
        keys = {v.key for v in varset.vars}
        assert {"port", "baudrate"} <= keys

    def test_create_encoder(self, context_initializer, machine):
        machine.set_dialect_uid("marlin")
        drv = MarlinSerialDriver(context_initializer, machine)
        encoder = drv.get_encoder()
        assert isinstance(encoder, GcodeEncoder)

    def test_setting_vars(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        varsets = drv.get_setting_vars()
        assert len(varsets) == 1

    def test_precheck_passes(self, mocker):
        mocker.patch(
            "rayforge.machine.transport.serial"
            ".SerialTransport.check_serial_permissions_globally"
        )
        MarlinSerialDriver.precheck()

    def test_precheck_permission_error(self, mocker):
        mocker.patch(
            "rayforge.machine.transport.serial"
            ".SerialTransport.check_serial_permissions_globally",
            side_effect=SerialPortPermissionError("no access"),
        )
        with pytest.raises(DriverPrecheckError):
            MarlinSerialDriver.precheck()

    def test_can_home(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        assert drv.can_home() is True
        assert drv.can_home(Axis.X) is True

    def test_can_jog(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        assert drv.can_jog() is True
        assert drv.can_jog(Axis.X) is True

    def test_setup_missing_port_raises(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        with pytest.raises(DriverSetupError):
            drv._setup_implementation(port="", baudrate=115200)

    def test_setup_missing_baudrate_raises(self, context_initializer, machine):
        drv = MarlinSerialDriver(context_initializer, machine)
        with pytest.raises(DriverSetupError):
            drv._setup_implementation(port="/dev/ttyUSB0", baudrate=None)


@pytest.mark.asyncio
async def test_resource_uri_delegates_to_session(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    assert drv.resource_uri is None
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        assert drv.resource_uri == drv._require_session().resource_uri
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_handshake_and_position_report(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        # The periodic M114 poll reports the emulator's position.
        await recorder.wait_for(
            "state_changed",
            pred=lambda state: (
                state.status == DeviceStatus.IDLE
                and state.machine_pos[0] is not None
            ),
        )
        assert drv.state.status == DeviceStatus.IDLE
        assert drv.boot_lines, "boot lines were not collected"
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_m114_tracks_emulator_position(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        emulator.mpos = [5.0, 6.0, 0.0]
        await recorder.wait_for(
            "state_changed",
            pred=lambda state: state.machine_pos[0] == 5.0,
        )
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_run_raw_streams_gcode(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        gcode = "G0 X10\nG1 X20 Y20\nG0 X0 Y0"
        await asyncio.wait_for(drv.run_raw(gcode), timeout=10.0)
        await recorder.wait_for("job_finished")
        assert emulator.mpos[0] == pytest.approx(0.0, abs=0.1)
        assert emulator.mpos[1] == pytest.approx(0.0, abs=0.1)
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_run_with_encoded_ops(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)
        doc = Doc()
        encoded = drv.get_encoder().encode(ops, machine, doc)
        await asyncio.wait_for(drv.run(encoded, doc, ops), timeout=10.0)
        await recorder.wait_for("job_finished")
        assert b"G1" in b"".join(mock.sent())
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_cancel_stops_streaming(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock, speed_factor=1)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        gcode = "\n".join([f"G0 X{i}" for i in range(50)])
        run_task = asyncio.create_task(drv.run_raw(gcode))
        await asyncio.sleep(0.2)
        await drv.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        await recorder.wait_for("job_finished")
        sent = b"".join(mock.sent())
        assert b"M410\n" in sent
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_sequential_commands(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        for cmd in ("G0 X10 Y10", "G1 X20 Y20", "G28"):
            await asyncio.wait_for(
                drv.execute_interactive_command(cmd), timeout=5.0
            )
        assert emulator.mpos == [0.0, 0.0, 0.0], (
            f"sent={b''.join(mock.sent())!r} mpos={emulator.mpos}"
        )
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_detect_unit_system(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        assert await drv.detect_unit_system() is not None
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_empty_run_raw(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        await drv.run_raw("")
        await drv.run_raw("   \n\n  \n")
        assert "job_finished" not in recorder.names()
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_not_implemented_methods(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        with pytest.raises(NotImplementedError):
            await drv.read_settings()
        with pytest.raises(NotImplementedError):
            await drv.write_setting("k", "v")
        with pytest.raises(NotImplementedError):
            await drv.read_wcs_offsets()
        with pytest.raises(NotImplementedError):
            await drv.run_probe_cycle(Axis.X, 10, 100)
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_set_hold_is_noop_with_warning(
    context_initializer, machine, caplog
):
    drv, mock = make_driver(context_initializer, machine)
    emulator = MarlinEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        with caplog.at_level("WARNING"):
            await drv.set_hold(True)
        assert "pause/resume" in caplog.text
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass
