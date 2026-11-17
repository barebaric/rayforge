"""Tests for the Rust-backed SmoothieDriver."""

import asyncio

import pytest
import pytest_asyncio
from raydriver.core.transport import MockTransport
from raydriver.smoothie import SmoothieSession
from raydriver.smoothie.emulator import SmoothieEmulator
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.core.doc import Doc
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DriverSetupError,
)
from rayforge.machine.driver.smoothie import SmoothieDriver
from rayforge.pipeline.encoder.gcode import GcodeEncoder

FAST_CONFIG = {
    "handshake_timeout": 2.0,
    "handshake_poll_interval": 0.02,
    "status_poll_interval": 0.05,
    "reconnect_delay": 0.2,
    "command_timeout": 2.0,
}


class SignalTracker:
    """A helper to track calls to a blinker Signal."""

    def __init__(self, signal):
        self.calls = []
        signal.connect(self._callback, weak=False)

    def _callback(self, sender, **kwargs):
        self.calls.append({"sender": sender, "kwargs": kwargs})


class StateAwaiter:
    """Helper to wait for a specific device state."""

    def __init__(self, driver, expected_status):
        self.driver = driver
        self.expected_status = expected_status
        self.event = asyncio.Event()
        self._signal = driver.state_changed.connect(
            self._on_state_changed, weak=False
        )

    def _on_state_changed(self, sender, **kwargs):
        state = kwargs.get("state")
        if state and state.status == self.expected_status:
            self.event.set()

    async def wait(self, timeout=5.0):
        try:
            await asyncio.wait_for(self.event.wait(), timeout)
        finally:
            self.driver.state_changed.disconnect(self._signal)


@pytest.fixture
def mock():
    return MockTransport()


@pytest_asyncio.fixture
async def emulator(mock):
    """The firmware emulator answering on the mock transport."""
    emulator = SmoothieEmulator(mock)
    task = asyncio.ensure_future(emulator.run())
    yield emulator
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest_asyncio.fixture
async def driver(context_initializer, machine, mock):
    """
    Provides a configured, but not connected, SmoothieDriver instance
    wired to a SmoothieEmulator through the mock transport.  Handles
    cleanup ensuring background tasks are stopped.
    """
    machine.dialect_uid = "smoothieware"
    driver_instance = SmoothieDriver(context_initializer, machine)
    driver_instance.did_setup = True
    driver_instance._session = SmoothieSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=driver_instance._dialect_templates(),
        event_callback=driver_instance._on_session_event,
    )

    yield driver_instance

    await driver_instance.cleanup()


@pytest_asyncio.fixture
async def connected_driver(driver: SmoothieDriver, emulator):
    """An async fixture that connects a driver and handles teardown."""
    await driver.connect()
    # Allow connection and initial tasks to settle.
    await asyncio.sleep(0.1)
    yield driver
    await driver.cleanup()


class TestSmoothieDriver:
    def test_get_encoder(self, driver):
        encoder = driver.get_encoder()
        assert isinstance(encoder, GcodeEncoder)
        assert encoder.dialect.uid == "smoothieware"

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, driver, emulator, mock):
        """Test the connect and cleanup flow."""
        assert driver._session is not None
        assert not mock.is_connected

        await driver.connect()

        awaiter = StateAwaiter(driver, DeviceStatus.IDLE)
        await awaiter.wait()
        assert mock.is_connected

        await driver.cleanup()
        assert driver._session is None

    @pytest.mark.asyncio
    async def test_status_polling_and_parsing(
        self, driver: SmoothieDriver, emulator, mock
    ):
        """
        Test that the driver periodically polls and correctly parses status.
        """
        state_tracker = SignalTracker(driver.state_changed)

        await driver.connect()

        # The driver polls periodically. Wait for state update.
        for _ in range(50):
            if driver.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.1)

        assert driver.state.status == DeviceStatus.IDLE
        assert driver.state.machine_pos[0] == pytest.approx(0.0, abs=0.1)
        assert len(state_tracker.calls) > 0

    @pytest.mark.asyncio
    async def test_send_and_wait(self, connected_driver: SmoothieDriver, mock):
        """Test executing a simple command that waits for 'ok'."""
        driver = connected_driver
        await driver.home(Axis.X)
        await asyncio.sleep(0.1)
        assert b"G28 X0" in b"".join(mock.sent())

    @pytest.mark.asyncio
    async def test_run_streams_gcode(
        self, connected_driver: SmoothieDriver, mock
    ):
        """Test the full G-code streaming process for a simple job."""
        driver = connected_driver
        doc = Doc()
        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)

        job_finished_called = []
        driver.job_finished.connect(
            lambda sender: job_finished_called.append(sender), weak=False
        )
        callback_calls = []

        encoded = driver.get_encoder().encode(ops, driver._machine, doc)
        await driver.run(
            encoded,
            doc,
            ops,
            lambda op_index: callback_calls.append(op_index),
        )

        # Check that the device received the correct G-code
        received_str = b"".join(mock.sent()).decode()
        assert "G0 X10 Y10" in received_str
        assert "G1 X20 Y20" in received_str

        # Check that callbacks were fired
        assert len(job_finished_called) == 1
        assert len(callback_calls) == 2

    @pytest.mark.asyncio
    async def test_run_raw(self, connected_driver: SmoothieDriver, mock):
        """Test running a raw multi-line G-code string."""
        driver = connected_driver
        gcode = "G0 X10\nM3 S0.5\nG0 Y10"

        await driver.run_raw(gcode)

        received_str = b"".join(mock.sent()).decode()
        assert "G0 X10" in received_str
        assert "M3 S0.5" in received_str
        assert "G0 Y10" in received_str

    @pytest.mark.asyncio
    async def test_server_disconnects_unexpectedly(
        self, driver: SmoothieDriver, emulator, mock
    ):
        """
        Test that the driver handles an abrupt server-side disconnect.
        """
        status_tracker = SignalTracker(driver.connection_status_changed)

        await driver.connect()

        # Wait for IDLE state to confirm connection is fully established
        for _ in range(50):
            if driver.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.1)
        assert driver.state.status == DeviceStatus.IDLE

        # Fail the link, as an unplugged cable would.
        mock.set_write_error("unplugged")

        # Wait for the driver to detect the disconnect
        def saw_disconnect():
            statuses = [
                call["kwargs"]["status"] for call in status_tracker.calls
            ]
            return any(
                status.name in ("DISCONNECTED", "ERROR", "SLEEPING")
                for status in statuses
            )

        for _ in range(50):
            if saw_disconnect():
                break
            await asyncio.sleep(0.1)
        assert saw_disconnect()

    @pytest.mark.asyncio
    async def test_set_wcs_offset(
        self, connected_driver: SmoothieDriver, mock
    ):
        """Test setting a WCS offset for Smoothie."""
        driver = connected_driver
        await driver.set_wcs_offset("G54", 1.1, 2.2, 3.3)
        await asyncio.sleep(0.1)
        assert b"G10 L20 P1 X1.1 Y2.2 Z3.3" in b"".join(mock.sent())

    @pytest.mark.asyncio
    async def test_read_wcs_offsets_raises_not_implemented(
        self, connected_driver: SmoothieDriver
    ):
        """Test that read_wcs_offsets raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await connected_driver.read_wcs_offsets()

    @pytest.mark.asyncio
    async def test_run_probe_cycle_raises_not_implemented(
        self, connected_driver: SmoothieDriver
    ):
        """Test that run_probe_cycle raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await connected_driver.run_probe_cycle(Axis.Z, -10, 100)

    @pytest.mark.asyncio
    async def test_read_parser_state_returns_none(
        self, connected_driver: SmoothieDriver
    ):
        """Test that read_parser_state returns None (not implemented)."""
        result = await connected_driver.read_parser_state()
        assert result is None


@pytest.mark.asyncio
async def test_setup_missing_host_raises(context_initializer, machine):
    drv = SmoothieDriver(context_initializer, machine)
    with pytest.raises(DriverSetupError):
        drv._setup_implementation(host="", port=23)


@pytest.mark.asyncio
async def test_cancel_sends_ctrl_c_and_safety_off(
    context_initializer, machine, mock
):
    """Cancel aborts the job with Ctrl+C and turns the laser off."""
    machine.dialect_uid = "smoothieware"
    drv = SmoothieDriver(context_initializer, machine)
    drv.did_setup = True
    drv._session = SmoothieSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    emulator = SmoothieEmulator(mock, speed_factor=1)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        awaiter = StateAwaiter(drv, DeviceStatus.IDLE)
        await awaiter.wait()
        gcode = "\n".join([f"G1 X{i % 50} F600" for i in range(50)])
        run_task = asyncio.create_task(drv.run_raw(gcode))
        await asyncio.sleep(0.2)
        await drv.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        sent = b"".join(mock.sent())
        assert b"\x03" in sent
        assert b"M5\n" in sent
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_hold_sends_realtime_bytes(context_initializer, machine, mock):
    """Hold and resume go out as the realtime bytes '!' and '~'."""
    machine.dialect_uid = "smoothieware"
    drv = SmoothieDriver(context_initializer, machine)
    drv.did_setup = True
    drv._session = SmoothieSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        awaiter = StateAwaiter(drv, DeviceStatus.IDLE)
        await awaiter.wait()
        mock.clear_sent()
        await drv.set_hold(True)
        assert b"!" in b"".join(mock.sent())
        await drv.set_hold(False)
        assert b"~" in b"".join(mock.sent())
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_interactive_command_filters_ok_and_status(
    context_initializer, machine, mock
):
    """execute_interactive_command returns reply lines only: the
    terminating ``ok`` and status-report noise are filtered out,
    as the Python implementation did."""
    machine.dialect_uid = "smoothieware"
    drv = SmoothieDriver(context_initializer, machine)
    drv.did_setup = True
    drv._session = SmoothieSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        # Let the handshake settle so its banner is not collected
        # as a command reply.
        for _ in range(50):
            if drv.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.05)
        alpha_max = await drv.execute_interactive_command(
            "config-get alpha_max"
        )
        assert alpha_max == ["200.0000"]
        assert "ok" not in alpha_max
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass
