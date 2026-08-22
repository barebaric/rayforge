import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
import pytest_asyncio
from raygeo.ops import Ops

from rayforge.core.doc import Doc
from rayforge.core.varset import Var, VarSet
from rayforge.machine.driver.driver import (
    Axis,
    DeviceConnectionError,
    DeviceStatus,
)
from rayforge.machine.driver.grbl.grbl_serial import GrblSerialDriver
from rayforge.machine.transport import SerialTransport, TransportStatus
from rayforge.machine.transport.grbl import GrblSerialTransport
from rayforge.pipeline.encoder.gcode import GcodeEncoder
from rayforge.shared.units.system import UnitSystem


@pytest.fixture
def mock_serial_transport(mocker):
    """Provides a fully mocked SerialTransport INSTANCE."""
    mock = mocker.create_autospec(SerialTransport, instance=True)
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.send = AsyncMock()
    mock.received = MagicMock()
    mock.status_changed = MagicMock()
    mock.port = "/dev/ttyUSB0"
    return mock


async def wait_for_send_call(mock_send, payload, timeout=5.0):
    """Waits until mock_send has been called with payload, tolerating
    slow CI runners."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (payload,) in [c.args for c in mock_send.call_args_list]:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"send({payload!r}) not observed within {timeout:.1f}s")


@pytest.fixture
def driver(context_initializer, machine, mock_serial_transport, mocker):
    """
    Provides a GrblSerialDriver instance with its transport already
    mocked.
    """
    mocker.patch(
        "rayforge.machine.driver.grbl.grbl_serial.SerialTransport.__init__",
        return_value=None,
    )

    driver_instance = GrblSerialDriver(context_initializer, machine)
    driver_instance.grbl_transport = GrblSerialTransport(mock_serial_transport)
    assert driver_instance.grbl_transport is not None
    driver_instance.grbl_transport.received.connect(
        driver_instance.on_serial_data_received
    )
    driver_instance.grbl_transport.status_changed.connect(
        driver_instance.on_serial_status_changed
    )
    driver_instance.did_setup = True
    return driver_instance


@pytest.fixture
def doc():
    """Provides a fresh Doc instance for each test."""
    return Doc()


@pytest_asyncio.fixture
async def connected_driver(
    driver: GrblSerialDriver, mock_serial_transport, mocker
):
    """
    An async fixture that takes a driver, connects it, and handles async
    teardown.
    """
    mocker.patch.object(
        mock_serial_transport,
        "is_connected",
        new_callable=PropertyMock,
        return_value=True,
    )

    connect_task = asyncio.create_task(driver.connect())
    await asyncio.sleep(0)

    driver.on_serial_status_changed(
        mock_serial_transport, TransportStatus.CONNECTED
    )
    await asyncio.sleep(0)
    welcome_msg = b"Grbl 1.1h ['$' for help]\r\n"
    driver.on_serial_data_received(mock_serial_transport, welcome_msg)
    await asyncio.sleep(0.01)
    version_response = b"[VER:1.1h:]\r\nok\r\n"
    driver.on_serial_data_received(mock_serial_transport, version_response)
    await asyncio.sleep(0.01)
    mock_serial_transport.send.reset_mock()

    yield driver

    await driver.cleanup()
    if not connect_task.done():
        connect_task.cancel()
    await asyncio.sleep(0.01)


class TestGrblSerialDriver:
    def test_get_encoder(self, driver: GrblSerialDriver):
        """Test that get_encoder returns a GcodeEncoder instance."""
        encoder = driver.get_encoder()
        assert isinstance(encoder, GcodeEncoder)
        assert driver._machine.dialect is not None
        assert encoder.dialect.uid == driver._machine.dialect.uid

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, driver: GrblSerialDriver):
        """Test the connect and cleanup flow."""
        assert driver._connection_task is None
        await driver.connect()
        assert driver._connection_task is not None

        await driver.cleanup()
        await asyncio.sleep(0.01)
        assert driver._connection_task is None

    @pytest.mark.asyncio
    async def test_handshake_timeout_on_phantom_port(
        self, driver: GrblSerialDriver, mock_serial_transport, mocker
    ):
        """Test connection fails when device doesn't respond."""
        mocker.patch.object(
            mock_serial_transport,
            "is_connected",
            new_callable=PropertyMock,
            return_value=True,
        )

        status_mock = MagicMock()
        driver.connection_status_changed.send = status_mock

        await driver.connect()
        sent_statuses = []
        for _ in range(50):
            await asyncio.sleep(0.1)
            sent_statuses = [
                call[1].get("status") for call in status_mock.call_args_list
            ]
            if TransportStatus.ERROR in sent_statuses:
                break
        assert TransportStatus.ERROR in sent_statuses

        await driver.cleanup()

    @pytest.mark.asyncio
    async def test_status_report_parsing(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that status reports are correctly parsed."""
        driver = connected_driver
        state_changed_mock = MagicMock()
        driver.state_changed.send = state_changed_mock

        report = b"<Idle|MPos:10.0,20.5,-1.0|FS:500,0>\r\n"
        driver.on_serial_data_received(mock_serial_transport, report)
        await asyncio.sleep(0)

        assert driver.state.status == DeviceStatus.IDLE
        assert driver.state.machine_pos[0] == 10.0
        assert driver.state.machine_pos[1] == 20.5
        assert driver.state.machine_pos[2] == -1.0
        state_changed_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_command_ok(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test executing a simple command that succeeds."""
        driver = connected_driver

        cmd_task = asyncio.create_task(driver._execute_command("$X"))
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$X\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        response = await cmd_task
        assert response == ["ok"]

    @pytest.mark.asyncio
    async def test_execute_command_error(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test executing a command that returns an error."""
        driver = connected_driver

        cmd_task = asyncio.create_task(driver._execute_command("G999"))
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"G999\n")
        driver.on_serial_data_received(mock_serial_transport, b"error:20\r\n")
        response = await cmd_task
        assert response == ["error:20"]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_set_hold_updates_device_state(
        self, connected_driver: GrblSerialDriver
    ):
        """Pausing must flag the device state as HOLD so the UI can offer a
        resume, even though status polling is disabled while a job runs and
        the firmware's HOLD status is never observed."""
        driver = connected_driver
        driver._job_running = True

        await driver.set_hold(True)
        assert driver._is_holding is True
        assert driver.state.status == DeviceStatus.HOLD

        await driver.set_hold(False)
        assert driver._is_holding is False
        assert driver.state.status == DeviceStatus.RUN

    @pytest.mark.asyncio
    async def test_job_start_updates_device_state_without_polls(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """With status polling disabled during jobs (the default), no
        firmware reports arrive while a job runs, so the driver must
        reflect RUN at job start itself -- otherwise the UI keeps
        showing Idle during the whole job."""
        driver = connected_driver
        assert driver._poll_status_while_running is False

        driver.on_serial_data_received(
            mock_serial_transport, b"<Idle|MPos:0,0,0|FS:0,0>\r\n"
        )
        await asyncio.sleep(0)
        assert driver.state.status == DeviceStatus.IDLE

        state_changed_mock = MagicMock()
        driver.state_changed.send = state_changed_mock

        run_task = asyncio.create_task(driver.run_raw("G0 X10"))
        await asyncio.sleep(0.01)

        assert driver._job_running is True
        assert driver.state.status == DeviceStatus.RUN
        state_changed_mock.assert_called_once()

        # After the job ends, the first status report (polling has
        # resumed) must correct the state back to Idle.
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task
        assert driver._job_running is False
        driver.on_serial_data_received(
            mock_serial_transport, b"<Idle|MPos:0,0,0|FS:0,0>\r\n"
        )
        await asyncio.sleep(0)
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_job_start_does_not_mask_alarm(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Starting a job while the machine is in ALARM must not
        replace the ALARM state; the job aborts immediately."""
        driver = connected_driver
        driver.on_serial_data_received(
            mock_serial_transport, b"<Alarm|MPos:0,0,0|FS:0,0>\r\n"
        )
        await asyncio.sleep(0)
        assert driver.state.status == DeviceStatus.ALARM

        run_task = asyncio.create_task(driver.run_raw("G0 X10"))
        await asyncio.sleep(0.01)

        assert driver.state.status == DeviceStatus.ALARM
        assert driver._job_running is False
        await run_task

    @pytest.mark.asyncio
    async def test_run_raw_sends_realtime_commands_directly(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """GRBL realtime characters (~, !, ?) are executed by the
        firmware on receipt and never acknowledged, so they must
        bypass the streaming protocol and must not start a job (a
        console '~' that clobbered job state once wedged a paused
        machine for good)."""
        driver = connected_driver
        driver.on_serial_data_received(
            mock_serial_transport, b"<Idle|MPos:0,0,0|FS:0,0>\r\n"
        )
        await asyncio.sleep(0)
        mock_serial_transport.send.reset_mock()

        await driver.run_raw("~")

        sent = [c.args[0] for c in mock_serial_transport.send.await_args_list]
        assert b"~" in sent
        assert driver._job_running is False
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_run_raw_streams_mixed_realtime_and_gcode(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Realtime lines are sent directly while remaining lines are
        streamed as a regular job."""
        driver = connected_driver
        mock_serial_transport.send.reset_mock()

        run_task = asyncio.create_task(driver.run_raw("G0 X10\n~\n"))
        await asyncio.sleep(0.01)

        sent = [c.args[0] for c in mock_serial_transport.send.await_args_list]
        assert b"~" in sent
        assert b"G0 X10\n" in sent
        assert driver._job_running is True
        assert driver.state.status == DeviceStatus.RUN

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task
        assert driver._job_running is False

    @pytest.mark.asyncio
    async def test_run_streams_gcode_and_completes(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, doc
    ):
        """Test the full G-code streaming process for a simple job."""
        driver = connected_driver

        driver._machine.set_active_wcs("G54")

        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)

        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock
        callback_mock = MagicMock()

        encoded = driver.get_encoder().encode(ops, driver._machine, doc)
        run_task = asyncio.create_task(
            driver.run(encoded, doc, ops, callback_mock)
        )

        gcode_lines = [
            b"G0 X10 Y10\n",
            b"G1 X20 Y20\n",
        ]

        for line in gcode_lines:
            await asyncio.sleep(0.01)
            mock_serial_transport.send.assert_any_call(line)
            driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")

        await run_task
        job_finished_mock.assert_called_once_with(driver)
        assert callback_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_run_respects_buffer_size(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that the driver waits for buffer space before sending."""
        driver = connected_driver

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"
        line2 = b"G1 X20 Y20\n"
        assert len(line1) + len(line2) > 128

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(line1)

        await asyncio.sleep(0.05)
        mock_serial_transport.send.assert_called_once_with(line1)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_with(line2)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task

    @pytest.mark.asyncio
    async def test_status_polling_continues_during_buffer_stall(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Status polls must not starve while the streamer waits for
        buffer space: the gcode send holds _cmd_lock for the entire
        wait, so polling for the lock would leave the driver state
        stale (e.g. HOLD unobserved while a job is paused)."""
        driver = connected_driver
        driver._poll_status_while_running = True

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"
        line2 = b"G1 X20 Y20\n"
        assert len(line1) + len(line2) > 127

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        # Wait for line1 to be sent; the streamer must now be blocked
        # inside send_gcode(), holding _cmd_lock while waiting for
        # buffer space for line2.
        await asyncio.sleep(0.05)
        mock_serial_transport.send.assert_called_once_with(line1)
        assert driver._job_running is True

        mock_serial_transport.send.reset_mock()
        await asyncio.sleep(1.2)
        mock_serial_transport.send.assert_any_call(b"?")

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_with(line2)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task

    @pytest.mark.asyncio
    async def test_buffer_stall_retries_while_machine_responds(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """A busy machine that answers status polls must never be
        aborted as dead, no matter how long the buffer stays full
        (regression: false ALARM:3 aborts during slow moves)."""
        driver = connected_driver

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"
        line2 = b"G1 X20 Y20\n"
        assert len(line1) + len(line2) > 127

        async def respond_with_run(data):
            driver.on_serial_data_received(
                mock_serial_transport, b"<Run|MPos:1,2,3|FS:100,0>\r\n"
            )
            return 0

        assert driver.grbl_transport is not None
        driver.grbl_transport.send_poll = respond_with_run
        driver.STALL_TIMEOUT_DEFAULT = 0.05
        driver.POLL_RESPONSE_ATTEMPTS = 3
        driver.POLL_RESPONSE_INTERVAL = 0.01

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        # Many stall cycles far exceed UNANSWERED_POLL_LIMIT; the
        # machine answering each poll must keep the job alive.
        await asyncio.sleep(1.0)
        assert driver._job_running is True
        assert driver._job_exception is None
        assert driver._consecutive_unanswered_polls == 0

        # Unblock: ack both lines and let the job finish normally.
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await wait_for_send_call(mock_serial_transport.send, line2)
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task

    @pytest.mark.asyncio
    async def test_buffer_stall_counts_uninterpretable_report_as_alive(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """A status report that cannot be interpreted (unknown state
        word) still proves the device is alive: it must not count as
        an unanswered poll and trigger a dead-device abort."""
        driver = connected_driver

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"
        line2 = b"G1 X20 Y20\n"

        async def respond_with_unknown_state(data):
            driver.on_serial_data_received(
                mock_serial_transport, b"<Foo|MPos:1,2,3|FS:0,0>\r\n"
            )
            return 0

        assert driver.grbl_transport is not None
        driver.grbl_transport.send_poll = respond_with_unknown_state
        driver.STALL_TIMEOUT_DEFAULT = 0.05
        driver.POLL_RESPONSE_ATTEMPTS = 3
        driver.POLL_RESPONSE_INTERVAL = 0.01

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        await asyncio.sleep(0.5)
        assert driver._raw_grbl_status == DeviceStatus.UNKNOWN
        assert driver._consecutive_unanswered_polls == 0
        assert driver._job_running is True
        assert driver._job_exception is None

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await run_task

    @pytest.mark.asyncio
    async def test_buffer_stall_aborts_when_device_stops_responding(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """A device that stops responding to status polls entirely
        must abort the job instead of stalling forever, even with
        deadlock detection disabled."""
        driver = connected_driver

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"
        line2 = b"G1 X20 Y20\n"

        driver.STALL_TIMEOUT_DEFAULT = 0.05
        driver.POLL_RESPONSE_ATTEMPTS = 2
        driver.POLL_RESPONSE_INTERVAL = 0.01

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        try:
            await asyncio.wait_for(run_task, timeout=10.0)
        except (asyncio.CancelledError, DeviceConnectionError):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert isinstance(driver._job_exception, DeviceConnectionError)
        assert "stopped responding" in str(driver._job_exception)
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

    @pytest.mark.asyncio
    async def test_drain_phase_aborts_when_device_stops_responding(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Total silence after all gcode was sent (ack drain phase)
        must also abort instead of retrying forever."""
        driver = connected_driver

        driver.STALL_TIMEOUT_DEFAULT = 0.05
        driver.POLL_RESPONSE_ATTEMPTS = 2
        driver.POLL_RESPONSE_INTERVAL = 0.01

        # Send a single short line, never ack it. All gcode is sent,
        # then _drain_pending_acks polls; silence must abort.
        run_task = asyncio.create_task(driver.run_raw("G0 X10"))

        try:
            await asyncio.wait_for(run_task, timeout=10.0)
        except (asyncio.CancelledError, DeviceConnectionError):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert "stopped responding" in str(driver._job_exception)
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

    @pytest.mark.asyncio
    async def test_run_handles_mid_job_error(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that an error from GRBL during a job halts the stream."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        gcode = "G0 X10\nG999\nG0 Y10"
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_any_call(b"G0 X10\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_any_call(b"G999\n")
        driver.on_serial_data_received(mock_serial_transport, b"error:20\r\n")

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert isinstance(driver._job_exception, DeviceConnectionError)
        assert "error:20" in str(driver._job_exception)
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_comments_stripped_from_gcode(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that comments are stripped before sending to device."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        gcode = (
            "G0 X10 ; rapid move\n"
            "(this is a comment line)\n"
            "G1 X20 (cut move) Y30\n"
            "; pure comment\n"
            "G0 X40"
        )
        run_task = asyncio.create_task(driver.run_raw(gcode))

        expected = [
            b"G0 X10\n",
            b"G1 X20  Y30\n",
            b"G0 X40\n",
        ]
        for cmd in expected:
            await asyncio.sleep(0.01)
            mock_serial_transport.send.assert_any_call(cmd)
            driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")

        await run_task
        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"G0 X10\n")
        mock_serial_transport.send.assert_any_call(b"G1 X20  Y30\n")
        mock_serial_transport.send.assert_any_call(b"G0 X40\n")

        try:
            await asyncio.wait_for(run_task, timeout=0.1)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        job_finished_mock.assert_called_once_with(driver)

    @pytest.mark.asyncio
    async def test_read_settings(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, mocker
    ):
        """Test reading and parsing device settings."""
        driver = connected_driver
        settings_read_mock = MagicMock()
        driver.settings_read.send = settings_read_mock

        mock_setting_varset = VarSet(
            vars=[Var(key="0", label="$0 Step pulse", var_type=str)]
        )
        mocker.patch.object(
            driver,
            "get_setting_vars",
            return_value=[mock_setting_varset],
        )

        settings_response = b"$0=10\r\n$999=123\r\nok\r\n"
        read_task = asyncio.create_task(driver.read_settings())

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_with(b"$$\n")
        driver.on_serial_data_received(
            mock_serial_transport, settings_response
        )
        await read_task

        settings_read_mock.assert_called_once()
        settings = settings_read_mock.call_args.kwargs["settings"]

        step_pulse_varset = next(
            (s for s in settings if "0" in s.keys()),  # noqa: SIM118
            None,
        )
        assert step_pulse_varset is not None
        assert step_pulse_varset["0"].value == "10"

        unknown_varset = next(
            (s for s in settings if s.title == "Unknown Settings"),
            None,
        )
        assert unknown_varset is not None
        assert unknown_varset["999"].value == "123"

    @pytest.mark.asyncio
    async def test_set_wcs_offset(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, mocker
    ):
        """Test setting a WCS offset."""
        driver = connected_driver

        mocker.patch(
            "rayforge.machine.driver.grbl.grbl_serial.gcode_to_p_number",
            return_value=2,
        )

        cmd_task = asyncio.create_task(
            driver.set_wcs_offset("G55", 10.5, -20.0, 0.1)
        )

        done, _pending = await asyncio.wait([cmd_task], timeout=0.1)

        assert cmd_task not in done

        mock_serial_transport.send.assert_called_once_with(
            b"G10 L2 P2 X10.5 Y-20.0 Z0.1\n"
        )

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")

        await cmd_task

    @pytest.mark.asyncio
    async def test_read_wcs_offsets(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test reading and parsing WCS offsets."""
        driver = connected_driver

        response_data = (
            b"[G54:1.000,2.000,3.000]\r\n[G55:4.000,5.000,6.000]\r\nok\r\n"
        )

        cmd_task = asyncio.create_task(driver.read_wcs_offsets())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$#\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        offsets = await cmd_task
        assert offsets["G54"] == (1.0, 2.0, 3.0)
        assert offsets["G55"] == (4.0, 5.0, 6.0)

    @pytest.mark.asyncio
    async def test_read_wcs_offsets_without_z(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test reading WCS offsets when machine omits Z coordinate."""
        driver = connected_driver

        response_data = b"[G54:1.000,2.000]\r\n[G55:4.000,5.000]\r\nok\r\n"

        cmd_task = asyncio.create_task(driver.read_wcs_offsets())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$#\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        offsets = await cmd_task
        assert offsets["G54"] == (1.0, 2.0, 0.0)
        assert offsets["G55"] == (4.0, 5.0, 0.0)

    @pytest.mark.asyncio
    async def test_probe_cycle_success(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test a successful probing cycle."""
        driver = connected_driver

        response_data = b"[PRB:10.123,20.456,-0.500:1]\r\nok\r\n"

        cmd_task = asyncio.create_task(
            driver.run_probe_cycle(Axis.Z, -15, 100)
        )
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"G38.2 Z-15 F100\n"
        )
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result == (10.123, 20.456, -0.5)

    @pytest.mark.asyncio
    async def test_probe_cycle_failure(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test a probing cycle that does not trigger."""
        driver = connected_driver

        response_data = b"[PRB:0.000,0.000,0.000:0]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.run_probe_cycle(Axis.X, 20, 150))
        await asyncio.sleep(0.01)
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result is None

    @pytest.mark.asyncio
    async def test_read_parser_state_g54(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test read_parser_state parses $G response with G54."""
        driver = connected_driver

        response_data = b"[G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.read_parser_state())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$G\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result == "G54"

    @pytest.mark.asyncio
    async def test_read_parser_state_g59(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test read_parser_state parses $G response with G59."""
        driver = connected_driver

        response_data = b"[G59 G17 G21 G90 G94 M5 M9 T0 F0 S0]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.read_parser_state())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$G\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result == "G59"

    @pytest.mark.asyncio
    async def test_read_parser_state_no_wcs_found(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test read_parser_state returns None when no G54-G59 found."""
        driver = connected_driver

        response_data = b"[G17 G21 G90 G94 M5 M9 T0 F0 S0]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.read_parser_state())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$G\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result is None

    @pytest.mark.asyncio
    async def test_read_parser_state_connection_error(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, mocker
    ):
        """Test read_parser_state returns None on connection error."""
        driver = connected_driver
        from rayforge.machine.driver.driver import (
            DeviceConnectionError,
        )

        execute_command_mock = mocker.patch.object(
            driver,
            "execute_interactive_command",
            side_effect=DeviceConnectionError("Connection lost"),
        )

        result = await driver.read_parser_state()
        assert result is None
        execute_command_mock.assert_called_once_with("$G")

    @pytest.mark.asyncio
    async def test_alarm_stops_sending_and_driver_state_consistent(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that ALARM stops G-code sending and driver is consistent."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        gcode_lines = [f"G0 X{i}" for i in range(10)]
        gcode = "\n".join(gcode_lines)
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)

        first_commands = [b"G0 X0\n", b"G0 X1\n", b"G0 X2\n"]
        for cmd in first_commands:
            mock_serial_transport.send.assert_any_call(cmd)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        mock_serial_transport.send.reset_mock()

        driver.on_serial_data_received(mock_serial_transport, b"ALARM:1\r\n")
        await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert isinstance(driver._job_exception, DeviceConnectionError)
        assert "ALARM" in str(driver._job_exception)

        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

        job_finished_mock.assert_called_once_with(driver)

        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_alarm_in_buffer_wait_stops_sending(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that ALARM while waiting for buffer space stops sending."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        long_line = "G1 X0 " + "A" * 120
        gcode = f"{long_line}\nG0 X10\nG0 Y10"
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once()
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        driver.on_serial_data_received(mock_serial_transport, b"ALARM:2\r\n")
        await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_alarm_state_check_stops_sending(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that status ALARM stops sending in stream loop."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock
        state_changed_mock = MagicMock()
        driver.state_changed.send = state_changed_mock

        gcode_lines = [f"G0 X{i}" for i in range(5)]
        gcode = "\n".join(gcode_lines)
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_any_call(b"G0 X0\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        alarm_report = b"<Alarm|MPos:0,0,0|FS:0,0>\r\n"
        driver.on_serial_data_received(mock_serial_transport, alarm_report)
        await asyncio.sleep(0.01)

        assert driver.state.status == DeviceStatus.ALARM

        mock_serial_transport.send.reset_mock()
        await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_alarm_clears_command_queue(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Test that alarm clears both streaming and command queues."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        gcode = "G0 X0\nG0 X1\nG0 X2"
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_any_call(b"G0 X0\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        driver.on_serial_data_received(mock_serial_transport, b"ALARM:3\r\n")
        await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0
        assert driver._job_running is False

        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_alarm_during_job_with_callbacks(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, doc
    ):
        """Test alarm handling with on_command_done callbacks."""
        driver = connected_driver
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock
        callback_mock = MagicMock()

        driver._machine.set_active_wcs("G54")

        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)
        ops.line_to(30, 30, 0)

        encoded = driver.get_encoder().encode(ops, driver._machine, doc)
        run_task = asyncio.create_task(
            driver.run(encoded, doc, ops, callback_mock)
        )

        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_any_call(b"G0 X10 Y10\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        mock_serial_transport.send.assert_any_call(b"G1 X20 Y20\n")
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await asyncio.sleep(0.01)

        driver.on_serial_data_received(mock_serial_transport, b"ALARM:4\r\n")
        await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(run_task, timeout=0.5)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver._job_exception is not None
        assert driver.grbl_transport is not None
        assert driver.grbl_transport.pending_queue.empty()
        assert driver.grbl_transport.buffer_count == 0

        job_finished_mock.assert_called_once_with(driver)
        mock_serial_transport.send.assert_any_call(b"\x18")

    @pytest.mark.asyncio
    async def test_move_to_metric_unchanged(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Metric machine: move_to sends mm values unchanged."""
        driver = connected_driver
        assert driver._machine.unit_system == UnitSystem.METRIC

        cmd_task = asyncio.create_task(driver.move_to(10.5, 20.0))
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"$J=G90 G21 F1500 X10.5 Y20.0\n"
        )
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await cmd_task

    @pytest.mark.asyncio
    async def test_move_to_imperial_converts(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Imperial machine: move_to converts mm values to inches."""
        driver = connected_driver
        driver._machine.unit_system = UnitSystem.IMPERIAL

        cmd_task = asyncio.create_task(driver.move_to(25.4, 50.8))
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"$J=G90 G21 F59.0551 X1.0 Y2.0\n"
        )
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await cmd_task

    @pytest.mark.asyncio
    async def test_jog_imperial_converts(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Imperial machine: jog converts speed and deltas to inches."""
        driver = connected_driver
        driver._machine.unit_system = UnitSystem.IMPERIAL

        cmd_task = asyncio.create_task(driver.jog(1500, x=25.4, y=50.8))
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"$J=G91 G21 F59.0551 X1.0 Y2.0\n"
        )
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await cmd_task

    @pytest.mark.asyncio
    async def test_set_wcs_offset_imperial_converts(
        self, connected_driver: GrblSerialDriver, mock_serial_transport, mocker
    ):
        """Imperial machine: set_wcs_offset converts offsets to inches."""
        driver = connected_driver
        driver._machine.unit_system = UnitSystem.IMPERIAL

        mocker.patch(
            "rayforge.machine.driver.grbl.grbl_serial.gcode_to_p_number",
            return_value=2,
        )

        cmd_task = asyncio.create_task(
            driver.set_wcs_offset("G55", 25.4, 50.8, 12.7)
        )
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"G10 L2 P2 X1.0 Y2.0 Z0.5\n"
        )
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await cmd_task

    @pytest.mark.asyncio
    async def test_probe_cycle_imperial_converts(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Imperial machine: probe travel and feed rate are in inches."""
        driver = connected_driver
        driver._machine.unit_system = UnitSystem.IMPERIAL

        cmd_task = asyncio.create_task(
            driver.run_probe_cycle(Axis.Z, -25.4, 2540)
        )
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(
            b"G38.2 Z-1.0 F100.0\n"
        )
        driver.on_serial_data_received(mock_serial_transport, b"ok\r\n")
        await cmd_task

    @pytest.mark.asyncio
    async def test_probe_cycle_result_converted_to_mm(
        self, connected_driver: GrblSerialDriver, mock_serial_transport
    ):
        """Probe results reported in inches are converted back to mm."""
        driver = connected_driver
        driver._machine.unit_system = UnitSystem.IMPERIAL
        driver._report_in_inches = True

        response_data = b"[PRB:1.000,2.000,-0.500:1]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.run_probe_cycle(Axis.Z, -1, 100))
        await asyncio.sleep(0.01)
        driver.on_serial_data_received(mock_serial_transport, response_data)

        result = await cmd_task
        assert result == pytest.approx((25.4, 50.8, -12.7))

    @pytest.mark.asyncio
    async def test_detect_unit_system_sets_report_in_inches(
        self, driver: GrblSerialDriver, mocker
    ):
        """detect_unit_system caches the $13 reporting flag."""
        mocker.patch.object(
            driver,
            "execute_interactive_command",
            new=AsyncMock(return_value=["$0=10", "$13=1", "$20=0"]),
        )

        detected = await driver.detect_unit_system()
        assert detected == UnitSystem.IMPERIAL
        assert driver._report_in_inches is True

    @pytest.mark.asyncio
    async def test_detect_unit_system_metric_flag(
        self, driver: GrblSerialDriver, mocker
    ):
        """detect_unit_system clears the flag when $13 is zero."""
        mocker.patch.object(
            driver,
            "execute_interactive_command",
            new=AsyncMock(return_value=["$13=0"]),
        )

        detected = await driver.detect_unit_system()
        assert detected == UnitSystem.METRIC
        assert driver._report_in_inches is False

    @pytest.mark.asyncio
    async def test_read_settings_sets_report_in_inches(
        self, driver: GrblSerialDriver, mocker
    ):
        """read_settings caches the $13 reporting flag."""
        mocker.patch.object(
            driver,
            "execute_interactive_command",
            new=AsyncMock(return_value=["$13=1"]),
        )

        await driver.read_settings()
        assert driver._report_in_inches is True

    @pytest.mark.asyncio
    async def test_read_wcs_offsets_converts_when_report_inches(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """WCS offsets reported in inches are converted back to mm."""
        driver = connected_driver
        driver._report_in_inches = True

        response_data = b"[G54:1.000,2.000,3.000]\r\nok\r\n"
        cmd_task = asyncio.create_task(driver.read_wcs_offsets())
        await asyncio.sleep(0.01)
        mock_serial_transport.send.assert_called_once_with(b"$#\n")
        driver.on_serial_data_received(mock_serial_transport, response_data)

        offsets = await cmd_task
        assert offsets["G54"] == pytest.approx((25.4, 50.8, 76.2))

    @pytest.mark.asyncio
    async def test_status_report_converted_when_report_inches(
        self,
        connected_driver: GrblSerialDriver,
        mock_serial_transport,
    ):
        """Status positions reported in inches are converted back to mm."""
        driver = connected_driver
        driver._report_in_inches = True

        report = b"<Idle|MPos:1.0,2.0,0.5|WCO:0.5,1.0,0.25>\r\n"
        driver.on_serial_data_received(mock_serial_transport, report)
        await asyncio.sleep(0)

        assert driver.state.machine_pos == pytest.approx((25.4, 50.8, 12.7))
        assert driver.state.work_pos == pytest.approx((12.7, 25.4, 6.35))

    @pytest.mark.asyncio
    async def test_report_in_inches_defaults_to_false(
        self, driver: GrblSerialDriver
    ):
        assert driver._report_in_inches is False
