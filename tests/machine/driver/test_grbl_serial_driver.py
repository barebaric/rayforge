import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock, PropertyMock
from typing import cast
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.core.varset import VarSet, Var
from rayforge.machine.driver.grbl_serial import GrblSerialDriver
from rayforge.machine.transport import TransportStatus, SerialTransport
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DeviceConnectionError,
)
from rayforge.pipeline.encoder.gcode import GcodeEncoder


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


@pytest.fixture
def driver(context_initializer, machine, mock_serial_transport, mocker):
    """
    Provides a GrblSerialDriver instance with its transport already mocked.
    """
    mocker.patch(
        "rayforge.machine.driver.grbl_serial.SerialTransport.__init__",
        return_value=None,
    )

    driver_instance = GrblSerialDriver(context_initializer, machine)
    driver_instance.serial_transport = mock_serial_transport
    assert driver_instance.serial_transport is not None
    driver_instance.serial_transport.received.connect(
        driver_instance.on_serial_data_received
    )
    driver_instance.serial_transport.status_changed.connect(
        driver_instance.on_serial_status_changed
    )
    driver_instance.did_setup = True
    return driver_instance


@pytest.fixture
def doc():
    """Provides a fresh Doc instance for each test."""
    return Doc()


@pytest_asyncio.fixture
async def connected_driver(driver: GrblSerialDriver, mocker):
    """
    An async fixture that takes a driver, connects it, and handles async
    teardown.
    """
    transport_mock = driver.serial_transport
    assert transport_mock is not None

    mocker.patch.object(
        transport_mock,
        "is_connected",
        new_callable=PropertyMock,
        return_value=True,
    )

    connect_task = asyncio.create_task(driver.connect())
    await asyncio.sleep(0)

    driver.on_serial_status_changed(transport_mock, TransportStatus.CONNECTED)
    await asyncio.sleep(0)
    welcome_msg = b"Grbl 1.1h ['$' for help]\r\n"
    driver.on_serial_data_received(transport_mock, welcome_msg)
    await asyncio.sleep(0.01)
    cast(MagicMock, transport_mock.send).reset_mock()

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
        # Verify it's configured with the machine's dialect
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
    async def test_status_report_parsing(
        self, connected_driver: GrblSerialDriver
    ):
        """Test that status reports are correctly parsed and update state."""
        driver = connected_driver
        assert driver.serial_transport is not None
        state_changed_mock = MagicMock()
        driver.state_changed.send = state_changed_mock

        report = b"<Idle|MPos:10.0,20.5,-1.0|FS:500,0>\r\n"
        driver.on_serial_data_received(driver.serial_transport, report)
        await asyncio.sleep(0)

        assert driver.state.status == DeviceStatus.IDLE
        assert driver.state.machine_pos[0] == 10.0
        assert driver.state.machine_pos[1] == 20.5
        assert driver.state.machine_pos[2] == -1.0
        state_changed_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_command_ok(
        self, connected_driver: GrblSerialDriver
    ):
        """Test executing a simple command that succeeds."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None

        cmd_task = asyncio.create_task(driver._execute_command("$X"))
        await asyncio.sleep(0.01)
        cast(MagicMock, transport_mock.send).assert_called_once_with(b"$X\n")
        driver.on_serial_data_received(transport_mock, b"ok\r\n")
        response = await cmd_task
        assert response == ["ok"]

    @pytest.mark.asyncio
    async def test_execute_command_error(
        self, connected_driver: GrblSerialDriver
    ):
        """Test executing a command that returns an error."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None

        cmd_task = asyncio.create_task(driver._execute_command("G999"))
        await asyncio.sleep(0.01)
        cast(MagicMock, transport_mock.send).assert_called_once_with(b"G999\n")
        driver.on_serial_data_received(transport_mock, b"error:20\r\n")
        response = await cmd_task
        assert response == ["error:20"]

    @pytest.mark.asyncio
    async def test_run_streams_gcode_and_completes(
        self, connected_driver: GrblSerialDriver, doc
    ):
        """Test the full G-code streaming process for a simple job."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None

        ops = Ops()
        ops.add(MoveToCommand((10, 10, 0)))
        ops.add(LineToCommand((20, 20, 0)))

        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock
        callback_mock = MagicMock()

        run_task = asyncio.create_task(driver.run(ops, doc, callback_mock))

        gcode_lines = [
            b"G0 X10.000 Y10.000 Z0.000\n",
            b"G1 X20.000 Y20.000 Z0.000\n",
        ]

        for line in gcode_lines:
            await asyncio.sleep(0.01)
            send_mock = cast(MagicMock, transport_mock.send)
            send_mock.assert_any_call(line)
            driver.on_serial_data_received(transport_mock, b"ok\r\n")

        await run_task
        job_finished_mock.assert_called_once_with(driver)
        assert callback_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_run_respects_buffer_size(
        self, connected_driver: GrblSerialDriver
    ):
        """Test that the driver waits for buffer space before sending."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None
        send_mock = cast(MagicMock, transport_mock.send)

        line1 = b"G1 X10 Y10 " + b"A" * 110 + b"\n"  # approx 123 bytes
        line2 = b"G1 X20 Y20\n"  # 11 bytes
        assert len(line1) + len(line2) > 128

        run_task = asyncio.create_task(
            driver.run_raw(line1.decode() + line2.decode())
        )

        await asyncio.sleep(0.01)
        send_mock.assert_called_once_with(line1)

        await asyncio.sleep(0.05)
        send_mock.assert_called_once_with(line1)

        driver.on_serial_data_received(transport_mock, b"ok\r\n")
        await asyncio.sleep(0.01)
        send_mock.assert_called_with(line2)

        driver.on_serial_data_received(transport_mock, b"ok\r\n")
        await run_task

    @pytest.mark.asyncio
    async def test_run_handles_mid_job_error(
        self, connected_driver: GrblSerialDriver
    ):
        """Test that an error from GRBL during a job halts the stream."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None
        send_mock = cast(MagicMock, transport_mock.send)
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        gcode = "G0 X10\nG999\nG0 Y10"
        run_task = asyncio.create_task(driver.run_raw(gcode))

        await asyncio.sleep(0.01)
        # Fix: Assert the raw G-code, which is what run_raw sends
        send_mock.assert_any_call(b"G0 X10\n")
        driver.on_serial_data_received(transport_mock, b"ok\r\n")

        await asyncio.sleep(0.01)
        send_mock.assert_any_call(b"G999\n")
        driver.on_serial_data_received(transport_mock, b"error:20\r\n")

        with pytest.raises(DeviceConnectionError):
            await run_task

        send_mock.assert_any_call(b"\x18")
        job_finished_mock.assert_called_once_with(driver)

    @pytest.mark.asyncio
    async def test_cancel_stops_job(self, connected_driver: GrblSerialDriver):
        """Test that calling cancel() stops a running job."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None
        send_mock = cast(MagicMock, transport_mock.send)
        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock

        long_gcode = "\n".join([f"G0 X{i}" for i in range(50)])
        run_task = asyncio.create_task(driver.run_raw(long_gcode))
        await asyncio.sleep(0.05)

        await driver.cancel()
        send_mock.assert_any_call(b"\x18")

        try:
            await asyncio.wait_for(run_task, timeout=0.1)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        job_finished_mock.assert_called_once_with(driver)

    @pytest.mark.asyncio
    async def test_read_settings(
        self, connected_driver: GrblSerialDriver, mocker
    ):
        """Test reading and parsing device settings."""
        driver = connected_driver
        transport_mock = driver.serial_transport
        assert transport_mock is not None
        send_mock = cast(MagicMock, transport_mock.send)
        settings_read_mock = MagicMock()
        driver.settings_read.send = settings_read_mock

        # Fix: Use the key format ('0') that the regex extracts
        mock_setting_varset = VarSet(
            vars=[Var(key="0", label="$0 Step pulse", var_type=str)]
        )
        mocker.patch.object(
            driver, "get_setting_vars", return_value=[mock_setting_varset]
        )

        settings_response = b"$0=10\r\n$999=123\r\nok\r\n"
        read_task = asyncio.create_task(driver.read_settings())

        await asyncio.sleep(0.01)
        send_mock.assert_called_with(b"$$\n")
        driver.on_serial_data_received(transport_mock, settings_response)
        await read_task

        settings_read_mock.assert_called_once()
        settings = settings_read_mock.call_args.kwargs["settings"]

        # Find the VarSet by checking its keys, not the object itself
        step_pulse_varset = next(
            (s for s in settings if "0" in s.keys()), None
        )
        assert step_pulse_varset is not None
        assert step_pulse_varset["0"].value == "10"

        unknown_varset = next(
            (s for s in settings if s.title == "Unknown Settings"), None
        )
        assert unknown_varset is not None
        # Note: the key for unknown vars is also just the number string
        assert unknown_varset["999"].value == "123"
