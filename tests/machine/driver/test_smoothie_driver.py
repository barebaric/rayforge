import asyncio
import logging
import socket
import pytest
import pytest_asyncio
from unittest.mock import MagicMock

from rayforge.machine.driver.smoothie import SmoothieDriver
from rayforge.machine.transport import TransportStatus
from rayforge.machine.driver.driver import DeviceStatus, Axis
from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand

logger = logging.getLogger(__name__)


class SignalTracker:
    """A helper to track calls to a blinker Signal."""

    def __init__(self, signal):
        self.calls = []
        # Use weak=False to ensure the callback persists during the test
        signal.connect(self._callback, weak=False)

    def _callback(self, sender, **kwargs):
        self.calls.append({"sender": sender, "kwargs": kwargs})


class MockSmoothieServer:
    """
    A mock Telnet server that behaves like a Smoothieware controller.
    """

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.port = port
        self.server = None
        self._tasks = set()
        self._writers = set()
        self.received_data = bytearray()
        self._stopping = False

    async def start(self):
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        self.host, self.port = self.server.sockets[0].getsockname()
        self._stopping = False
        return self.host, self.port

    async def stop(self):
        self._stopping = True

        # 1. Stop the listening server first
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        # 2. Aggressively close all active writers to break connections
        # We access the socket to force a shutdown, which ensures the
        # client receives an EOF/Error immediately.
        for writer in list(self._writers):
            try:
                sock = writer.get_extra_info("socket")
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except (OSError, AttributeError):
                        pass
                    sock.close()
            except Exception:
                pass

            try:
                writer.close()
            except Exception:
                pass

        # 3. Cancel all client tasks
        for task in list(self._tasks):
            task.cancel()

        # 4. Wait for tasks to cleanup, but don't hang forever
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "MockSmoothieServer: Timeout waiting for tasks to stop"
                )

        self._tasks.clear()
        self._writers.clear()

    async def _handle_client(self, reader, writer):
        task = asyncio.current_task()
        self._tasks.add(task)
        self._writers.add(writer)

        try:
            writer.write(b"Smoothie\n")
            await writer.drain()

            while not self._stopping:
                try:
                    # Use timeout to allow checking self._stopping periodically
                    # and to prevent hanging on read during shutdown
                    data = await asyncio.wait_for(
                        reader.read(1024), timeout=0.5
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    break

                if not data:
                    break

                self.received_data.extend(data)

                # Process data for responses
                # Note: On Windows/CI, packet coalescing may merge '?'
                # and commands.
                # We must check for both independently.

                # 1. Status Poll Response
                if b"?" in data:
                    writer.write(b"<Idle|MPos:1.2,3.4,5.6|FS:100,0>\n")

                # 2. Command Acknowledgement
                # Strip '?' and whitespace to see if there is a real command
                # payload
                cmd_part = data.replace(b"?", b"").strip()
                if cmd_part:
                    # If we received a command (like G28), send 'ok'
                    writer.write(b"ok\n")

                if b"?" in data or cmd_part:
                    await writer.drain()

        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._stopping:
                logger.error(f"Mock server client error: {e}")
        finally:
            self._writers.discard(writer)
            self._tasks.discard(task)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


@pytest_asyncio.fixture
async def smoothie_server():
    """Manages the lifecycle of the MockSmoothieServer for a test."""
    server = MockSmoothieServer()
    await server.start()
    yield server
    await server.stop()


@pytest_asyncio.fixture
async def driver(context_initializer, machine, smoothie_server):
    """
    Provides a configured, but not connected, SmoothieDriver instance.
    Handles cleanup ensuring background tasks are stopped.
    """
    host, port = smoothie_server.host, smoothie_server.port
    driver_instance = SmoothieDriver(context_initializer, machine)
    driver_instance.setup(host=host, port=port)

    yield driver_instance

    # Robust cleanup to prevent test leakage
    await driver_instance.cleanup()
    # Explicitly cancel connection task if it's still running (e.g. test
    # failure)
    if (
        driver_instance._connection_task
        and not driver_instance._connection_task.done()
    ):
        driver_instance._connection_task.cancel()
        try:
            await driver_instance._connection_task
        except asyncio.CancelledError:
            pass


@pytest_asyncio.fixture
async def connected_driver(driver: SmoothieDriver):
    """An async fixture that connects a driver and handles teardown."""
    await driver.connect()
    # Allow connection and initial tasks to settle. Increased for Windows CI.
    await asyncio.sleep(0.1)
    yield driver
    # The driver fixture handles the actual cleanup
    await driver.cleanup()


class TestSmoothieDriver:
    @pytest.mark.asyncio
    async def test_connection_lifecycle(
        self, driver: SmoothieDriver, smoothie_server
    ):
        """Test the connect and cleanup flow."""
        assert driver._connection_task is None
        assert driver.telnet is not None
        assert not driver.telnet.is_connected
        assert len(smoothie_server._tasks) == 0

        await driver.connect()
        await asyncio.sleep(0.1)  # Allow time for connection

        assert driver._connection_task is not None
        assert driver.telnet.is_connected
        assert len(smoothie_server._tasks) >= 1

        await driver.cleanup()
        await asyncio.sleep(0.1)  # Allow time for disconnect
        assert driver._connection_task.done()

    @pytest.mark.asyncio
    async def test_status_polling_and_parsing(
        self, driver: SmoothieDriver, smoothie_server
    ):
        """
        Test that the driver periodically polls and correctly parses status.
        """
        state_tracker = SignalTracker(driver.state_changed)

        await driver.connect()

        # The driver polls periodically. Wait for state update.
        # Use a loop with timeout for robustness
        for _ in range(50):
            if driver.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.1)

        assert driver.state.status == DeviceStatus.IDLE
        assert driver.state.machine_pos[0] == 1.2
        assert len(state_tracker.calls) > 0

    @pytest.mark.asyncio
    async def test_send_and_wait(
        self, connected_driver: SmoothieDriver, smoothie_server
    ):
        """Test executing a simple command that waits for 'ok'."""
        driver = connected_driver
        await driver.home(Axis.X)
        await asyncio.sleep(0.1)
        assert b"G28 X0" in smoothie_server.received_data

    @pytest.mark.asyncio
    async def test_run_streams_gcode(
        self, connected_driver: SmoothieDriver, smoothie_server
    ):
        """Test the full G-code streaming process for a simple job."""
        driver = connected_driver
        doc = Doc()
        ops = Ops()
        ops.add(MoveToCommand((10, 10, 0)))
        ops.add(LineToCommand((20, 20, 0)))

        job_finished_mock = MagicMock()
        driver.job_finished.send = job_finished_mock
        callback_mock = MagicMock()

        await driver.run(ops, doc, callback_mock)

        # Check that the server received the correct G-code
        received_str = smoothie_server.received_data.decode()
        assert "G0 X10.000 Y10.000 Z0.000" in received_str
        assert "G1 X20.000 Y20.000 Z0.000" in received_str

        # Check that callbacks were fired
        job_finished_mock.assert_called_once_with(driver)
        assert callback_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_run_raw(
        self, connected_driver: SmoothieDriver, smoothie_server
    ):
        """Test running a raw multi-line G-code string."""
        driver = connected_driver
        gcode = "G0 X10\nM3 S0.5\nG0 Y10"

        await driver.run_raw(gcode)

        received_str = smoothie_server.received_data.decode()
        assert "G0 X10" in received_str
        assert "M3 S0.5" in received_str
        assert "G0 Y10" in received_str

    @pytest.mark.asyncio
    async def test_server_disconnects_unexpectedly(
        self, driver: SmoothieDriver, smoothie_server
    ):
        """
        Test that the driver handles an abrupt server-side disconnect.
        """
        status_tracker = SignalTracker(driver.connection_status_changed)

        await driver.connect()

        # Wait for IDLE state to confirm connection is fully established
        for _ in range(5000):
            if driver.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.1)
        assert driver.state.status == DeviceStatus.IDLE

        # Stop the server. This should be immediate now.
        await smoothie_server.stop()

        # Wait for the driver to detect the disconnect
        for _ in range(5000):
            if driver.state.status == DeviceStatus.UNKNOWN:
                break
            await asyncio.sleep(0.1)

        assert driver.state.status == DeviceStatus.UNKNOWN

        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert (
            TransportStatus.DISCONNECTED in statuses
            or TransportStatus.ERROR in statuses
        )
