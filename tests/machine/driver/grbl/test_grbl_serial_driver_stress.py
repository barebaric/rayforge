"""
Stress tests for GrblSerialDriver with fragmented serial delivery.

Exercises random serial packet boundaries at the driver level,
including interleaved status reports, errors, and acks delivered
in arbitrary-sized chunks.

Run with:  pixi run test -m "stress"
"""

import random
import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock, PropertyMock

from rayforge.machine.transport.grbl import GrblSerialTransport
from rayforge.machine.transport import SerialTransport, TransportStatus
from rayforge.machine.driver.grbl.grbl_serial import GrblSerialDriver
from rayforge.machine.driver.driver import DeviceConnectionError


pytestmark = pytest.mark.stress


def _fragment(data: bytes, rng: random.Random) -> list[bytes]:
    """Split *data* into random-sized chunks (1-max_chunk bytes)."""
    if not data:
        return []
    chunks = []
    i = 0
    while i < len(data):
        chunk_len = rng.randint(1, min(12, len(data) - i))
        chunks.append(data[i : i + chunk_len])
        i += chunk_len
    return chunks


STATUS_REPORTS = [
    b"<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n",
    b"<Run|MPos:1.234,5.678,0.000|FS:500,0>\r\n",
    b"<Jog|MPos:10.0,20.0,0.0|FS:1000,0>\r\n",
    b"<Home|MPos:0,0,0|FS:0,0>\r\n",
]


class TestDriverStreamingFuzz:
    """Stress the GrblSerialDriver with fragmented ack delivery."""

    @pytest.fixture
    def mock_serial_transport(self, mocker):
        mock = mocker.create_autospec(SerialTransport, instance=True)
        mock.connect = AsyncMock()
        mock.disconnect = AsyncMock()
        mock.send = AsyncMock()
        mock.received = MagicMock()
        mock.status_changed = MagicMock()
        mock.port = "/dev/ttyUSB0"
        return mock

    @pytest.fixture
    def driver(
        self, context_initializer, machine, mock_serial_transport, mocker
    ):
        mocker.patch(
            "rayforge.machine.driver.grbl.grbl_serial"
            ".SerialTransport.__init__",
            return_value=None,
        )
        driver_instance = GrblSerialDriver(context_initializer, machine)
        driver_instance.grbl_transport = GrblSerialTransport(
            mock_serial_transport
        )
        driver_instance.grbl_transport.received.connect(
            driver_instance.on_serial_data_received
        )
        driver_instance.grbl_transport.status_changed.connect(
            driver_instance.on_serial_status_changed
        )
        driver_instance.did_setup = True
        return driver_instance

    @pytest_asyncio.fixture
    async def connected_driver(self, driver, mock_serial_transport, mocker):
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
        welcome = b"Grbl 1.1h ['$' for help]\r\n"
        driver.on_serial_data_received(mock_serial_transport, welcome)
        await asyncio.sleep(0)

        status = b"<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n"
        driver.on_serial_data_received(mock_serial_transport, status)
        await asyncio.sleep(0)

        version_response = b"[VER:1.1h:]\r\nok\r\n"
        driver.on_serial_data_received(mock_serial_transport, version_response)
        await asyncio.sleep(0.01)
        mock_serial_transport.send.reset_mock()

        yield driver

        await driver.cleanup()
        if not connect_task.done():
            connect_task.cancel()
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_job_random_chunk_acks(
        self, connected_driver, mock_serial_transport
    ):
        driver = connected_driver
        rng = random.Random(111)
        driver.job_finished.send = MagicMock()

        gcode = "\n".join(f"G0X{i}" for i in range(1000))
        run_task = asyncio.create_task(driver.run_raw(gcode))

        for i in range(1000):
            await asyncio.sleep(0.005)
            ack = b"ok\r\n"
            if rng.random() < 0.25:
                ack = rng.choice(STATUS_REPORTS) + ack
            chunks = _fragment(ack, rng)
            for chunk in chunks:
                driver.on_serial_data_received(mock_serial_transport, chunk)

        await run_task
        assert driver.grbl_transport.buffer_count == 0
        assert driver.grbl_transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_20_repeated_small_jobs_fragmented_acks(
        self, connected_driver, mock_serial_transport
    ):
        driver = connected_driver
        rng = random.Random(222)

        for job in range(20):
            gcode = "\n".join(f"G0X{i}" for i in range(10))
            run_task = asyncio.create_task(driver.run_raw(gcode))

            for _ in range(10):
                await asyncio.sleep(0.002)
                ack = b"ok\r\n"
                if rng.random() < 0.3:
                    ack = rng.choice(STATUS_REPORTS) + ack
                chunks = _fragment(ack, rng)
                for chunk in chunks:
                    driver.on_serial_data_received(
                        mock_serial_transport, chunk
                    )

            await run_task
            assert driver.grbl_transport.buffer_count == 0, f"job {job}"

    @pytest.mark.asyncio
    async def test_single_byte_ack_delivery_through_driver(
        self, connected_driver, mock_serial_transport
    ):
        driver = connected_driver
        driver.job_finished.send = MagicMock()

        gcode = "\n".join(f"G0X{i}" for i in range(20))
        run_task = asyncio.create_task(driver.run_raw(gcode))

        for i in range(20):
            await asyncio.sleep(0.002)
            for byte in b"ok\r\n":
                driver.on_serial_data_received(
                    mock_serial_transport, bytes([byte])
                )

        await run_task
        assert driver.grbl_transport.buffer_count == 0
        assert driver.grbl_transport.pending_queue.empty()

    @pytest.mark.asyncio
    async def test_interleaved_error_mid_job_fragmented(
        self, connected_driver, mock_serial_transport
    ):
        driver = connected_driver
        driver.job_finished.send = MagicMock()

        gcode = "\n".join(f"G0X{i}" for i in range(10))
        run_task = asyncio.create_task(driver.run_raw(gcode))

        for _ in range(3):
            await asyncio.sleep(0.005)
            for chunk in _fragment(b"ok\r\n", random.Random()):
                driver.on_serial_data_received(mock_serial_transport, chunk)

        await asyncio.sleep(0.005)
        interleaved = b"<Alarm|MPos:0,0,0|FS:0,0>\r\nerror:20\r\n"
        for chunk in _fragment(interleaved, random.Random()):
            driver.on_serial_data_received(mock_serial_transport, chunk)

        try:
            await asyncio.wait_for(run_task, timeout=1.0)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            DeviceConnectionError,
        ):
            pass

        assert driver._job_running is False
        assert driver.grbl_transport.buffer_count == 0
        assert driver.grbl_transport.pending_queue.empty()
