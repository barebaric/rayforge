import asyncio
from unittest.mock import AsyncMock, PropertyMock

import pytest

from rayforge.machine.driver.grbl.grbl_serial_simple import (
    GrblSerialSimpleDriver,
)
from rayforge.machine.transport import SerialTransport


@pytest.fixture
def simple_driver(context_initializer, machine, mocker):
    """A GrblSerialSimpleDriver with a mocked, connected transport."""
    transport = mocker.create_autospec(SerialTransport, instance=True)
    transport.send = AsyncMock()
    transport.disconnect = AsyncMock()
    transport.received = mocker.MagicMock()
    mocker.patch.object(
        transport,
        "is_connected",
        new_callable=PropertyMock,
        return_value=True,
    )

    driver = GrblSerialSimpleDriver(context_initializer, machine)
    driver.did_setup = True
    driver._transport = transport
    yield driver


@pytest.mark.asyncio
async def test_cancel_wakes_inflight_ping_pong(simple_driver):
    """Cancel must not leave an in-flight command waiting for timeout."""
    driver = simple_driver

    ping_task = asyncio.create_task(driver._ping_pong("M5", timeout=30.0))
    await asyncio.sleep(0)
    assert driver._pending is not None

    await asyncio.wait_for(driver.cancel(), timeout=1.0)

    lines = await asyncio.wait_for(ping_task, timeout=1.0)
    assert lines == []
    assert driver._pending is None


@pytest.mark.asyncio
async def test_cancel_interrupts_streaming_job_promptly(simple_driver):
    """Cancelling mid-job stops streaming immediately, without errors."""
    driver = simple_driver
    driver._start_job()

    lines = ["M4 S20", "G1 X10", "G1 Y10", "M5"]
    stream_task = asyncio.create_task(driver._stream_gcode_ping_pong(lines))
    await asyncio.sleep(0)

    await asyncio.wait_for(driver.cancel(), timeout=1.0)
    await asyncio.wait_for(stream_task, timeout=1.0)

    assert driver._job_running is False
