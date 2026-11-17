import asyncio

import pytest

from rayforge.machine.driver.grbl.grbl_network import GrblNetworkDriver
from rayforge.machine.models.machine import Machine


@pytest.mark.usefixtures("lite_context")
@pytest.mark.asyncio
async def test_connect_does_not_start_a_second_connection_loop(
    lite_context, monkeypatch
):
    machine = Machine(lite_context)
    driver = GrblNetworkDriver(lite_context, machine)
    driver.setup(
        host="192.0.2.1",
        port=80,
        ws_port=81,
        protocol="ESP3D",
    )
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def connection_loop():
        nonlocal call_count
        call_count += 1
        started.set()
        await release.wait()

    monkeypatch.setattr(driver, "_connection_loop", connection_loop)

    await driver.connect()
    await started.wait()
    first_task = driver._connection_task
    assert first_task is not None

    await driver.connect()

    assert driver._connection_task is first_task
    assert call_count == 1

    release.set()
    await first_task
    await driver.cleanup()
