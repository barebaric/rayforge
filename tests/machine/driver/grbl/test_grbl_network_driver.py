"""Tests for the GRBL network driver's SD job handling."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from rayforge.machine.driver.driver import DeviceState, DeviceStatus
from rayforge.machine.driver.grbl import GrblNetworkDriver
from rayforge.machine.driver.grbl import grbl_network as network_module
from rayforge.pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
)

POLL_S = 0.01
START_TIMEOUT_S = 0.2


@pytest.fixture
def driver(context_initializer, machine):
    drv = GrblNetworkDriver(context_initializer, machine)
    drv.host = "192.0.2.1"
    drv.keep_running = True
    yield drv


def set_status(drv: GrblNetworkDriver, status: DeviceStatus):
    state = DeviceState()
    state.status = status
    drv.state = state


class TestWaitForJobCompletion:
    @pytest.mark.asyncio
    async def test_returns_after_run_ends(self, driver, monkeypatch):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        monkeypatch.setattr(
            network_module, "JOB_START_TIMEOUT_S", START_TIMEOUT_S
        )
        set_status(driver, DeviceStatus.IDLE)

        async def simulate_run():
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.RUN)
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.IDLE)

        simulation = asyncio.create_task(simulate_run())
        await asyncio.wait_for(
            driver._wait_for_job_completion(),
            timeout=10 * POLL_S + 1.0,
        )
        await simulation
        assert driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_waits_through_hold(self, driver, monkeypatch):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        monkeypatch.setattr(
            network_module, "JOB_START_TIMEOUT_S", START_TIMEOUT_S
        )
        set_status(driver, DeviceStatus.RUN)

        async def simulate_hold_then_idle():
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.HOLD)
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.IDLE)

        simulation = asyncio.create_task(simulate_hold_then_idle())
        await asyncio.wait_for(
            driver._wait_for_job_completion(),
            timeout=10 * POLL_S + 1.0,
        )
        await simulation

    @pytest.mark.asyncio
    async def test_returns_when_job_never_starts(self, driver, monkeypatch):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        monkeypatch.setattr(
            network_module, "JOB_START_TIMEOUT_S", START_TIMEOUT_S
        )
        set_status(driver, DeviceStatus.IDLE)
        await asyncio.wait_for(
            driver._wait_for_job_completion(),
            timeout=10 * START_TIMEOUT_S,
        )

    @pytest.mark.asyncio
    async def test_returns_when_cancelled(self, driver, monkeypatch):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        set_status(driver, DeviceStatus.RUN)

        async def cancel_soon():
            await asyncio.sleep(4 * POLL_S)
            driver._is_cancelled = True

        simulation = asyncio.create_task(cancel_soon())
        await asyncio.wait_for(
            driver._wait_for_job_completion(),
            timeout=10 * POLL_S + 1.0,
        )
        await simulation


class TestRunWaitsForMachine:
    @pytest.mark.asyncio
    async def test_job_finished_after_machine_idle(self, driver, monkeypatch):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        monkeypatch.setattr(
            network_module, "JOB_START_TIMEOUT_S", START_TIMEOUT_S
        )
        driver._upload = AsyncMock()
        driver._execute = AsyncMock()
        set_status(driver, DeviceStatus.IDLE)

        finished = asyncio.Event()
        driver.job_finished.connect(
            lambda *a, **kw: finished.set(), weak=False
        )

        async def simulate_machine():
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.RUN)
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.IDLE)

        simulation = asyncio.create_task(simulate_machine())
        encoded = EncodedOutput(text="G1 X1 F100", op_map=MachineCodeOpMap())
        await asyncio.wait_for(
            driver.run(encoded, None, None),
            timeout=10 * POLL_S + 1.0,
        )
        await simulation

        assert driver._execute.await_count == 1
        assert finished.is_set()

    @pytest.mark.asyncio
    async def test_job_finished_not_sent_when_cancelled(
        self, driver, monkeypatch
    ):
        monkeypatch.setattr(network_module, "JOB_COMPLETION_POLL_S", POLL_S)
        monkeypatch.setattr(
            network_module, "JOB_START_TIMEOUT_S", START_TIMEOUT_S
        )
        driver._upload = AsyncMock()
        driver._execute = AsyncMock()
        set_status(driver, DeviceStatus.IDLE)

        finished = asyncio.Event()
        driver.job_finished.connect(
            lambda *a, **kw: finished.set(), weak=False
        )

        async def cancel_mid_run():
            await asyncio.sleep(4 * POLL_S)
            set_status(driver, DeviceStatus.RUN)
            await asyncio.sleep(4 * POLL_S)
            driver._is_cancelled = True

        simulation = asyncio.create_task(cancel_mid_run())
        encoded = EncodedOutput(text="G1 X1 F100", op_map=MachineCodeOpMap())
        await asyncio.wait_for(
            driver.run(encoded, None, None),
            timeout=10 * POLL_S + 1.0,
        )
        await simulation

        driver._execute.assert_awaited_once()
        assert not finished.is_set()
