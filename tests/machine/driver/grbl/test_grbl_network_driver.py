import asyncio

import pytest
import pytest_asyncio
from raydriver.esp3d.emulator import Esp3dEmulator

from rayforge.machine.driver.grbl.grbl_network import GrblNetworkDriver
from rayforge.machine.models.machine import Machine


async def wait_for_connected(driver, timeout=5.0):
    connected = asyncio.Event()

    def _on_status(sender, status=None, message=None, **kwargs):
        if status is not None and status.name == "CONNECTED":
            connected.set()

    driver.connection_status_changed.connect(_on_status)
    try:
        await driver.connect()
        await asyncio.wait_for(connected.wait(), timeout=timeout)
    finally:
        driver.connection_status_changed.disconnect(_on_status)


@pytest_asyncio.fixture
async def esp3d_device():
    emulator = Esp3dEmulator()
    await emulator.start()
    try:
        yield emulator
    finally:
        await emulator.stop()


@pytest_asyncio.fixture
async def connected_driver(lite_context, esp3d_device):
    machine = Machine(lite_context)
    driver = GrblNetworkDriver(lite_context, machine)
    driver.setup(
        host="127.0.0.1",
        port=esp3d_device.http_port,
        ws_port=esp3d_device.ws_port,
        protocol="ESP3D",
    )
    await wait_for_connected(driver)
    try:
        yield driver, esp3d_device
    finally:
        await driver.cleanup()


class TestRegistry:
    def test_driver_registered(self):
        from rayforge.machine.driver import get_driver_cls

        assert get_driver_cls("GrblNetworkDriver") is GrblNetworkDriver

    def test_metadata(self):
        assert GrblNetworkDriver.supports_settings
        assert GrblNetworkDriver.supports_probing
        assert GrblNetworkDriver.supports_unit_detection
        assert not GrblNetworkDriver.reports_granular_progress

    def test_setup_vars(self):
        keys = set(GrblNetworkDriver.get_setup_vars().keys())
        assert {"host", "port", "ws_port", "protocol"} <= keys


class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_reports_resource_uri(self, connected_driver):
        driver, emulator = connected_driver
        assert driver.resource_uri == (f"tcp://127.0.0.1:{emulator.http_port}")

    def test_precheck_rejects_invalid_host(self):
        from rayforge.machine.driver.driver import DriverPrecheckError

        with pytest.raises(DriverPrecheckError):
            GrblNetworkDriver.precheck(host="bad host!")


class TestCommands:
    @pytest.mark.asyncio
    async def test_read_settings(self, connected_driver):
        driver, _ = connected_driver
        varsets = await driver._session.read_settings()
        assert ("13", "0") in varsets

    @pytest.mark.asyncio
    async def test_read_parser_state(self, connected_driver):
        driver, _ = connected_driver
        assert await driver.read_parser_state() == "G54"

    @pytest.mark.asyncio
    async def test_detect_unit_system(self, connected_driver):
        driver, _ = connected_driver
        assert (await driver.detect_unit_system()).name == "METRIC"

    @pytest.mark.asyncio
    async def test_run_raw_uploads_and_starts(self, connected_driver):
        driver, emulator = connected_driver
        await asyncio.wait_for(driver.run_raw("M5\n"), 5.0)
        assert emulator.uploaded == b"M5\n"
        assert emulator.started_files == ["rayforge.gcode"]
