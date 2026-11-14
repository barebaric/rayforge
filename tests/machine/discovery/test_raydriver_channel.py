"""Tests for the raydriver daemon discovery channel.

Every discovered device uses the single generic remote driver; the
protocol is carried as info, never used to pick a driver class.
Drives the channel against real in-test daemons over loopback, with
the mDNS browse stubbed to resolve to them.
"""

import pytest
import pytest_asyncio

raydriver_remote = pytest.importorskip("raydriver.remote.daemon")

from raydriver.remote.daemon import Daemon, DaemonConfig  # noqa: E402

from rayforge.machine.discovery import raydriver_channel  # noqa: E402
from rayforge.machine.discovery.types import device_key  # noqa: E402
from rayforge.machine.transport.mdns_scan import MDNSService  # noqa: E402

REMOTE_DRIVER = "RaydriverRemoteDriver"


def service_for(port: int) -> MDNSService:
    return MDNSService(
        service_type="_raydriver._tcp",
        name="test daemon",
        host="127.0.0.1",
        port=port,
    )


@pytest_asyncio.fixture
async def open_daemon():
    """An unauthenticated daemon with one configured grbl device."""
    daemon = Daemon(
        DaemonConfig(
            port=0,
            sessions=[
                {
                    "protocol": "grbl",
                    "config": {
                        "port": "/dev/ttyUSB0",
                        "baudrate": 115200,
                    },
                }
            ],
        )
    )
    server = await daemon.start()
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        await daemon.stop()


@pytest_asyncio.fixture
async def token_daemon():
    daemon = Daemon(DaemonConfig(port=0, token="secret"))
    server = await daemon.start()
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_enumerates_devices(open_daemon, monkeypatch):
    async def fake_scan(service_types):
        return [service_for(open_daemon)]

    monkeypatch.setattr(raydriver_channel, "scan_mdns_services", fake_scan)
    devices = await raydriver_channel.find_raydriver_devices()
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == REMOTE_DRIVER
    assert device.params["remote_host"] == "127.0.0.1"
    assert device.params["remote_port"] == open_daemon
    assert device.params["device"] == "/dev/ttyUSB0"
    assert device.params["_raydriver_protocol"] == "grbl"
    assert "raydriver://" in device.key


@pytest.mark.asyncio
async def test_telnet_device_uses_generic_driver(monkeypatch):
    daemon = Daemon(
        DaemonConfig(
            port=0,
            sessions=[
                {
                    "protocol": "grbl",
                    "config": {"host": "device.local", "tcp_port": 23},
                }
            ],
        )
    )
    server = await daemon.start()
    port = server.sockets[0].getsockname()[1]

    async def fake_scan(service_types):
        return [service_for(port)]

    monkeypatch.setattr(raydriver_channel, "scan_mdns_services", fake_scan)
    try:
        devices = await raydriver_channel.find_raydriver_devices()
        assert len(devices) == 1
        assert devices[0].driver_name == REMOTE_DRIVER
        assert devices[0].params["device"] == "device.local"
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_token_daemon_yields_daemon_row(token_daemon, monkeypatch):
    async def fake_scan(service_types):
        return [service_for(token_daemon)]

    monkeypatch.setattr(raydriver_channel, "scan_mdns_services", fake_scan)
    devices = await raydriver_channel.find_raydriver_devices()
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == REMOTE_DRIVER
    assert device.params["remote_host"] == "127.0.0.1"
    assert device.params["remote_port"] == token_daemon
    assert "device" not in device.params
    assert "token required" in device.detail


@pytest.mark.asyncio
async def test_no_services_yields_nothing(monkeypatch):
    async def fake_scan(service_types):
        return []

    monkeypatch.setattr(raydriver_channel, "scan_mdns_services", fake_scan)
    assert await raydriver_channel.find_raydriver_devices() == []


@pytest.mark.asyncio
async def test_unreachable_daemon_yields_row(monkeypatch):
    async def fake_scan(service_types):
        return [service_for(1)]  # nothing listens here

    monkeypatch.setattr(raydriver_channel, "scan_mdns_services", fake_scan)
    devices = await raydriver_channel.find_raydriver_devices()
    assert len(devices) == 1
    assert "unreachable" in devices[0].detail


def test_device_key_remote_uniqueness():
    local = device_key("GrblSerialDriver", {"port": "/dev/ttyUSB0"})
    remote_a = device_key(
        REMOTE_DRIVER,
        {
            "remote_host": "pi-a.local",
            "remote_port": 8765,
            "device": "/dev/ttyUSB0",
        },
    )
    remote_b = device_key(
        REMOTE_DRIVER,
        {
            "remote_host": "pi-b.local",
            "remote_port": 8765,
            "device": "/dev/ttyUSB0",
        },
    )
    assert len({local, remote_a, remote_b}) == 3
    assert remote_a.startswith(f"{REMOTE_DRIVER}:raydriver://pi-a.local:")
