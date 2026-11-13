"""Tests for the generic remote-machine driver.

The driver connects to a device on a raydriver daemon without naming
a protocol; the daemon resolves it and exposes capabilities as info.
The protocol-specific shells stay local-only. Driven against a real
daemon + GRBL firmware emulator over loopback.
"""

import asyncio

import pytest
import pytest_asyncio

raydriver_remote = pytest.importorskip("raydriver.remote.daemon")
raydriver_client = pytest.importorskip("raydriver.remote.client")

from raydriver.core.transport import MockTransport  # noqa: E402
from raydriver.grbl import GrblSession  # noqa: E402
from raydriver.grbl.emulator import GrblEmulator  # noqa: E402

from rayforge.machine.driver.grbl import GrblSerialDriver  # noqa: E402
from rayforge.machine.driver.remote_driver import (  # noqa: E402
    RaydriverRemoteDriver,
)

RemoteSession = raydriver_client.RemoteSession

FAST_CONFIG = {
    "handshake_timeout": 2.0,
    "handshake_poll_interval": 0.02,
    "status_poll_interval": 0.05,
    "reconnect_delay": 0.2,
    "command_timeout": 2.0,
    "poll_response_interval": 0.01,
    "safety_shutdown_delay": 0.02,
    "stall_timeout_default": 3.0,
    "stall_timeout_min": 1.0,
    "stall_timeout_max": 5.0,
}

DEVICE = "/dev/ttyUSB0"


@pytest_asyncio.fixture
async def daemon_port():
    """A raydriver daemon hosting one configured grbl device."""
    mock = MockTransport()
    emulator = GrblEmulator(mock, speed_factor=10)
    device_task = asyncio.ensure_future(emulator.run())

    def factory(protocol_name, config, dialect, event_callback):
        return GrblSession.with_transport(
            mock,
            config={**FAST_CONFIG, **(config or {})},
            dialect=dialect,
            event_callback=event_callback,
        )

    daemon = raydriver_remote.Daemon(
        raydriver_remote.DaemonConfig(
            port=0,
            keepalive_interval=0.5,
            sessions=[{"protocol": "grbl", "config": {"port": DEVICE}}],
        ),
        session_factory=factory,
    )
    server = await daemon.start()
    port = server.sockets[0].getsockname()[1]
    try:
        yield port
    finally:
        await daemon.stop()
        device_task.cancel()
        try:
            await device_task
        except asyncio.CancelledError:
            pass


def _remote_kwargs(port):
    return {
        "remote_host": "127.0.0.1",
        "remote_port": port,
        "device": DEVICE,
        "_raydriver_protocol": "grbl",
    }


def test_setup_vars_are_endpoint_plus_device():
    keys = {var.key for var in RaydriverRemoteDriver.get_setup_vars().vars}
    assert {
        "remote_host",
        "remote_port",
        "token",
        "device",
        "_raydriver_protocol",
    } <= keys
    # No local/remote toggle: a remote machine is always remote.
    assert "remote" not in keys
    assert "port" not in keys


def test_capabilities_come_from_metadata():
    assert RaydriverRemoteDriver.supports_probing_for(
        {"_raydriver_protocol": "grbl"}
    )
    assert not RaydriverRemoteDriver.supports_probing_for(
        {"_raydriver_protocol": "smoothie"}
    )
    assert not RaydriverRemoteDriver.supports_probing_for({})


def test_local_shells_have_no_remote_vars():
    for cls in (GrblSerialDriver,):
        keys = {var.key for var in cls.get_setup_vars().vars}
        assert "remote_host" not in keys
        assert "remote" not in keys


def test_local_setup_routes_to_rust_session(context_initializer, machine):
    drv = GrblSerialDriver(context_initializer, machine)
    drv.setup(port=DEVICE, baudrate=115200)
    assert isinstance(drv._session, GrblSession)


def test_remote_setup_routes_to_proxy(
    context_initializer, machine, daemon_port
):
    drv = RaydriverRemoteDriver(context_initializer, machine)
    drv.setup(**_remote_kwargs(daemon_port))
    assert isinstance(drv._session, RemoteSession)


@pytest.mark.asyncio
async def test_remote_connect_and_command(
    context_initializer, machine, daemon_port
):
    drv = RaydriverRemoteDriver(context_initializer, machine)
    drv.setup(**_remote_kwargs(daemon_port))
    connected = asyncio.Event()

    def on_status(sender, status=None, **kwargs):
        if status is not None and status.name == "CONNECTED":
            connected.set()

    drv.connection_status_changed.connect(on_status)
    try:
        await drv.connect()
        await asyncio.wait_for(connected.wait(), 10.0)
        lines = await drv.execute_interactive_command("$I")
        assert any("VER:" in line for line in lines)
        # The protocol was resolved by the daemon, not named by us.
        assert drv._session.protocol == "grbl"
    finally:
        drv.connection_status_changed.disconnect(on_status)
        await drv.cleanup()


@pytest.mark.asyncio
async def test_probe_over_remote(context_initializer, daemon_port):
    profile, _warnings = await RaydriverRemoteDriver.probe(
        context_initializer, **_remote_kwargs(daemon_port)
    )
    assert profile.machine_config.driver == "RaydriverRemoteDriver"
    assert profile.machine_config.driver_args is not None
    assert profile.machine_config.driver_args["device"] == DEVICE
