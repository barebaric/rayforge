"""Remote-endpoint tests for the migrated driver shells.

Drives the shells against a real raydriver daemon + GRBL firmware
emulator over loopback: setup routing, connect, interactive
commands and probe-over-remote. Skipped when the installed
raydriver predates remote sessions.
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
from rayforge.machine.driver.grbl.grbl_serial_simple import (  # noqa: E402
    GrblSerialSimpleDriver,
)
from rayforge.machine.driver.grbl.grbl_telnet import (  # noqa: E402
    GrblTelnetDriver,
)
from rayforge.machine.driver.marlin.marlin_serial import (  # noqa: E402
    MarlinSerialDriver,
)
from rayforge.machine.driver.remote_setup import (  # noqa: E402
    remote_config,
)
from rayforge.machine.driver.smoothie.smoothie_driver import (  # noqa: E402
    SmoothieDriver,
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

ALL_SHELLS = [
    GrblSerialDriver,
    GrblTelnetDriver,
    GrblSerialSimpleDriver,
    SmoothieDriver,
    MarlinSerialDriver,
]


@pytest_asyncio.fixture
async def daemon_port():
    """A raydriver daemon hosting emulator-backed grbl sessions."""
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
        raydriver_remote.DaemonConfig(port=0, keepalive_interval=0.5),
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


def test_setup_vars_include_remote_group():
    for cls in ALL_SHELLS:
        keys = {var.key for var in cls.get_setup_vars().vars}
        assert {
            "remote",
            "remote_host",
            "remote_port",
            "token",
            "tls_ca",
            "tls_hostname",
        } <= keys, cls.__name__


def test_remote_vars_visibility_gating():
    for cls in ALL_SHELLS:
        varset = cls.get_setup_vars()
        host_var = next(var for var in varset.vars if var.key == "remote_host")
        assert host_var.visible_when is not None
        assert host_var.visible_when({"remote": False}) is False
        assert host_var.visible_when({"remote": True}) is True
        assert host_var.required_when is not None
        assert host_var.required_when({"remote": True}) is True


def test_remote_config_mapping():
    assert remote_config({"port": "/dev/ttyUSB0"}) == {}
    config = remote_config(
        {
            "remote": True,
            "remote_host": "pi.local",
            "remote_port": 9999,
            "token": "secret",
            "tls_ca": "",
        }
    )
    assert config == {
        "remote_host": "pi.local",
        "remote_port": 9999,
        "token": "secret",
    }


def test_local_setup_routes_to_rust_session(context_initializer, machine):
    drv = GrblSerialDriver(context_initializer, machine)
    drv.setup(port="/dev/ttyUSB0", baudrate=115200)
    assert isinstance(drv._session, GrblSession)


def test_remote_setup_routes_to_proxy(
    context_initializer, machine, daemon_port
):
    drv = GrblSerialDriver(context_initializer, machine)
    drv.setup(
        port="/dev/ttyUSB0",
        baudrate=115200,
        remote=True,
        remote_host="127.0.0.1",
        remote_port=daemon_port,
    )
    assert isinstance(drv._session, RemoteSession)


@pytest.mark.asyncio
async def test_remote_connect_and_command(
    context_initializer, machine, daemon_port
):
    drv = GrblSerialDriver(context_initializer, machine)
    drv.setup(
        port="/dev/ttyUSB0",
        baudrate=115200,
        remote=True,
        remote_host="127.0.0.1",
        remote_port=daemon_port,
    )
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
    finally:
        drv.connection_status_changed.disconnect(on_status)
        await drv.cleanup()


@pytest.mark.asyncio
async def test_probe_over_remote(context_initializer, daemon_port):
    profile, _warnings = await GrblSerialDriver.probe(
        context_initializer,
        port="/dev/ttyUSB0",
        baudrate=115200,
        remote=True,
        remote_host="127.0.0.1",
        remote_port=daemon_port,
    )
    assert profile.machine_config.driver == "GrblSerialDriver"


def test_other_shells_route_remote(context_initializer, machine, daemon_port):
    cases = [
        (GrblTelnetDriver, {"host": "device.local", "port": 23}),
        (
            GrblSerialSimpleDriver,
            {"port": "/dev/ttyUSB0", "baudrate": 115200},
        ),
        (SmoothieDriver, {"host": "device.local", "port": 23}),
        (
            MarlinSerialDriver,
            {"port": "/dev/ttyUSB0", "baudrate": 115200},
        ),
    ]
    for cls, kwargs in cases:
        drv = cls(context_initializer, machine)
        drv.setup(
            **kwargs,
            remote=True,
            remote_host="127.0.0.1",
            remote_port=daemon_port,
        )
        assert isinstance(drv._session, RemoteSession), cls.__name__
