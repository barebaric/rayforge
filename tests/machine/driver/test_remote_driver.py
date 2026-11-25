"""Tests for the generic remote-machine driver.

The driver connects to a device on a raydriver daemon without naming
a protocol; the daemon resolves it and exposes capabilities as info.
The protocol-specific shells stay local-only. Driven against a real
daemon + GRBL firmware emulator over loopback.
"""

import asyncio
import time

import pytest
import pytest_asyncio

raydriver_remote = pytest.importorskip("raydriver.remote.daemon")
raydriver_client = pytest.importorskip("raydriver.remote.client")

import raydriver.grbl  # noqa: E402
from raydriver.core.transport import MockTransport  # noqa: E402
from raydriver.grbl.emulator import GrblEmulator  # noqa: E402
from raygeo.ops import Ops  # noqa: E402

from rayforge.core.doc import Doc  # noqa: E402
from rayforge.machine.driver.grbl import GrblSerialDriver  # noqa: E402
from rayforge.machine.driver.remote_driver import (  # noqa: E402
    RaydriverRemoteDriver,
)
from rayforge.pipeline.encode_inputs import encode_gcode  # noqa: E402

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


def _slow_job(lines: int = 80) -> str:
    return "".join(f"G0 X{i}\n" for i in range(lines))


async def _wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


async def _connect_and_wait(driver, timeout: float = 10.0) -> None:
    connected = asyncio.Event()

    def on_status(sender, status=None, **kwargs):
        if status is not None and status.name == "CONNECTED":
            connected.set()

    driver.connection_status_changed.connect(on_status)
    try:
        await driver.connect()
        await asyncio.wait_for(connected.wait(), timeout)
    finally:
        driver.connection_status_changed.disconnect(on_status)


@pytest_asyncio.fixture
async def daemon_port():
    """A raydriver daemon hosting one configured grbl device."""
    mock = MockTransport()
    emulator = GrblEmulator(mock, speed_factor=10)
    device_task = asyncio.ensure_future(emulator.run())

    def factory(protocol_name, config, dialect, event_callback):
        return raydriver.grbl.GrblSerialDriver.with_transport(
            mock,
            config={**FAST_CONFIG, **(config or {})},
            dialect=dialect,
            event_callback=event_callback,
        )

    daemon = raydriver_remote.Daemon(
        raydriver_remote.DaemonConfig(
            port=0,
            keepalive_interval=0.5,
            sessions=[
                {"protocol": "GrblSerialDriver", "config": {"port": DEVICE}}
            ],
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
        "_raydriver_protocol": "GrblSerialDriver",
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


def test_tls_vars_hidden_for_loopback():
    varset = RaydriverRemoteDriver.get_setup_vars()
    for key in ("tls_ca", "tls_hostname"):
        var = next(v for v in varset.vars if v.key == key)
        assert var.visible_when is not None
        assert var.visible_when({"remote_host": "127.0.0.1"}) is False
        assert var.visible_when({"remote_host": "localhost"}) is False
        assert var.visible_when({"remote_host": ""}) is False
        assert var.visible_when({"remote_host": "pi.local"}) is True


def test_capabilities_come_from_metadata():
    assert RaydriverRemoteDriver.supports_probing_for(
        {"_raydriver_protocol": "GrblSerialDriver"}
    )
    assert not RaydriverRemoteDriver.supports_probing_for(
        {"_raydriver_protocol": "XtoolM2Driver"}
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
    assert isinstance(drv._session, raydriver.grbl.GrblSerialDriver)


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
        assert drv._session.protocol == "GrblSerialDriver"
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


@pytest.mark.asyncio
async def test_remote_run_progress_and_finish(
    context_initializer, machine, daemon_port
):
    drv = RaydriverRemoteDriver(context_initializer, machine)
    drv.setup(**_remote_kwargs(daemon_port))
    job_finished = asyncio.Event()

    def on_job_finished(sender, **kwargs):
        job_finished.set()

    drv.job_finished.connect(on_job_finished)
    try:
        await _connect_and_wait(drv)

        ops = Ops()
        ops.set_power(1.0)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 0, 0)
        doc = Doc()
        encoded = encode_gcode(ops, drv._machine, doc)
        progress: list[int] = []

        await asyncio.wait_for(
            drv.run(encoded, doc, ops, progress.append), timeout=20.0
        )
        await asyncio.wait_for(job_finished.wait(), timeout=10.0)

        assert progress
    finally:
        drv.job_finished.disconnect(on_job_finished)
        await drv.cleanup()


@pytest.mark.asyncio
async def test_remote_cancel_stops_job(
    context_initializer, machine, daemon_port
):
    drv = RaydriverRemoteDriver(context_initializer, machine)
    drv.setup(**_remote_kwargs(daemon_port))
    try:
        await _connect_and_wait(drv)

        op_map = {i: i for i in range(300)}
        job = asyncio.ensure_future(
            drv._session.run(_slow_job(300), op_line_map=op_map)
        )
        await _wait_until(lambda: drv._session.job_running)

        await drv.cancel()
        await asyncio.wait_for(job, timeout=10.0)

        assert not drv._session.job_running
    finally:
        await drv.cleanup()


@pytest.mark.asyncio
async def test_remote_disconnect_sleeping_and_attach(
    context_initializer, machine, daemon_port
):
    drv = RaydriverRemoteDriver(context_initializer, machine)
    drv.setup(**_remote_kwargs(daemon_port))
    sleeping = asyncio.Event()
    job_finished = asyncio.Event()

    def on_status(sender, status=None, **kwargs):
        if status is not None and status.name == "SLEEPING":
            sleeping.set()

    def on_job_finished(sender, **kwargs):
        job_finished.set()

    drv.connection_status_changed.connect(on_status)
    drv.job_finished.connect(on_job_finished)
    try:
        await _connect_and_wait(drv)

        op_map = {i: i for i in range(500)}
        job = asyncio.ensure_future(
            drv._session.run(_slow_job(500), op_line_map=op_map)
        )
        await _wait_until(lambda: drv._session.job_running)

        drv._session.drop_connection()
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(job, timeout=10.0)
        await asyncio.wait_for(sleeping.wait(), timeout=5.0)

        await drv._session.attach()
        await asyncio.wait_for(job_finished.wait(), timeout=20.0)

        assert not drv._session.job_running
    finally:
        drv.connection_status_changed.disconnect(on_status)
        drv.job_finished.disconnect(on_job_finished)
        await drv.cleanup()
