"""Tests for the Rust-backed SmoothieDriver."""

import asyncio

import pytest
import pytest_asyncio
from raydriver.core.transport import MockTransport
from raydriver.smoothie import SmoothieSession
from raydriver.smoothie.emulator import SmoothieEmulator
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.core.doc import Doc
from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DriverPrecheckError,
    DriverSetupError,
)
from rayforge.machine.driver.smoothie import SmoothieDriver
from rayforge.pipeline.encoder.gcode import GcodeEncoder

FAST_CONFIG = {
    "handshake_timeout": 2.0,
    "handshake_poll_interval": 0.02,
    "status_poll_interval": 0.05,
    "reconnect_delay": 0.2,
    "command_timeout": 2.0,
}


class SignalRecorder:
    """Collects blinker signal emissions from a driver.

    Handlers are held by strong references: blinker drops weak
    receiver references as soon as they are garbage-collected.
    """

    def __init__(self, driver):
        self.events = []
        driver.state_changed.connect(
            lambda sender, state: self.events.append(("state_changed", state)),
            weak=False,
        )
        driver.connection_status_changed.connect(
            lambda sender, status, message=None: self.events.append(
                ("connection_status_changed", {"status": status})
            ),
            weak=False,
        )
        driver.job_finished.connect(
            lambda sender: self.events.append(("job_finished", None)),
            weak=False,
        )

    def names(self):
        return [name for name, _ in self.events]

    async def wait_for(self, name: str, pred=None, timeout: float = 5.0):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            for event_name, payload in self.events:
                if event_name != name:
                    continue
                if pred is None or pred(payload):
                    return payload
            await asyncio.sleep(0.01)
        raise AssertionError(
            f"event {name!r} not observed; got {self.names()}"
        )

    async def wait_connected(self, timeout: float = 5.0):
        return await self.wait_for(
            "connection_status_changed",
            pred=lambda p: p["status"].name == "CONNECTED",
            timeout=timeout,
        )


def make_driver(context, machine) -> tuple[SmoothieDriver, MockTransport]:
    """Build a SmoothieDriver around an emulator-backed session,
    mirroring what ``_setup_implementation`` does for a real
    telnet host."""
    drv = SmoothieDriver(context, machine)
    drv.did_setup = True
    mock = MockTransport()
    session = SmoothieSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    drv._session = session
    return drv, mock


@pytest_asyncio.fixture
async def connected_driver(context_initializer, machine):
    """A driver connected to a responsive Smoothie emulator.

    Yields (driver, mock, emulator, recorder).
    """
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        yield drv, mock, emulator, recorder
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("SmoothieDriver") is SmoothieDriver

    def test_metadata(self, context_initializer, machine):
        drv = SmoothieDriver(context_initializer, machine)
        assert drv.machine_space_wcs == "G53"
        assert drv.supports_settings is False
        assert SmoothieDriver.reports_granular_progress is True
        assert SmoothieDriver.supports_probing is True

    def test_setup_vars(self):
        varset = SmoothieDriver.get_setup_vars()
        keys = {v.key for v in varset.vars}
        assert {"host", "port"} <= keys

    def test_create_encoder(self, context_initializer, machine):
        drv = SmoothieDriver(context_initializer, machine)
        encoder = drv.get_encoder()
        assert isinstance(encoder, GcodeEncoder)

    def test_precheck_rejects_invalid_host(self):
        with pytest.raises(DriverPrecheckError):
            SmoothieDriver.precheck(host="not a hostname!")

    def test_setup_missing_host_raises(self, context_initializer, machine):
        drv = SmoothieDriver(context_initializer, machine)
        with pytest.raises(DriverSetupError):
            drv._setup_implementation(host="", port=23)


@pytest.mark.asyncio
async def test_connects_and_reports_state(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        assert drv.state.status == DeviceStatus.IDLE
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_probe_queries(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        version = await drv.execute_interactive_command("version")
        assert version[-1] == "ok"
        alpha_max = await drv.execute_interactive_command(
            "config-get alpha_max"
        )
        assert alpha_max[0] == "200.0000"
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_run_with_encoded_ops(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        ops = Ops()
        ops.move_to(10, 10, 0)
        ops.line_to(20, 20, 0)
        doc = Doc()
        encoded = drv.get_encoder().encode(ops, machine, doc)
        await asyncio.wait_for(drv.run(encoded, doc, ops), timeout=10.0)
        await recorder.wait_for("job_finished")
        sent = b"".join(mock.sent())
        assert b"G1" in sent
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_run_raw_streams_gcode(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        await asyncio.wait_for(drv.run_raw("G0 X10\nG1 X20 Y20"), timeout=10.0)
        await recorder.wait_for("job_finished")
        assert emulator.mpos[0] == pytest.approx(20.0, abs=0.1)
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_cancel_sends_ctrl_c_and_safety_off(
    context_initializer, machine
):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock, speed_factor=1)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        gcode = "\n".join([f"G1 X{i % 50} F600" for i in range(50)])
        run_task = asyncio.create_task(drv.run_raw(gcode))
        await asyncio.sleep(0.2)
        await drv.cancel()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        await recorder.wait_for("job_finished")
        sent = b"".join(mock.sent())
        assert b"\x03" in sent
        assert b"M5\n" in sent
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_hold_sends_realtime_bytes(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        mock.clear_sent()
        await drv.set_hold(True)
        assert b"!" in b"".join(mock.sent())
        await drv.set_hold(False)
        assert b"~" in b"".join(mock.sent())
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_home_uses_dialect(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        mock.clear_sent()
        await asyncio.wait_for(drv.home(), timeout=5.0)
        assert b"$H" in b"".join(mock.sent())
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_not_implemented_methods(context_initializer, machine):
    drv, mock = make_driver(context_initializer, machine)
    emulator = SmoothieEmulator(mock)
    emulator_task = asyncio.ensure_future(emulator.run())
    try:
        await drv.connect()
        with pytest.raises(NotImplementedError):
            await drv.read_settings()
        with pytest.raises(NotImplementedError):
            await drv.write_setting("k", "v")
        with pytest.raises(NotImplementedError):
            await drv.read_wcs_offsets()
        with pytest.raises(NotImplementedError):
            await drv.run_probe_cycle(Axis.X, 10, 100)
    finally:
        await drv.cleanup()
        emulator_task.cancel()
        try:
            await emulator_task
        except asyncio.CancelledError:
            pass
