"""Tests for the Rust-backed XtoolNetworkDriver, driven against the
xTool WiFi firmware emulator (raygeo.driver.xtool_d1.emulator)."""

import asyncio
import time

import pytest
import pytest_asyncio
import raygeo.driver.xtool_d1
from raygeo.driver.xtool_d1.emulator import XtoolEmulator
from raygeo.ops import Ops

from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import DriverMaturity
from rayforge.machine.driver.xtool import (
    DEFAULT_WS_PORT,
    XtoolNetworkDriver,
)
from rayforge.pipeline.artifact.encoded import EncodedOutput, MachineCodeOpMap

FAST_CONFIG = {
    "host": "127.0.0.1",
    "request_timeout": 2.0,
    "connect_timeout": 2.0,
    "reconnect_delay": 0.2,
    "homing_timeout": 5.0,
    "job_start_timeout": 5.0,
    "poll_interval": 0.05,
}


async def wait_until(predicate, timeout: float = 5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("condition not met")


class SignalRecorder:
    """Collects blinker signal emissions from a driver.

    Handlers are held by strong references: blinker drops weak
    receiver references as soon as they are garbage-collected.
    """

    def __init__(self, driver: XtoolNetworkDriver):
        self.events: list[tuple[str, dict]] = []
        self._keepalive: list = []
        driver.state_changed.connect(self._keep(self._send_state))
        driver.connection_status_changed.connect(
            self._keep(self._send_connection_status)
        )
        driver.job_finished.connect(self._keep(self._send_job_finished))
        driver.command_status_changed.connect(
            self._keep(self._send_command_status)
        )

    def _keep(self, fn):
        self._keepalive.append(fn)
        return fn

    def _send_state(self, sender, state=None, **kwargs):
        self.events.append(("state_changed", {"state": state}))

    def _send_connection_status(self, sender, status=None, message=None):
        self.events.append(
            (
                "connection_status_changed",
                {"status": status, "message": message},
            )
        )

    def _send_job_finished(self, sender):
        self.events.append(("job_finished", {}))

    def _send_command_status(self, sender, status=None, message=None):
        self.events.append(("command_status_changed", {"status": status}))

    def names(self):
        return [name for name, _ in self.events]

    async def wait_for(self, name: str, pred=None, timeout: float = 5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
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


def make_driver(context, machine, emulator) -> XtoolNetworkDriver:
    """Build an XtoolNetworkDriver around a session pointed at the
    emulator's ports, mirroring what ``_setup_implementation`` does
    with user-supplied host settings."""
    drv = XtoolNetworkDriver(context, machine)
    drv.did_setup = True
    config = {
        **FAST_CONFIG,
        "http_port": emulator.http_port,
        "ws_port": emulator.ws_port,
    }
    session = raygeo.driver.xtool_d1.XtoolNetworkDriver(
        config=config,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    drv._session = session
    return drv


@pytest_asyncio.fixture
async def connected_driver(context_initializer, machine):
    """A driver connected to a responsive xTool emulator.

    Yields (driver, emulator, recorder).
    """
    emulator = XtoolEmulator()
    await emulator.start()
    drv = make_driver(context_initializer, machine, emulator)
    recorder = SignalRecorder(drv)
    try:
        await drv.connect()
        await recorder.wait_connected()
        recorder.events.clear()
        yield drv, emulator, recorder
    finally:
        await drv.cleanup()
        await emulator.stop()


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("XtoolNetworkDriver") is XtoolNetworkDriver

    def test_is_experimental(self):
        assert XtoolNetworkDriver.maturity == DriverMaturity.EXPERIMENTAL

    def test_metadata(self):
        assert XtoolNetworkDriver.DISCOVERY is None
        assert XtoolNetworkDriver.supports_settings is False

    def test_setup_vars(self):
        varset = XtoolNetworkDriver.get_setup_vars()
        keys = set(varset.keys())
        assert {"host", "http_port", "ws_port"} <= keys

    def test_precheck_rejects_invalid_host(self):
        with pytest.raises(Exception, match="[Hh]ostname"):
            XtoolNetworkDriver.precheck(host="not a host")

    def test_setup_requires_host(self, context_initializer, machine):
        drv = XtoolNetworkDriver(context_initializer, machine)
        drv.setup()
        assert drv.state.error is not None
        assert "Hostname" in drv.state.error.title
        assert drv.resource_uri is None

    @pytest.mark.asyncio
    async def test_machine_space_wcs(self, context_initializer, machine):
        drv = XtoolNetworkDriver(context_initializer, machine)
        assert drv.machine_space_wcs == "G53"


@pytest.mark.asyncio
class TestConnection:
    async def test_connect_reports_connected(
        self, context_initializer, machine
    ):
        emulator = XtoolEmulator()
        await emulator.start()
        drv = make_driver(context_initializer, machine, emulator)
        recorder = SignalRecorder(drv)
        try:
            await drv.connect()
            await recorder.wait_connected()
            assert drv.resource_uri == (
                f"tcp://127.0.0.1:{emulator.http_port}"
            )
        finally:
            await drv.cleanup()
            await emulator.stop()

    async def test_foreign_device_fails_with_actionable_error(
        self, context_initializer, machine
    ):
        emulator = XtoolEmulator(machine_type="Some Router Board")
        await emulator.start()
        drv = make_driver(context_initializer, machine, emulator)
        recorder = SignalRecorder(drv)
        try:
            await drv.connect()
            payload = await recorder.wait_for(
                "connection_status_changed",
                pred=lambda p: p["status"].name == "ERROR",
            )
            assert "did not identify" in (payload["message"] or "")
        finally:
            await drv.cleanup()
            await emulator.stop()


@pytest.mark.asyncio
class TestCommands:
    async def test_interactive_command_acknowledged(self, connected_driver):
        drv, emulator, recorder = connected_driver
        lines = await drv.execute_interactive_command("M5")
        assert lines == []
        assert emulator.commands == ["M5"]
        await recorder.wait_for(
            "command_status_changed",
            pred=lambda p: p["status"].name == "IDLE",
        )

    async def test_home_sends_homing_sequence(self, connected_driver):
        drv, emulator, _ = connected_driver
        await drv.home()
        assert emulator.commands == ["M17 S1", "M207 S0", "M28", "M18"]

    async def test_home_single_axis_is_rejected(self, connected_driver):
        from raygeo.ops.axis import Axis

        drv, _, _ = connected_driver
        assert drv.can_home() is True
        assert drv.can_home(Axis.X) is False
        with pytest.raises(Exception, match="homing all axes"):
            await drv.home(Axis.X)

    async def test_jog_builds_manual_move_block(self, connected_driver):
        drv, emulator, _ = connected_driver
        await drv.jog(1000, x=10.0)
        assert len(emulator.blocks) == 1
        block = emulator.blocks[0]
        assert "G92 X0 Y0" in block
        assert "G1 X10 F1000 S0" in block

    async def test_move_to_builds_absolute_block(self, connected_driver):
        drv, emulator, _ = connected_driver
        await drv.move_to(10, 20, None, 600)
        assert len(emulator.blocks) == 1
        assert "G1 X10 Y20 F600 S0" in emulator.blocks[0]

    async def test_set_power_uses_dialect(self, connected_driver):
        drv, emulator, _ = connected_driver

        class Head:
            max_power = 1000

        await drv.set_power(Head(), 0.5)
        assert emulator.commands[-1] == "M4 S500"
        await drv.set_power(Head(), 0.0)
        assert emulator.commands[-1] == "M5"

    async def test_wcs_operations_unsupported(self, connected_driver):
        drv, _, _ = connected_driver
        assert drv.supported_wcs == []
        assert await drv.read_wcs_offsets() == {}
        assert await drv.read_parser_state() is None
        with pytest.raises(Exception, match="coordinate offsets"):
            await drv.set_wcs_offset("G54", 1, 2, None)


@pytest.mark.asyncio
class TestJobs:
    async def test_run_uploads_and_completes(self, connected_driver):
        drv, emulator, recorder = connected_driver
        gcode = "M17 S1\nG1 X10 F1000\nM5\n"
        encoded = EncodedOutput(text=gcode, op_map=MachineCodeOpMap())
        await drv.run(encoded, None, Ops())
        assert emulator.uploaded == gcode.encode()
        assert emulator.started_files == ["/tmp.gcode"]
        await recorder.wait_for("job_finished")
        await recorder.wait_for(
            "state_changed",
            pred=lambda p: p["state"].status.name == "IDLE",
        )

    async def test_cancel_stops_the_job(self, connected_driver):
        drv, emulator, recorder = connected_driver
        emulator.job_duration = 5.0
        encoded = EncodedOutput(text="M5\n", op_map=MachineCodeOpMap())
        task = asyncio.ensure_future(drv.run(encoded, None, Ops()))
        await recorder.wait_for(
            "state_changed",
            pred=lambda p: p["state"].status.name == "RUN",
        )
        await drv.cancel()
        await recorder.wait_for("job_finished")
        assert "stop" in emulator.job_actions
        await asyncio.wait_for(task, 5.0)

    async def test_hold_and_resume(self, connected_driver):
        drv, _, recorder = connected_driver
        encoded = EncodedOutput(text="M5\n", op_map=MachineCodeOpMap())
        task = asyncio.ensure_future(drv.run(encoded, None, Ops()))
        await recorder.wait_for(
            "state_changed",
            pred=lambda p: p["state"].status.name == "RUN",
        )
        await drv.set_hold(True)
        await recorder.wait_for(
            "state_changed",
            pred=lambda p: p["state"].status.name == "HOLD",
        )
        await drv.set_hold(False)
        await recorder.wait_for(
            "state_changed",
            pred=lambda p: p["state"].status.name == "RUN",
        )
        await asyncio.wait_for(task, 5.0)


@pytest.mark.asyncio
class TestProbe:
    async def test_probe_builds_profile_from_machine_type(
        self, context_initializer
    ):
        emulator = XtoolEmulator()
        await emulator.start()
        try:
            profile, warnings = await XtoolNetworkDriver.probe(
                context_initializer,
                host="127.0.0.1",
                http_port=emulator.http_port,
            )
            assert profile.meta.name == "xTool D1Pro"
            assert profile.machine_config.driver == "XtoolNetworkDriver"
            args = profile.machine_config.driver_args
            assert args is not None
            assert args["http_port"] == emulator.http_port
            assert args["ws_port"] == DEFAULT_WS_PORT
            assert warnings == []
        finally:
            await emulator.stop()

    async def test_probe_fails_without_a_device(self, context_initializer):
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            dead_port = sock.getsockname()[1]
        with pytest.raises(Exception, match="No xTool machine"):
            await XtoolNetworkDriver.probe(
                context_initializer,
                host="127.0.0.1",
                http_port=dead_port,
            )


@pytest.mark.asyncio
class TestEvents:
    async def test_safety_alarm_surfaces_in_state(self, connected_driver):
        drv, emulator, _ = connected_driver
        emulator.trigger_error("limitCheck")
        await wait_until(lambda: drv.state.status.name == "ALARM")
        assert drv.state.status.name == "ALARM"
        assert drv.state.error is not None
        assert drv.state.error.title == "Limit Switch Triggered"

    async def test_clear_alarm_resets_local_state(self, connected_driver):
        drv, emulator, _ = connected_driver
        emulator.trigger_error("tiltCheck")
        await wait_until(lambda: drv.state.status.name == "ALARM")
        await drv.clear_alarm()
        assert drv.state.error is None
