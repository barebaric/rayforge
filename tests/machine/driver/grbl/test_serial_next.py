"""Tests for the Rust-backed GrblSerialNextDriver shell."""

import asyncio
import time

import pytest
import pytest_asyncio
from raydriver.emulator import GrblEmulator
from raydriver.grbl import GrblSession, MockTransport
from raygeo.ops import Ops
from raygeo.ops.axis import Axis

from rayforge.core.doc import Doc
from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DriverMaturity,
)
from rayforge.machine.driver.grbl import GrblSerialNextDriver
from rayforge.pipeline.encoder.gcode import GcodeEncoder

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


def _parser_module():
    """The raydriver.grbl.parser module, resolved through the runtime
    package so pyright sees the Rust stubs."""
    import raydriver.grbl.parser

    return raydriver.grbl.parser


async def wait_until(predicate, timeout: float = 5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    raise AssertionError("condition not met")


class SignalRecorder:
    """Collects blinker signal emissions from a driver.

    Handlers are held by strong references: blinker drops weak
    receiver references as soon as they are garbage-collected.
    """

    def __init__(self, driver: GrblSerialNextDriver):
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
        driver.probe_status_changed.connect(
            self._keep(self._send_probe_status)
        )
        driver.wcs_updated.connect(self._keep(self._send_wcs_updated))

    def _keep(self, fn):
        self._keepalive.append(fn)
        return fn

    def _send_state(self, sender, state=None, **kwargs):
        self.events.append(("state_changed", {"state": state}))

    def _send_wcs_updated(self, sender, offsets=None):
        self.events.append(("wcs_updated", {"offsets": offsets}))

    def _send_connection_status(self, sender, status=None, message=None):
        self.events.append(("connection_status_changed", {"status": status}))

    def _send_job_finished(self, sender):
        self.events.append(("job_finished", {}))

    def _send_command_status(self, sender, status=None, message=None):
        self.events.append(("command_status_changed", {"status": status}))

    def _send_probe_status(self, sender, message=None):
        self.events.append(("probe_status_changed", {"message": message}))

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


def sent_text(mock) -> bytes:
    return b"".join(mock.sent())


def make_driver(
    context, machine
) -> tuple[GrblSerialNextDriver, MockTransport]:
    """Build a GrblSerialNextDriver around an emulator-backed
    session, mirroring what ``_setup_implementation`` does for a
    real serial port."""
    drv = GrblSerialNextDriver(context, machine)
    drv.did_setup = True
    mock = MockTransport()
    session = GrblSession.with_transport(
        mock,
        config=FAST_CONFIG,
        dialect=drv._dialect_templates(),
        event_callback=drv._on_session_event,
    )
    drv._session = session
    return drv, mock


@pytest_asyncio.fixture
async def connected_driver(context_initializer, machine):
    """A driver connected to a responsive Grbl emulator.

    Yields (driver, mock, emulator, recorder).
    """
    drv, mock = make_driver(context_initializer, machine)
    emulator = GrblEmulator(mock)
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
        assert get_driver_cls("GrblSerialNextDriver") is GrblSerialNextDriver

    def test_metadata(self):
        assert GrblSerialNextDriver.machine_space_wcs
        assert GrblSerialNextDriver.supports_settings
        assert GrblSerialNextDriver.maturity == DriverMaturity.EXPERIMENTAL

    def test_setup_vars(self):
        varset = GrblSerialNextDriver.get_setup_vars()
        keys = set(varset.keys())
        assert {"port", "baudrate", "deadlock_detection"} <= keys

    def test_create_encoder(self, context_initializer, machine):
        drv = GrblSerialNextDriver(context_initializer, machine)
        encoder = drv.get_encoder()
        assert isinstance(encoder, GcodeEncoder)
        assert encoder.dialect.uid == machine.dialect.uid

    def test_convert_state(self):
        from raydriver.grbl.types import DeviceState as RDState

        state = _parser_module().parse_state(
            "<Alarm:1|FS:500,0>", RDState(), False
        )
        converted = GrblSerialNextDriver._convert_state(state)
        assert converted.status == DeviceStatus.ALARM
        assert converted.feed_rate == 500
        assert converted.error is not None
        assert converted.error.code == 1

    def test_setup_with_cached_rx_buffer(self, context_initializer, machine):
        drv = GrblSerialNextDriver(context_initializer, machine)
        drv.config["rx_buffer_size"] = 511
        drv.setup(port="/dev/ttyUSB0", baudrate=115200)
        assert drv.did_setup
        assert drv.state.error is None
        assert drv._session is not None

    def test_setup_without_cached_rx_buffer(
        self, context_initializer, machine
    ):
        drv = GrblSerialNextDriver(context_initializer, machine)
        drv.setup(port="/dev/ttyUSB0", baudrate=115200)
        assert drv.did_setup
        assert drv.state.error is None
        assert drv._session is not None

    @pytest.mark.asyncio
    async def test_commands_without_session_raise(
        self, context_initializer, machine
    ):
        from rayforge.machine.driver.driver import DeviceConnectionError

        drv = GrblSerialNextDriver(context_initializer, machine)
        with pytest.raises(DeviceConnectionError):
            await drv.home(None)


@pytest.mark.asyncio
class TestConnection:
    async def test_connect_reports_connected(
        self, context_initializer, machine
    ):
        drv, mock = make_driver(context_initializer, machine)
        recorder = SignalRecorder(drv)
        emulator = GrblEmulator(mock)
        task = asyncio.ensure_future(emulator.run())
        try:
            await drv.connect()
            await recorder.wait_connected()
            assert b"$I\n" in sent_text(mock)
        finally:
            await drv.cleanup()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_resource_uri_none_when_disconnected(
        self, context_initializer, machine
    ):
        drv, _mock = make_driver(context_initializer, machine)
        assert drv.resource_uri is None


@pytest.mark.asyncio
class TestDeviceOperations:
    async def test_move_to_uses_dialect_template(self, connected_driver):
        drv, mock, _emulator, _recorder = connected_driver
        await drv.move_to(10, -20.5)
        expected = (
            drv.dialect.format_move_to(x=10.0, y=-20.5, speed=1500, z=None)
            + "\n"
        ).encode()
        assert expected in sent_text(mock)

    async def test_home_sequence(self, connected_driver):
        drv, mock, _emulator, _recorder = connected_driver
        await drv.home(None)
        text = sent_text(mock)
        assert b"$H\n" in text
        assert b"G4 P0.01\n" in text
        assert text.index(b"$H\n") < text.index(b"G55\n")
        assert text.index(b"G55\n") < text.index(b"G54\n")

    async def test_set_power(self, connected_driver):
        drv, mock, _emulator, _recorder = connected_driver

        class Head:
            max_power = 1000

        mock.clear_sent()
        await drv.set_power(Head(), 0.5)
        cmd = drv.dialect.laser_on.format(power=500)
        assert (cmd + "\n").encode() in sent_text(mock)
        await drv.set_power(Head(), 0)
        assert (drv.dialect.laser_off + "\n").encode() in sent_text(mock)

    async def test_jog(self, connected_driver):
        drv, mock, _emulator, _recorder = connected_driver
        mock.clear_sent()
        await drv.jog(1000, x=10.0)
        sent = sent_text(mock)
        template = drv.dialect.jog.format(speed=1000)
        assert (template + " X10.0\n").encode() in sent

    async def test_write_setting(self, connected_driver):
        drv, mock, _emulator, _recorder = connected_driver
        mock.clear_sent()
        await drv.write_setting("110", 750.000)
        assert b"$110=750.0\n" in sent_text(mock)

    async def test_set_wcs_offset_and_read(self, connected_driver):
        drv, mock, _emulator, recorder = connected_driver
        mock.clear_sent()
        await drv.set_wcs_offset("G55", 10, 20, None)
        assert b"G10 L2 P2 X10.0 Y20.0\n" in sent_text(mock)
        offsets = await drv.read_wcs_offsets()
        assert offsets["G55"] == (10.0, 20.0, 0.0)
        await recorder.wait_for("wcs_updated")

    async def test_read_settings_emits_varsets(self, connected_driver):
        drv, _mock, _emulator, _recorder = connected_driver

        class SettingsReceiver:
            def __init__(self, drv):
                self.collected: list = []
                drv.settings_read.connect(self._on_settings_read)

            def _on_settings_read(self, sender, settings=None):
                self.collected.append(settings)

        receiver = SettingsReceiver(drv)
        await drv.read_settings()
        assert receiver.collected
        values: dict[str, float] = {}
        for varset in receiver.collected[0]:
            for key in varset.keys():  # noqa: SIM118
                value = varset[key].value
                if isinstance(value, (int, float)):
                    values[varset[key].label] = float(value)
        assert "$110" in values
        assert values["$110"] != 0.0

    async def test_detect_unit_system(self, connected_driver):
        drv, _mock, _emulator, _recorder = connected_driver
        result = await drv.detect_unit_system()
        assert result is not None

    async def test_probe_cycle_success(self, connected_driver):
        drv, _mock, emulator, recorder = connected_driver
        emulator.probe_touch = (0.0, 0.0, -5.0)
        pos = await drv.run_probe_cycle(Axis.Z, 10.0, 100)
        assert pos is not None
        assert -10.0 < pos[2] <= 0.0
        await recorder.wait_for("probe_status_changed")

    def test_get_error(self):
        drv = GrblSerialNextDriver.__new__(GrblSerialNextDriver)
        error = drv.get_error("20")
        assert error is not None
        assert error.code == 20


@pytest.mark.asyncio
class TestStreaming:
    async def test_run_raw_completes(self, connected_driver):
        drv, _mock, _emulator, recorder = connected_driver
        await drv.run_raw("G1 X5 F1000\nG1 X0 F1000\n")
        await recorder.wait_for("job_finished")
        await asyncio.sleep(0.1)
        assert drv._session.buffer_count == 0

    async def test_error_halts_job(self, connected_driver):
        drv, _mock, _emulator, recorder = connected_driver
        await drv.run_raw("G999\nG1 X1 F1000\n")
        await recorder.wait_for("job_finished")
        await asyncio.sleep(0.05)
        states = [
            payload["state"]
            for name, payload in recorder.events
            if name == "state_changed"
        ]
        assert any(
            state is not None and state.error is not None for state in states
        )

    async def test_run_with_encoded_output_and_progress(
        self, connected_driver
    ):
        drv, _mock, _emulator, recorder = connected_driver
        encoder = drv.get_encoder()
        ops = Ops()
        ops.set_power(1.0)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 0, 0)
        encoded = encoder.encode(ops, drv._machine, Doc())
        done: list[int] = []

        def progress(op_index: int):
            done.append(op_index)

        await drv.run(encoded, Doc(), ops, progress)
        await recorder.wait_for("job_finished")
        assert done, "on_command_done was never called"

    async def test_cancel_within_job(self, connected_driver):
        drv, mock, emulator, _recorder = connected_driver
        emulator.speed_factor = 2

        async def job():
            await drv.run_raw(_slow_job(40))

        task = asyncio.ensure_future(job())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if drv._session.job_running:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("job never started")
        mock.clear_sent()
        await drv.cancel()
        await asyncio.sleep(0.3)
        assert task.done()
        sent = sent_text(mock)
        assert b"\x18" in sent
        assert b"M5\n" in sent


def _slow_job(n_lines: int) -> str:
    """Long alternating moves that keep the planner busy."""
    return "\n".join(f"G1 X{50 if i % 2 else 0} F600" for i in range(n_lines))
