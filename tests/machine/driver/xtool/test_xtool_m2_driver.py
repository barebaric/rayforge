"""Tests for the Rust-backed XtoolM2Driver shell.

The M2's WS-V2 protocol runs over TLS, which the hermetic test
environment cannot emulate without certificates, so these tests
cover the shell wiring (registry, setup, config, event translation)
without a live connection.  The live protocol behavior is covered
by raydriver's own M2 session tests."""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from rayforge.core.doc import Doc

import pytest
from raydriver.core import DriverEvent
from raydriver.core.types import DeviceState, DeviceStatus

from rayforge.machine.driver import get_driver_cls
from rayforge.machine.driver.driver import DriverMaturity
from rayforge.machine.driver.xtool import XtoolM2Driver
from rayforge.machine.models.laser import Laser
from rayforge.pipeline.artifact.encoded import EncodedOutput, MachineCodeOpMap


class TestRegistry:
    def test_driver_registered(self):
        assert get_driver_cls("XtoolM2Driver") is XtoolM2Driver

    def test_is_experimental(self):
        assert XtoolM2Driver.maturity == DriverMaturity.EXPERIMENTAL

    def test_metadata(self):
        assert XtoolM2Driver.DISCOVERY is None
        assert XtoolM2Driver.supports_settings is False
        assert XtoolM2Driver.uses_gcode is False
        assert XtoolM2Driver.supports_probing is False

    @pytest.mark.asyncio
    async def test_machine_space_wcs(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        assert drv.machine_space_wcs == "G53"

    def test_setup_vars(self):
        varset = XtoolM2Driver.get_setup_vars()
        keys = set(varset.keys())
        assert {"host", "port", "tls_allow_invalid_certs"} <= keys

    def test_precheck_rejects_invalid_host(self):
        with pytest.raises(Exception, match="[Hh]ostname"):
            XtoolM2Driver.precheck(host="not a host")


class TestSetup:
    def test_setup_requires_host(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        drv.setup()
        assert drv.state.error is not None
        assert "Hostname" in drv.state.error.title
        assert drv.resource_uri is None

    def test_setup_builds_m2_session(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        drv.setup(host="192.0.2.20")
        assert drv.did_setup
        assert drv.resource_uri == "wss://192.0.2.20:28900/websocket"


class _Recorder:
    """Collects blinker signal emissions from a driver."""

    def __init__(self, driver: XtoolM2Driver):
        self.events: list[tuple[str, dict]] = []
        self._keepalive: list = []
        driver.state_changed.connect(self._keep(self._send_state))
        driver.connection_status_changed.connect(
            self._keep(self._send_connection_status)
        )
        driver.job_finished.connect(self._keep(self._send_job_finished))

    def _keep(self, fn):
        self._keepalive.append(fn)
        return fn

    def _send_state(self, sender, state=None, **kwargs):
        self.events.append(("state_changed", {"state": state}))

    def _send_connection_status(self, sender, status=None, message=None):
        self.events.append(("connection_status_changed", {"status": status}))

    def _send_job_finished(self, sender):
        self.events.append(("job_finished", {}))


def make_session_event(state: DeviceState) -> DriverEvent:
    return DriverEvent.StateChanged(state=state)


class TestEventTranslation:
    def test_state_event_is_forwarded(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        recorder = _Recorder(drv)
        state = DeviceState()
        state.status = DeviceStatus.RUN
        drv._on_session_event(make_session_event(state))
        assert drv.state.status.name == "RUN"
        names = [name for name, _ in recorder.events]
        assert "state_changed" in names

    def test_connection_status_event_is_forwarded(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        recorder = _Recorder(drv)
        drv._on_session_event(
            DriverEvent.ConnectionStatusChanged(
                status="CONNECTED", message=None
            )
        )
        names = [name for name, _ in recorder.events]
        assert "connection_status_changed" in names
        assert recorder.events[0][1]["status"].name == "CONNECTED"

    def test_job_finished_event_is_forwarded(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        recorder = _Recorder(drv)
        drv._on_session_event(DriverEvent.JobFinished())
        assert ("job_finished", {}) in recorder.events


@pytest.mark.asyncio
class TestUnsupportedOperations:
    async def test_run_raises_explaining_no_streaming(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        encoded = EncodedOutput(text="M5\n", op_map=MachineCodeOpMap())
        doc = cast("Doc", None)
        ops = cast("Ops", None)
        with pytest.raises(Exception, match="Stage the job"):
            await drv.run(encoded, doc, ops)

    async def test_run_raw_raises_explaining_no_streaming(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        with pytest.raises(Exception, match="Stage the job"):
            await drv.run_raw("M5")

    async def test_power_control_unsupported(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        head = cast("Laser", None)

        with pytest.raises(Exception, match="laser power"):
            await drv.set_power(head, 0.5)
        with pytest.raises(Exception, match="laser power"):
            await drv.set_focus_power(head, 0.5)

    async def test_settings_unsupported(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        assert drv.get_setting_vars() == []
        with pytest.raises(Exception, match="settings"):
            await drv.write_setting("30", 1000)

    async def test_wcs_unsupported(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        assert drv.supported_wcs == []
        assert await drv.read_wcs_offsets() == {}
        assert await drv.read_parser_state() is None
        with pytest.raises(Exception, match="coordinate offsets"):
            await drv.set_wcs_offset("G54", 1, 2, None)

    async def test_probing_and_tool_change_unsupported(
        self, context_initializer, machine
    ):
        from raygeo.ops.axis import Axis

        drv = XtoolM2Driver(context_initializer, machine)
        with pytest.raises(Exception, match="probing"):
            await drv.run_probe_cycle(Axis.X, -10, 100)
        with pytest.raises(Exception, match="tool changes"):
            await drv.select_tool(1)

    async def test_home_single_axis_is_rejected(
        self, context_initializer, machine
    ):
        from raygeo.ops.axis import Axis

        drv = XtoolM2Driver(context_initializer, machine)
        assert drv.can_home() is True
        assert drv.can_home(Axis.X) is False
        with pytest.raises(Exception, match="homing all axes"):
            await drv.home(Axis.X)

    async def test_move_to_rejects_z(self, context_initializer, machine):
        drv = XtoolM2Driver(context_initializer, machine)
        with pytest.raises(Exception, match="Z moves"):
            await drv.move_to(1, 2, 3)

    async def test_clear_alarm_resets_local_state(
        self, context_initializer, machine
    ):
        drv = XtoolM2Driver(context_initializer, machine)
        drv._on_session_event(DriverEvent.StateChanged(state=DeviceState()))
        await drv.clear_alarm()
        assert drv.state.error is None
