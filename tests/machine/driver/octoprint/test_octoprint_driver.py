import asyncio
import json

import pytest
import pytest_asyncio
from raydriver.octoprint.emulator import OctoPrintEmulator
from raygeo.ops.axis import Axis

from rayforge.core.varset import AppKeyVar, HostnameVar, PortVar
from rayforge.machine.driver.driver import (
    DeviceStatus,
    DriverMaturity,
    DriverPrecheckError,
)
from rayforge.machine.driver.octoprint import OctoPrintDriver
from rayforge.machine.driver.octoprint.octoprint_driver import (
    _extract_api_key,
    _normalize_path_prefix,
)

API_KEY = "test-api-key"


def _api_key_json(key=API_KEY):
    return json.dumps({"api_key": key})


async def wait_until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout:.1f}s")


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


def _push_state(emulator, text, **flags):
    emulator.set_state_text(text, **flags)


@pytest_asyncio.fixture
async def octoprint_device():
    emulator = OctoPrintEmulator(api_key=API_KEY)
    await emulator.start()
    try:
        yield emulator
    finally:
        await emulator.stop()


@pytest_asyncio.fixture
async def driver(lite_context, machine, octoprint_device):
    drv = OctoPrintDriver(lite_context, machine)
    drv.setup(
        host="127.0.0.1",
        port=octoprint_device.http_port,
        api_key=_api_key_json(),
    )
    try:
        yield drv
    finally:
        await drv.cleanup()


@pytest_asyncio.fixture
async def connected_driver(driver):
    await wait_for_connected(driver)
    await wait_until(lambda: driver.state.status == DeviceStatus.IDLE)
    yield driver


@pytest_asyncio.fixture
async def unconfigured_driver(lite_context, machine):
    return OctoPrintDriver(lite_context, machine)


class TestClassAttributes:
    def test_label(self):
        assert OctoPrintDriver.label == "OctoPrint"

    def test_subtitle(self):
        assert "OctoPrint" in OctoPrintDriver.subtitle

    def test_maturity(self):
        assert OctoPrintDriver.maturity == DriverMaturity.STABLE

    def test_uses_gcode(self):
        assert OctoPrintDriver.uses_gcode is True

    def test_reports_granular_progress(self):
        assert OctoPrintDriver.reports_granular_progress is False

    def test_supports_settings(self):
        assert OctoPrintDriver.supports_settings is False

    def test_mdns_services(self):
        spec = OctoPrintDriver.DISCOVERY
        assert spec is not None and spec.mdns is not None
        assert spec.mdns.services == ("_octoprint._tcp.local.",)


class TestProperties:
    def test_machine_space_wcs(self, connected_driver):
        assert connected_driver.machine_space_wcs == "G53"

    def test_machine_space_wcs_display_name(self, connected_driver):
        assert "G53" in connected_driver.machine_space_wcs_display_name

    def test_resource_uri(self, connected_driver, octoprint_device):
        assert connected_driver.resource_uri == (
            f"tcp://127.0.0.1:{octoprint_device.http_port}"
        )

    def test_resource_uri_no_host(self, unconfigured_driver):
        assert unconfigured_driver.resource_uri is None


class TestPrecheck:
    def test_valid_hostname(self):
        OctoPrintDriver.precheck(host="octoprint.local")

    def test_valid_ip(self):
        OctoPrintDriver.precheck(host="192.168.1.100")

    def test_empty_hostname_ok(self):
        OctoPrintDriver.precheck(host="")

    def test_invalid_hostname_raises(self):
        with pytest.raises(DriverPrecheckError):
            OctoPrintDriver.precheck(host="!!!invalid!!!")

    def test_none_hostname_ok(self):
        OctoPrintDriver.precheck(host=None)


class TestGetSetupVars:
    def test_returns_varset(self):
        vs = OctoPrintDriver.get_setup_vars()
        keys = list(vs.keys())
        assert "host" in keys
        assert "port" in keys
        assert "api_key" in keys
        assert isinstance(vs.get("host"), HostnameVar)
        assert isinstance(vs.get("port"), PortVar)
        assert isinstance(vs.get("api_key"), AppKeyVar)


class TestExtractApiKey:
    def test_from_appkey_json(self):
        data = json.dumps({"api_key": "abc123"})
        assert _extract_api_key(data) == "abc123"

    def test_from_oauth_json_backward_compat(self):
        data = json.dumps({"access_token": "abc123", "expires_at": None})
        assert _extract_api_key(data) == "abc123"

    def test_api_key_takes_precedence(self):
        data = json.dumps({"api_key": "new", "access_token": "old"})
        assert _extract_api_key(data) == "new"

    def test_raw_key_string(self):
        assert _extract_api_key("raw-key-xyz") == "raw-key-xyz"

    def test_empty_string(self):
        assert _extract_api_key("") is None

    def test_none(self):
        assert _extract_api_key(None) is None

    def test_whitespace_stripped(self):
        assert _extract_api_key("  key  ") == "key"


class TestPathPrefixHelpers:
    def test_root_is_empty(self):
        assert _normalize_path_prefix("/") == ""

    def test_empty_is_empty(self):
        assert _normalize_path_prefix("") == ""

    def test_custom_prefix_kept(self):
        assert _normalize_path_prefix("/octoprint") == "/octoprint"

    def test_trailing_slash_stripped(self):
        assert _normalize_path_prefix("/octoprint/") == "/octoprint"


class TestSetup:
    def test_setup_success(self, driver, octoprint_device):
        assert driver.did_setup is True
        assert driver.resource_uri == (
            f"tcp://127.0.0.1:{octoprint_device.http_port}"
        )

    @pytest.mark.asyncio
    async def test_setup_path_prefix(self, lite_context, machine):
        """A path from mDNS is normalized and used for API and
        WebSocket requests."""
        emulator = OctoPrintEmulator(api_key=API_KEY, path_prefix="/octoprint")
        await emulator.start()
        drv = OctoPrintDriver(lite_context, machine)
        try:
            drv.setup(
                host="127.0.0.1",
                port=emulator.http_port,
                api_key=_api_key_json(),
                path="/octoprint/",
            )
            await wait_for_connected(drv)
            await wait_until(lambda: drv.state.status == DeviceStatus.IDLE)
            await drv.run_raw("M999")
            await wait_until(lambda: {"command": "M999"} in emulator.commands)
        finally:
            await drv.cleanup()
            await emulator.stop()

    def test_setup_no_host_raises(self, unconfigured_driver):
        unconfigured_driver.setup(host="", port=80, api_key=_api_key_json())
        assert unconfigured_driver.state.error is not None
        assert "Hostname" in unconfigured_driver.state.error.title

    def test_setup_no_api_key_raises(self, unconfigured_driver):
        unconfigured_driver.setup(host="x", port=80, api_key="")
        assert unconfigured_driver.state.error is not None
        assert "API key" in unconfigured_driver.state.error.title

    def test_setup_raw_api_key(self, unconfigured_driver):
        unconfigured_driver.setup(host="x", port=80, api_key="plain-key")
        assert unconfigured_driver.did_setup is True


class TestConnection:
    @pytest.mark.asyncio
    async def test_connect_reports_idle_state(self, connected_driver):
        assert connected_driver.state.status == DeviceStatus.IDLE

    @pytest.mark.asyncio
    async def test_websocket_authenticates_immediately(
        self, connected_driver, octoprint_device
    ):
        # ``/sockjs/websocket`` sends no ``o`` open frame; the
        # driver must authenticate right after connecting.
        await wait_until(
            lambda: any("auth" in msg for msg in octoprint_device.ws_messages)
        )
        auth = json.loads(octoprint_device.ws_messages[0])
        assert auth == {"auth": "rayforge:emulator-session-key"}

    @pytest.mark.asyncio
    async def test_connect_with_wrong_api_key_reports_error(
        self, lite_context, machine
    ):
        emulator = OctoPrintEmulator(api_key=API_KEY)
        await emulator.start()
        drv = OctoPrintDriver(lite_context, machine)
        errors = []

        def _on_status(sender, status=None, message=None, **kwargs):
            if status is not None and status.name == "ERROR":
                errors.append(message)

        drv.connection_status_changed.connect(_on_status)
        try:
            drv.setup(
                host="127.0.0.1",
                port=emulator.http_port,
                api_key=_api_key_json("wrong-key"),
            )
            await drv.connect()
            await wait_until(lambda: errors)
        finally:
            drv.connection_status_changed.disconnect(_on_status)
            await drv.cleanup()
            await emulator.stop()
        assert "authentication failed" in errors[0]

    @pytest.mark.asyncio
    async def test_state_update_from_history_frame(
        self, connected_driver, octoprint_device
    ):
        await wait_until(lambda: octoprint_device.ws_messages)
        octoprint_device.send_frame(
            json.dumps(
                {
                    "history": {
                        "state": {
                            "text": "Paused",
                            "flags": {
                                "operational": True,
                                "paused": True,
                            },
                        }
                    }
                }
            )
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.HOLD
        )

    @pytest.mark.asyncio
    async def test_state_update_operational(
        self, connected_driver, octoprint_device
    ):
        _push_state(
            octoprint_device,
            "Operational",
            operational=True,
            printing=False,
            paused=False,
            error=False,
            closedOrError=False,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.IDLE
        )

    @pytest.mark.asyncio
    async def test_state_update_printing(
        self, connected_driver, octoprint_device
    ):
        _push_state(
            octoprint_device,
            "Printing",
            operational=True,
            printing=True,
            paused=False,
            error=False,
            closedOrError=False,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.RUN
        )

    @pytest.mark.asyncio
    async def test_state_update_paused(
        self, connected_driver, octoprint_device
    ):
        _push_state(
            octoprint_device,
            "Paused",
            operational=True,
            printing=False,
            paused=True,
            error=False,
            closedOrError=False,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.HOLD
        )

    @pytest.mark.asyncio
    async def test_state_update_error(
        self, connected_driver, octoprint_device
    ):
        _push_state(
            octoprint_device,
            "Error",
            operational=False,
            printing=False,
            paused=False,
            error=True,
            closedOrError=True,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.ALARM
        )

    @pytest.mark.asyncio
    async def test_state_update_offline(
        self, connected_driver, octoprint_device
    ):
        _push_state(
            octoprint_device,
            "Offline",
            operational=False,
            printing=False,
            paused=False,
            error=False,
            closedOrError=True,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.UNKNOWN
        )

    @pytest.mark.asyncio
    async def test_state_changed_signal(
        self, connected_driver, octoprint_device
    ):
        calls = []

        def _on_state(sender, state=None, **kwargs):
            calls.append(state)

        connected_driver.state_changed.connect(_on_state)
        try:
            _push_state(
                octoprint_device,
                "Printing",
                operational=True,
                printing=True,
                paused=False,
                error=False,
                closedOrError=False,
            )
            await wait_until(
                lambda: connected_driver.state.status == DeviceStatus.RUN
            )
            await wait_until(
                lambda: any(
                    state.status == DeviceStatus.RUN for state in calls
                )
            )
        finally:
            connected_driver.state_changed.disconnect(_on_state)

    @pytest.mark.asyncio
    async def test_connect_without_setup_reports_error(
        self, unconfigured_driver
    ):
        errors = []

        def _on_status(sender, status=None, message=None, **kwargs):
            if status is not None and status.name == "ERROR":
                errors.append(message)

        unconfigured_driver.connection_status_changed.connect(_on_status)
        try:
            await unconfigured_driver.connect()
            await wait_until(lambda: errors)
        finally:
            unconfigured_driver.connection_status_changed.disconnect(
                _on_status
            )
        assert errors[0] == "Connection not configured"


class TestRunRaw:
    @pytest.mark.asyncio
    async def test_run_raw_single_line(
        self, connected_driver, octoprint_device
    ):
        finished = []
        connected_driver.job_finished.connect(
            lambda sender, **kwargs: finished.append(sender),
            weak=False,
        )
        await connected_driver.run_raw("M999")
        await wait_until(
            lambda: {"command": "M999"} in octoprint_device.commands
        )
        await wait_until(lambda: finished)

    @pytest.mark.asyncio
    async def test_run_raw_multiline(self, connected_driver, octoprint_device):
        finished = []
        connected_driver.job_finished.connect(
            lambda sender, **kwargs: finished.append(sender),
            weak=False,
        )
        await connected_driver.run_raw("G28\nM999\nG0 X0")
        await wait_until(
            lambda: {"command": "G28"} in octoprint_device.commands
        )
        assert {"command": "M999"} in octoprint_device.commands
        assert {"command": "G0 X0"} in octoprint_device.commands
        await wait_until(lambda: finished)

    @pytest.mark.asyncio
    async def test_run_raw_empty(self, connected_driver, octoprint_device):
        before = len(octoprint_device.commands)
        await connected_driver.run_raw("")
        assert len(octoprint_device.commands) == before

    @pytest.mark.asyncio
    async def test_run_raw_sockjs_batch_push(
        self, connected_driver, octoprint_device
    ):
        """Batch frames from a framed peer are accepted too."""
        await wait_until(lambda: octoprint_device.ws_messages)
        payload = json.dumps(
            {
                "current": {
                    "state": {
                        "text": "Paused",
                        "flags": {
                            "operational": True,
                            "paused": True,
                        },
                    }
                }
            }
        )
        octoprint_device.send_frame("a" + json.dumps([payload]))
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.HOLD
        )


class TestHoldCancel:
    @pytest.mark.asyncio
    async def test_set_hold_pause(self, connected_driver, octoprint_device):
        await connected_driver.set_hold(True)
        await wait_until(
            lambda: (
                {
                    "command": "pause",
                    "action": "pause",
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_set_hold_resume(self, connected_driver, octoprint_device):
        await connected_driver.set_hold(False)
        await wait_until(
            lambda: (
                {
                    "command": "pause",
                    "action": "resume",
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_cancel_without_job_is_a_noop(
        self, connected_driver, octoprint_device
    ):
        # OctoPrint answers 409 when nothing is printing; that must
        # not surface as a device error.
        await connected_driver.cancel()
        await wait_until(
            lambda: {"command": "cancel"} in octoprint_device.commands
        )


class TestHoming:
    def test_can_home(self, connected_driver):
        assert connected_driver.can_home() is True

    @pytest.mark.asyncio
    async def test_home_all(self, connected_driver, octoprint_device):
        await connected_driver.home()
        await wait_until(
            lambda: (
                {
                    "command": "home",
                    "axes": ["x", "y", "z"],
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_home_x_only(self, connected_driver, octoprint_device):
        await connected_driver.home(Axis.X)
        await wait_until(
            lambda: (
                {
                    "command": "home",
                    "axes": ["x"],
                }
                in octoprint_device.commands
            )
        )


class TestJogMove:
    def test_can_jog(self, connected_driver):
        assert connected_driver.can_jog() is True

    @pytest.mark.asyncio
    async def test_move_to(self, connected_driver, octoprint_device):
        await connected_driver.move_to(10.5, 20.3)
        await wait_until(
            lambda: (
                {
                    "command": "jog",
                    "x": 10.5,
                    "y": 20.3,
                    "absolute": True,
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_move_to_with_z(self, connected_driver, octoprint_device):
        await connected_driver.move_to(10.5, 20.3, 5.0)
        await wait_until(
            lambda: (
                {
                    "command": "jog",
                    "x": 10.5,
                    "y": 20.3,
                    "z": 5.0,
                    "absolute": True,
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_move_to_with_speed(
        self, connected_driver, octoprint_device
    ):
        await connected_driver.move_to(10.5, 20.3, speed=6000)
        await wait_until(
            lambda: (
                {
                    "command": "jog",
                    "x": 10.5,
                    "y": 20.3,
                    "absolute": True,
                    "speed": 6000,
                }
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_jog(self, connected_driver, octoprint_device):
        await connected_driver.jog(1500, x=5.0, y=-3.0)
        await wait_until(
            lambda: (
                {
                    "command": "jog",
                    "speed": 1500,
                    "x": 5.0,
                    "y": -3.0,
                }
                in octoprint_device.commands
            )
        )


class TestPowerControl:
    @pytest.mark.asyncio
    async def test_set_power_on(self, connected_driver, octoprint_device):
        head = type("Head", (), {"max_power": 1000})()
        await connected_driver.set_power(head, 0.5)
        await wait_until(
            lambda: {"command": "M3 S500"} in octoprint_device.commands
        )

    @pytest.mark.asyncio
    async def test_set_power_off(self, connected_driver, octoprint_device):
        head = type("Head", (), {"max_power": 1000})()
        await connected_driver.set_power(head, 0)
        await wait_until(
            lambda: {"command": "M5"} in octoprint_device.commands
        )

    @pytest.mark.asyncio
    async def test_set_focus_power(self, connected_driver, octoprint_device):
        head = type("Head", (), {"max_power": 1000})()
        await connected_driver.set_focus_power(head, 0.3)
        await wait_until(
            lambda: {"command": "M3 S300"} in octoprint_device.commands
        )


class TestWcsOffset:
    @pytest.mark.asyncio
    async def test_set_wcs_offset_g54(
        self, connected_driver, octoprint_device
    ):
        await connected_driver.set_wcs_offset("G54", 1.0, 2.0, 3.0)
        await wait_until(
            lambda: (
                {"command": "G10 L2 P1 X1.000 Y2.000 Z3.000"}
                in octoprint_device.commands
            )
        )

    @pytest.mark.asyncio
    async def test_set_wcs_offset_invalid_raises(self, connected_driver):
        with pytest.raises(Exception, match="invalid WCS slot"):
            await connected_driver.set_wcs_offset("G99", 1, 2, 3)

    @pytest.mark.asyncio
    async def test_read_wcs_offsets_empty(self, connected_driver):
        assert await connected_driver.read_wcs_offsets() == {}


class TestClearAlarm:
    @pytest.mark.asyncio
    async def test_clear_alarm(self, connected_driver, octoprint_device):
        _push_state(
            octoprint_device,
            "Error",
            operational=False,
            printing=False,
            paused=False,
            error=True,
            closedOrError=True,
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.ALARM
        )
        await connected_driver.clear_alarm()
        await wait_until(
            lambda: {"command": "M999"} in octoprint_device.commands
        )
        await wait_until(
            lambda: connected_driver.state.status == DeviceStatus.IDLE
        )


class TestSettings:
    @pytest.mark.asyncio
    async def test_read_settings(self, connected_driver):
        calls = []

        def _on_settings(sender, settings=None, **kwargs):
            calls.append(settings)

        connected_driver.settings_read.connect(_on_settings)
        try:
            await connected_driver.read_settings()
            await wait_until(lambda: calls)
        finally:
            connected_driver.settings_read.disconnect(_on_settings)
        assert calls[0] == []

    @pytest.mark.asyncio
    async def test_write_setting_raises(self, connected_driver):
        with pytest.raises(NotImplementedError):
            await connected_driver.write_setting("key", "val")

    def test_get_setting_vars_empty(self, connected_driver):
        assert connected_driver.get_setting_vars() == []


class TestProbe:
    @pytest.mark.asyncio
    async def test_run_probe_cycle_sends_command(
        self, connected_driver, octoprint_device
    ):
        result = await connected_driver.run_probe_cycle(Axis.Z, -10.0, 100)
        await wait_until(
            lambda: (
                {"command": "G38.2 Z-10.000 F100"} in octoprint_device.commands
            )
        )
        assert result is None


class TestSelectTool:
    @pytest.mark.asyncio
    async def test_select_tool(self, connected_driver, octoprint_device):
        await connected_driver.select_tool(2)
        await wait_until(
            lambda: {"command": "T2"} in octoprint_device.commands
        )


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self, driver):
        await driver.cleanup()
        assert driver.did_setup is False
        assert driver.session is None
