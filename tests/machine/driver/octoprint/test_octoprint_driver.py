import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from rayforge.core.ops import Axis
from rayforge.machine.driver.octoprint import OctoPrintDriver
from rayforge.machine.driver.octoprint.octoprint_driver import (
    _STATE_MAP,
)
from rayforge.machine.driver.driver import (
    DeviceConnectionError,
    DeviceStatus,
    DriverMaturity,
)
from rayforge.pipeline.encoder.gcode import GcodeEncoder


def _api_key_json(key="test-api-key-123"):
    return json.dumps({"api_key": key})


@pytest.fixture
def driver(lite_context, machine):
    machine.dialect_uid = "grbl"
    d = OctoPrintDriver(lite_context, machine)
    return d


@pytest.fixture
def setup_driver(driver):
    driver.setup(
        host="octoprint.local",
        port=80,
        api_key=_api_key_json("mykey"),
    )
    return driver


class TestClassAttributes:
    def test_label(self):
        assert OctoPrintDriver.label == "OctoPrint"

    def test_subtitle(self):
        assert "OctoPrint" in OctoPrintDriver.subtitle

    def test_maturity(self):
        assert OctoPrintDriver.maturity == DriverMaturity.UNTESTED

    def test_uses_gcode(self):
        assert OctoPrintDriver.uses_gcode is True

    def test_reports_granular_progress(self):
        assert OctoPrintDriver.reports_granular_progress is False

    def test_supports_settings(self):
        assert OctoPrintDriver.supports_settings is False


class TestProperties:
    def test_machine_space_wcs(self, setup_driver):
        assert setup_driver.machine_space_wcs == "G53"

    def test_machine_space_wcs_display_name(self, setup_driver):
        assert "G53" in setup_driver.machine_space_wcs_display_name

    def test_resource_uri(self, setup_driver):
        assert setup_driver.resource_uri == "tcp://octoprint.local:80"

    def test_resource_uri_no_host(self, driver):
        assert driver.resource_uri is None


class TestPrecheck:
    def test_valid_hostname(self):
        OctoPrintDriver.precheck(host="octoprint.local")

    def test_valid_ip(self):
        OctoPrintDriver.precheck(host="192.168.1.100")

    def test_empty_hostname_ok(self):
        OctoPrintDriver.precheck(host="")

    def test_invalid_hostname_raises(self):
        from rayforge.machine.driver.driver import DriverPrecheckError

        with pytest.raises(DriverPrecheckError):
            OctoPrintDriver.precheck(host="!!!invalid!!!")

    def test_none_hostname_ok(self):
        OctoPrintDriver.precheck(host=None)


class TestGetSetupVars:
    def test_returns_varset(self):
        from rayforge.core.varset import AppKeyVar, HostnameVar, PortVar

        vs = OctoPrintDriver.get_setup_vars()
        keys = list(vs.keys())
        assert "host" in keys
        assert "port" in keys
        assert "api_key" in keys
        assert isinstance(vs.get("host"), HostnameVar)
        assert isinstance(vs.get("port"), PortVar)
        assert isinstance(vs.get("api_key"), AppKeyVar)


class TestCreateEncoder:
    def test_returns_gcode_encoder(self, machine):
        machine.dialect_uid = "grbl"
        enc = OctoPrintDriver.create_encoder(machine)
        assert isinstance(enc, GcodeEncoder)


class TestExtractApiKey:
    def test_from_appkey_json(self):
        data = json.dumps({"api_key": "abc123"})
        assert OctoPrintDriver._extract_api_key(data) == "abc123"

    def test_from_oauth_json_backward_compat(self):
        data = json.dumps({"access_token": "abc123", "expires_at": None})
        assert OctoPrintDriver._extract_api_key(data) == "abc123"

    def test_api_key_takes_precedence(self):
        data = json.dumps({"api_key": "new", "access_token": "old"})
        assert OctoPrintDriver._extract_api_key(data) == "new"

    def test_raw_key_string(self):
        assert OctoPrintDriver._extract_api_key("raw-key-xyz") == "raw-key-xyz"

    def test_empty_string(self):
        assert OctoPrintDriver._extract_api_key("") is None

    def test_none(self):
        assert OctoPrintDriver._extract_api_key(None) is None

    def test_whitespace_stripped(self):
        assert OctoPrintDriver._extract_api_key("  key  ") == "key"


class TestSetup:
    def test_setup_success(self, driver):
        driver.setup(
            host="192.168.1.50",
            port=5000,
            api_key=_api_key_json("mykey"),
        )
        assert driver.host == "192.168.1.50"
        assert driver.port == 5000
        assert driver._api_key == "mykey"
        assert driver._base_url == "http://192.168.1.50:5000"
        assert driver.did_setup is True

    def test_setup_no_host_raises(self, driver):
        driver.setup(host="", port=80, api_key=_api_key_json())
        assert driver.state.error is not None
        assert "Hostname" in driver.state.error.title

    def test_setup_no_api_key_raises(self, driver):
        driver.setup(host="x", port=80, api_key="")
        assert driver.state.error is not None
        assert "API key" in driver.state.error.title

    def test_setup_raw_api_key(self, driver):
        driver.setup(host="x", port=80, api_key="plain-key")
        assert driver._api_key == "plain-key"


class TestApiRequest:
    @pytest.mark.asyncio
    async def test_api_request_sends_header(self, setup_driver, mocker):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.content_type = "application/json"
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"state": {}})
        mock_response.text = AsyncMock(return_value="")

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mocker.patch(
            "rayforge.machine.driver.octoprint.octoprint_driver"
            ".aiohttp.ClientSession",
            return_value=mock_session,
        )

        await setup_driver._api_request("GET", "/api/printer")
        call_kwargs = mock_session.request.call_args
        headers = call_kwargs[1].get("headers", {})
        assert headers["X-Api-Key"] == "mykey"

    @pytest.mark.asyncio
    async def test_api_request_403_raises(self, setup_driver, mocker):
        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.raise_for_status = MagicMock(
            side_effect=Exception("forbidden")
        )

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mocker.patch(
            "rayforge.machine.driver.octoprint.octoprint_driver"
            ".aiohttp.ClientSession",
            return_value=mock_session,
        )

        with pytest.raises(DeviceConnectionError, match="uthenti"):
            await setup_driver._api_request("GET", "/api/printer")

    @pytest.mark.asyncio
    async def test_api_request_204_returns_none(self, setup_driver, mocker):
        mock_response = AsyncMock()
        mock_response.status = 204
        mock_response.raise_for_status = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_ctx)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mocker.patch(
            "rayforge.machine.driver.octoprint.octoprint_driver"
            ".aiohttp.ClientSession",
            return_value=mock_session,
        )

        result = await setup_driver._api_request("POST", "/api/job")
        assert result is None


class TestStateMapping:
    def test_state_map_has_key_entries(self):
        assert _STATE_MAP["Operational"] == DeviceStatus.IDLE
        assert _STATE_MAP["Printing"] == DeviceStatus.RUN
        assert _STATE_MAP["Paused"] == DeviceStatus.HOLD
        assert _STATE_MAP["Error"] == DeviceStatus.ALARM
        assert _STATE_MAP["Offline"] == DeviceStatus.UNKNOWN

    def test_update_state_operational(self, setup_driver):
        setup_driver._update_state_from_octoprint(
            {
                "text": "Operational",
                "flags": {
                    "operational": True,
                    "printing": False,
                    "paused": False,
                    "error": False,
                    "closedOrError": False,
                },
            }
        )
        assert setup_driver.state.status == DeviceStatus.IDLE

    def test_update_state_printing(self, setup_driver):
        setup_driver._update_state_from_octoprint(
            {
                "text": "Printing",
                "flags": {
                    "operational": True,
                    "printing": True,
                    "paused": False,
                    "error": False,
                    "closedOrError": False,
                },
            }
        )
        assert setup_driver.state.status == DeviceStatus.RUN

    def test_update_state_paused(self, setup_driver):
        setup_driver._update_state_from_octoprint(
            {
                "text": "Paused",
                "flags": {
                    "operational": True,
                    "printing": False,
                    "paused": True,
                    "error": False,
                    "closedOrError": False,
                },
            }
        )
        assert setup_driver.state.status == DeviceStatus.HOLD

    def test_update_state_error(self, setup_driver):
        setup_driver._update_state_from_octoprint(
            {
                "text": "Error",
                "flags": {
                    "operational": False,
                    "printing": False,
                    "paused": False,
                    "error": True,
                    "closedOrError": True,
                },
            }
        )
        assert setup_driver.state.status == DeviceStatus.ALARM

    def test_update_state_offline(self, setup_driver):
        setup_driver._update_state_from_octoprint(
            {
                "text": "Offline",
                "flags": {
                    "operational": False,
                    "printing": False,
                    "paused": False,
                    "error": False,
                    "closedOrError": True,
                },
            }
        )
        assert setup_driver.state.status == DeviceStatus.UNKNOWN

    def test_state_changed_signal(self, setup_driver):
        tracker = MagicMock()
        setup_driver.state_changed.connect(tracker, weak=False)

        setup_driver._update_state_from_octoprint(
            {
                "text": "Printing",
                "flags": {
                    "operational": True,
                    "printing": True,
                    "paused": False,
                    "error": False,
                    "closedOrError": False,
                },
            }
        )

        tracker.assert_called_once()
        kwargs = tracker.call_args[1]
        assert kwargs["state"].status == DeviceStatus.RUN

        setup_driver.state_changed.disconnect(tracker)


class TestSockJSProcessing:
    def test_open_frame_ignored(self, setup_driver):
        setup_driver._process_sockjs_frame("o")

    def test_heartbeat_ignored(self, setup_driver):
        setup_driver._process_sockjs_frame("h")

    def test_close_frame_handled(self, setup_driver):
        setup_driver._process_sockjs_frame('c[3000,"bye"]')

    def test_data_frame_parsed(self, setup_driver):
        flags = {
            "operational": True,
            "printing": False,
            "paused": False,
            "error": False,
            "closedOrError": False,
        }
        state_data = {
            "text": "Operational",
            "flags": flags,
        }
        inner = json.dumps({"current": {"state": state_data}})
        frame = "a[" + json.dumps(inner) + "]"
        setup_driver._process_sockjs_frame(frame)
        assert setup_driver.state.status == DeviceStatus.IDLE

    def test_unknown_frame_ignored(self, setup_driver):
        setup_driver._process_sockjs_frame("x[123]")


class TestPushMessages:
    def test_print_done_event(self, setup_driver):
        setup_driver._job_active = True
        setup_driver._process_push_message({"event": {"type": "PrintDone"}})
        assert setup_driver.state.status == DeviceStatus.IDLE
        assert setup_driver._job_active is False

    def test_print_failed_event(self, setup_driver):
        setup_driver._job_active = True
        setup_driver._process_push_message({"event": {"type": "PrintFailed"}})
        assert setup_driver.state.status == DeviceStatus.ALARM
        assert setup_driver.state.error is not None
        assert setup_driver._job_active is False

    def test_print_cancelled_event(self, setup_driver):
        setup_driver._job_active = True
        setup_driver._process_push_message(
            {"event": {"type": "PrintCancelled"}}
        )
        assert setup_driver.state.status == DeviceStatus.IDLE

    def test_event_ignored_when_no_job(self, setup_driver):
        setup_driver._job_active = False
        setup_driver._process_push_message({"event": {"type": "PrintDone"}})
        assert setup_driver.state.status == DeviceStatus.UNKNOWN

    def test_current_update_with_state(self, setup_driver):
        setup_driver._process_push_message(
            {
                "current": {
                    "state": {
                        "text": "Operational",
                        "flags": {
                            "operational": True,
                            "printing": False,
                            "paused": False,
                            "error": False,
                            "closedOrError": False,
                        },
                    }
                }
            }
        )
        assert setup_driver.state.status == DeviceStatus.IDLE


class TestRunRaw:
    @pytest.mark.asyncio
    async def test_run_raw_single_line(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        tracker = MagicMock()
        setup_driver.job_finished.send = tracker

        await setup_driver.run_raw("M999")
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"command": "M999"},
        )
        tracker.assert_called_once_with(setup_driver)

    @pytest.mark.asyncio
    async def test_run_raw_multiline(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.run_raw("G28\nM999\nG0 X0")
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"commands": ["G28", "M999", "G0 X0"]},
        )

    @pytest.mark.asyncio
    async def test_run_raw_empty(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.run_raw("")
        setup_driver._api_request.assert_not_called()


class TestHoldCancel:
    @pytest.mark.asyncio
    async def test_set_hold_pause(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.set_hold(True)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/job",
            json={"command": "pause", "action": "pause"},
        )

    @pytest.mark.asyncio
    async def test_set_hold_resume(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.set_hold(False)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/job",
            json={"command": "pause", "action": "resume"},
        )

    @pytest.mark.asyncio
    async def test_cancel(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        tracker = MagicMock()
        setup_driver.job_finished.send = tracker
        await setup_driver.cancel()
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/job",
            json={"command": "cancel"},
        )
        tracker.assert_called_once_with(setup_driver)
        assert setup_driver._job_active is False


class TestHoming:
    def test_can_home(self, setup_driver):
        assert setup_driver.can_home() is True

    @pytest.mark.asyncio
    async def test_home_all(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.home()
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/printhead",
            json={"command": "home", "axes": ["x", "y", "z"]},
        )

    @pytest.mark.asyncio
    async def test_home_x_only(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.home(Axis.X)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/printhead",
            json={"command": "home", "axes": ["x"]},
        )


class TestJogMove:
    def test_can_jog(self, setup_driver):
        assert setup_driver.can_jog() is True

    @pytest.mark.asyncio
    async def test_move_to(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.move_to(10.5, 20.3)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/printhead",
            json={
                "command": "jog",
                "x": 10.5,
                "y": 20.3,
                "absolute": True,
            },
        )

    @pytest.mark.asyncio
    async def test_jog(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.jog(1500, x=5.0, y=-3.0)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/printhead",
            json={
                "command": "jog",
                "speed": 1500,
                "x": 5.0,
                "y": -3.0,
            },
        )


class TestPowerControl:
    @pytest.mark.asyncio
    async def test_set_power_on(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        head = MagicMock()
        head.max_power = 1000
        await setup_driver.set_power(head, 0.5)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"command": "M3 S500"},
        )

    @pytest.mark.asyncio
    async def test_set_power_off(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        head = MagicMock()
        await setup_driver.set_power(head, 0)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"command": "M5"},
        )

    @pytest.mark.asyncio
    async def test_set_focus_power_delegates(self, setup_driver, mocker):
        mocker.patch.object(setup_driver, "set_power", new_callable=AsyncMock)
        head = MagicMock()
        await setup_driver.set_focus_power(head, 0.3)
        setup_driver.set_power.assert_called_once_with(head, 0.3)


class TestWcsOffset:
    @pytest.mark.asyncio
    async def test_set_wcs_offset_g54(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.set_wcs_offset("G54", 1.0, 2.0, 3.0)
        call_args = setup_driver._api_request.call_args
        cmd = call_args[1]["json"]["command"]
        assert "G10 L2 P1" in cmd
        assert "X1.000" in cmd

    @pytest.mark.asyncio
    async def test_set_wcs_offset_invalid_raises(self, setup_driver, mocker):
        with pytest.raises(ValueError, match="Invalid WCS"):
            await setup_driver.set_wcs_offset("G99", 1, 2, 3)

    @pytest.mark.asyncio
    async def test_read_wcs_offsets_empty(self, setup_driver):
        result = await setup_driver.read_wcs_offsets()
        assert result == {}


class TestClearAlarm:
    @pytest.mark.asyncio
    async def test_clear_alarm(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.clear_alarm()
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"command": "M999"},
        )
        assert setup_driver.state.error is None
        assert setup_driver.state.status == DeviceStatus.IDLE


class TestSettings:
    @pytest.mark.asyncio
    async def test_read_settings(self, setup_driver):
        tracker = MagicMock()
        setup_driver.settings_read.send = tracker
        await setup_driver.read_settings()
        tracker.assert_called_once_with(setup_driver, settings=[])

    @pytest.mark.asyncio
    async def test_write_setting_raises(self, setup_driver):
        with pytest.raises(NotImplementedError):
            await setup_driver.write_setting("key", "val")

    def test_get_setting_vars_empty(self, setup_driver):
        assert setup_driver.get_setting_vars() == []


class TestProbe:
    @pytest.mark.asyncio
    async def test_run_probe_cycle_sends_command(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        result = await setup_driver.run_probe_cycle(Axis.Z, -10.0, 100)
        setup_driver._api_request.assert_called_once()
        assert result is None


class TestSelectTool:
    @pytest.mark.asyncio
    async def test_select_tool(self, setup_driver, mocker):
        mocker.patch.object(
            setup_driver, "_api_request", new_callable=AsyncMock
        )
        await setup_driver.select_tool(2)
        setup_driver._api_request.assert_called_once_with(
            "POST",
            "/api/printer/command",
            json={"command": "T2"},
        )


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_resets_state(self, setup_driver):
        await setup_driver.cleanup()
        assert setup_driver.keep_running is False
        assert setup_driver.did_setup is False


class TestConnectionNoHost:
    @pytest.mark.asyncio
    async def test_connect_no_host(self, driver):
        driver.host = None
        await driver._connect_implementation()
        assert driver._connection_task is None


class TestGetEncoder:
    def test_get_encoder_returns_gcode(self, setup_driver):
        enc = setup_driver.get_encoder()
        assert isinstance(enc, GcodeEncoder)
