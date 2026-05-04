import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from blinker import Signal

from rayforge.machine.driver.grbl.grbl_probe import (
    build_grbl_profile,
)
from rayforge.machine.driver.grbl.grbl_serial import GrblSerialDriver
from rayforge.machine.driver.grbl.grbl_util import (
    parse_grbl_settings,
    parse_ver,
    parse_msg,
    extract_device_name,
)
from rayforge.machine.transport import TransportStatus


class TestParseGrblSettings:
    def test_parse_standard_settings(self):
        lines = [
            "$0=10",
            "$1=25",
            "$130=400.0",
            "$131=300.0",
        ]
        result = parse_grbl_settings(lines)
        assert result == {
            "0": 10.0,
            "1": 25.0,
            "130": 400.0,
            "131": 300.0,
        }

    def test_empty_lines(self):
        assert parse_grbl_settings([]) == {}

    def test_non_setting_lines_ignored(self):
        lines = ["ok", "[VER:1.1h:]", "some garbage"]
        assert parse_grbl_settings(lines) == {}

    def test_negative_value(self):
        assert parse_grbl_settings(["$23=3"]) == {"23": 3.0}

    def test_float_value(self):
        assert parse_grbl_settings(["$130=410.000"])["130"] == 410.0


class TestParseVer:
    def test_standard_with_build(self):
        assert parse_ver("[VER:1.1h.ORTUR:]") == (
            "1.1h",
            "ORTUR",
        )

    def test_standard_no_build(self):
        assert parse_ver("[VER:1.1h:]") == ("1.1h", None)

    def test_numeric_build(self):
        assert parse_ver("[VER:1.1h.2023062602:]") == (
            "1.1h",
            "2023062602",
        )

    def test_comma_format(self):
        assert parse_ver("[VER:1.0.15,20240923:]") == (
            "1.0.15",
            None,
        )

    def test_not_a_ver_line(self):
        assert parse_ver("[OPT:VMP,31,511]") is None
        assert parse_ver("") is None
        assert parse_ver("garbage") is None

    def test_empty_content(self):
        assert parse_ver("[VER::]") is None


class TestParseMsg:
    def test_machine_name(self):
        assert parse_msg("[MSG:mechine:Sculpfun iCube]") == (
            "mechine",
            "Sculpfun iCube",
        )

    def test_machine_spelling(self):
        assert parse_msg("[MSG:Machine:Some Device]") == (
            "Machine",
            "Some Device",
        )

    def test_no_colon(self):
        assert parse_msg("[MSG:Mode=BT]") is None

    def test_not_a_msg_line(self):
        assert parse_msg("[VER:1.1h:]") is None
        assert parse_msg("") is None


class TestExtractDeviceName:
    def test_name_from_ver_with_build_info(self):
        lines = ["[VER:1.1h.ORTUR:]", "[OPT:VMPH,63,511]"]
        assert extract_device_name(lines) == "ORTUR"

    def test_no_build_info(self):
        lines = ["[VER:1.1h:]", "[OPT:VMPH,63,511]"]
        assert extract_device_name(lines) == "Unknown Grbl Device"

    def test_empty_lines(self):
        assert extract_device_name([]) == "Unknown Grbl Device"

    def test_numeric_build_info(self):
        lines = ["[VER:1.1h.2023062602:]"]
        assert extract_device_name(lines) == "2023062602"

    def test_name_from_msg_machine(self):
        lines = [
            "[VER:1.0.15,20240923:]",
            "[OPT:VMP,31,511]",
            "[MSG:mechine:Sculpfun iCube]",
        ]
        assert extract_device_name(lines) == "Sculpfun iCube"

    def test_name_from_msg_machine_spelling(self):
        lines = ["[VER:1.1h:]", "[MSG:Machine:Some Device]"]
        assert extract_device_name(lines) == "Some Device"

    def test_comma_ver_no_msg(self):
        lines = ["[VER:1.0.15,20240923:]", "[OPT:VMP,31,511]"]
        assert extract_device_name(lines) == "Unknown Grbl Device"

    def test_msg_takes_priority_over_ver(self):
        lines = [
            "[VER:1.1h.ORTUR:]",
            "[MSG:mechine:Real Name]",
        ]
        assert extract_device_name(lines) == "Real Name"

    def test_msg_without_colon_ignored(self):
        lines = ["[MSG:Mode=BT]"]
        assert extract_device_name(lines) == "Unknown Grbl Device"


class TestBuildGrblProfile:
    def test_full_profile(self):
        build_info = [
            "[VER:1.1h.ORTUR:]",
            "[OPT:VMPH,63,511]",
        ]
        settings_lines = [
            "$110=3000.0",
            "$111=3000.0",
            "$120=500.0",
            "$121=500.0",
            "$130=400.0",
            "$131=300.0",
            "$22=1.0",
            "$30=1000.0",
            "$32=1.0",
            "$13=0.0",
            "$12=0.002",
        ]
        profile, warnings = build_grbl_profile(build_info, settings_lines)
        assert profile.meta.name == "ORTUR"
        assert profile.machine_config.driver is None
        assert profile.machine_config.driver_args is None
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["rx_buffer_size"] == 63
        assert dc["firmware_version"] == "1.1h"
        assert dc["arc_tolerance"] == 0.002
        assert profile.machine_config.axis_extents == (400.0, 300.0)
        assert profile.machine_config.max_travel_speed == 3000
        assert profile.machine_config.max_cut_speed == 3000
        assert profile.machine_config.acceleration == 500
        assert profile.machine_config.home_on_start is True
        assert profile.machine_config.heads == [{"max_power": 1000}]
        assert profile.machine_config.single_axis_homing_enabled is True
        assert warnings == []

    def test_minimal_profile(self):
        profile, warnings = build_grbl_profile([], [])
        assert profile.meta.name == "Unknown Grbl Device"
        assert profile.machine_config.axis_extents is None
        assert profile.machine_config.max_travel_speed is None
        assert profile.machine_config.heads is None
        assert profile.machine_config.home_on_start is None

    def test_laser_mode_disabled_warning(self):
        profile, warnings = build_grbl_profile([], ["$32=0.0", "$13=0.0"])
        assert len(warnings) == 1
        assert "laser mode" in warnings[0].lower()

    def test_report_inches_warning(self):
        profile, warnings = build_grbl_profile([], ["$32=1.0", "$13=1.0"])
        assert len(warnings) == 1
        assert "inches" in warnings[0].lower()

    def test_different_speeds_per_axis(self):
        profile, warnings = build_grbl_profile(
            [],
            [
                "$110=5000.0",
                "$111=3000.0",
                "$120=800.0",
                "$121=600.0",
            ],
        )
        assert profile.machine_config.max_travel_speed == 3000
        assert profile.machine_config.acceleration == 600


def _make_mock_serial():
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.send = AsyncMock()
    mock.received = Signal()
    mock.status_changed = Signal()
    mock.is_connected = True
    return mock


class TestDriverProbe:
    """
    Tests for GrblSerialDriver.probe() verifying that it creates
    a driver, connects, queries the device, and builds a profile.
    """

    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
        mock_serial = _make_mock_serial()
        mocker.patch(
            "rayforge.machine.driver.grbl.grbl_serial.SerialTransport",
            return_value=mock_serial,
        )
        mocker.patch.object(
            mock_serial,
            "is_connected",
            new_callable=PropertyMock,
            return_value=True,
        )

        build_info = [
            "[VER:1.1h.ORTUR:]",
            "[OPT:VMPH,63,511]",
        ]
        settings_lines = [
            "$130=400.000",
            "$131=300.000",
            "$110=3000.000",
            "$111=3000.000",
            "$120=500.000",
            "$121=500.000",
            "$30=1000.000",
            "$22=1.000",
            "$32=1.000",
            "$13=0.000",
        ]

        async def fake_connect(self):
            self._handshake_received.set()
            self._update_connection_status(TransportStatus.CONNECTED)

        async def fake_interactive(self, command):
            if command == "$I":
                return build_info
            if command == "$$":
                return settings_lines
            return []

        mocker.patch.object(
            GrblSerialDriver,
            "_connect_implementation",
            fake_connect,
        )
        mocker.patch.object(
            GrblSerialDriver,
            "execute_interactive_command",
            fake_interactive,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(GrblSerialDriver, "cleanup", mock_cleanup)

        profile, warnings = await GrblSerialDriver.probe(
            context_initializer,
            port="/dev/ttyUSB0",
            baudrate=115200,
        )

        mock_cleanup.assert_awaited_once()
        assert profile.meta.name == "ORTUR"
        assert profile.machine_config.driver == ("GrblSerialDriver")
        assert profile.machine_config.driver_args == {
            "port": "/dev/ttyUSB0",
            "baudrate": 115200,
        }
        assert profile.machine_config.axis_extents == (
            400.0,
            300.0,
        )
        assert profile.machine_config.max_travel_speed == 3000
        assert profile.machine_config.single_axis_homing_enabled is True

    @pytest.mark.asyncio
    async def test_probe_cleanup_on_error(self, context_initializer, mocker):
        mock_serial = _make_mock_serial()
        mocker.patch(
            "rayforge.machine.driver.grbl.grbl_serial.SerialTransport",
            return_value=mock_serial,
        )

        async def failing_connect(self):
            raise ConnectionError("Port not found")

        mocker.patch.object(
            GrblSerialDriver,
            "_connect_implementation",
            failing_connect,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(GrblSerialDriver, "cleanup", mock_cleanup)

        with pytest.raises(ConnectionError):
            await GrblSerialDriver.probe(
                context_initializer,
                port="/dev/ttyUSB0",
            )

        mock_cleanup.assert_awaited_once()
