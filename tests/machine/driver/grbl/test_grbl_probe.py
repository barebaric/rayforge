from unittest.mock import AsyncMock

import pytest
from raygeo.driver.grbl.probe import describe

from rayforge.machine.driver.grbl.grbl_probe import (
    build_grbl_profile,
)
from rayforge.machine.driver.grbl.grbl_serial import GrblSerialDriver
from rayforge.machine.transport import TransportStatus
from rayforge.shared.units.system import UnitSystem

FULL_BUILD_INFO = [
    "[VER:1.1h.ORTUR:]",
    "[OPT:VMPH,63,511]",
]
FULL_SETTINGS_LINES = [
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


class TestBuildGrblProfile:
    def test_full_profile(self):
        info = describe(FULL_BUILD_INFO, FULL_SETTINGS_LINES)
        profile, warnings = build_grbl_profile(info)
        assert profile.meta.name == "ORTUR"
        assert profile.machine_config.driver is None
        assert profile.machine_config.driver_args is None
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["rx_buffer_size"] == 511
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
        profile, _warnings = build_grbl_profile(describe([], []))
        assert profile.meta.name == "Unknown Grbl Device"
        assert profile.machine_config.axis_extents is None
        assert profile.machine_config.max_travel_speed is None
        assert profile.machine_config.heads is None
        assert profile.machine_config.home_on_start is None

    def test_laser_mode_disabled_warning(self):
        info = describe([], ["$32=0.0", "$13=0.0"])
        _profile, warnings = build_grbl_profile(info)
        assert len(warnings) == 1
        assert "laser mode" in warnings[0].lower()

    def test_report_inches_warning(self):
        info = describe([], ["$32=1.0", "$13=1.0"])
        profile, warnings = build_grbl_profile(info)
        assert len(warnings) == 1
        assert "inches" in warnings[0].lower()
        assert profile.machine_config.unit_system == UnitSystem.IMPERIAL

    def test_unit_system_defaults_to_metric(self):
        profile, _ = build_grbl_profile(describe([], []))
        assert profile.machine_config.unit_system == UnitSystem.METRIC

    def test_unit_system_metric_when_report_inches_zero(self):
        profile, _ = build_grbl_profile(describe([], ["$13=0.0"]))
        assert profile.machine_config.unit_system == UnitSystem.METRIC

    def test_different_speeds_per_axis(self):
        info = describe(
            [],
            [
                "$110=5000.0",
                "$111=3000.0",
                "$120=800.0",
                "$121=600.0",
            ],
        )
        profile, _warnings = build_grbl_profile(info)
        assert profile.machine_config.max_travel_speed == 3000
        assert profile.machine_config.acceleration == 600


class TestDialectMapping:
    def test_stock_grbl_gets_compat_dialect(self):
        info = describe(FULL_BUILD_INFO[:1], FULL_SETTINGS_LINES[:6])
        profile, _ = build_grbl_profile(info)
        assert profile.dialect_config
        assert "laser_on" in profile.dialect_config
        assert profile.dialect_config["laser_on"] == "M4 S{power:.0f}"

    def test_grblhal_gets_dynamic_dialect(self):
        info = describe(
            ["[VER:grblHAL 1.1f:]", "[OPT:VMP,31,511]"], ["$400=200.0"]
        )
        profile, _ = build_grbl_profile(info)
        assert profile.dialect_config
        assert profile.dialect_config["laser_on"] == "M4 S0"
        assert "s_command" in profile.dialect_config["linear_move"]

    def test_fluidnc_gets_dynamic_dialect(self):
        info = describe(["[VER:FluidNC 3.6.1:]", "[OPT:VMP,31,511]"], [])
        profile, _ = build_grbl_profile(info)
        assert profile.dialect_config
        assert profile.dialect_config["laser_on"] == "M4 S0"

    def test_unknown_firmware_gets_empty_dialect_config(self):
        profile, _ = build_grbl_profile(describe([], []))
        assert profile.dialect_config == {}


class TestDriverProbe:
    """
    Tests for GrblSerialDriver.probe() verifying that it creates
    a driver, connects, runs the device census over the session,
    and builds a profile.
    """

    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
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
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTED, message=None
            )

        mocker.patch.object(
            GrblSerialDriver,
            "_connect_implementation",
            fake_connect,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(GrblSerialDriver, "cleanup", mock_cleanup)
        info = describe(build_info, settings_lines)
        fake_session = mocker.MagicMock()
        mocker.patch.object(
            GrblSerialDriver,
            "session",
            new_callable=mocker.PropertyMock,
            return_value=fake_session,
        )
        probe_mock = mocker.patch.object(
            fake_session,
            "probe_device",
            AsyncMock(return_value=info),
        )

        profile, _warnings = await GrblSerialDriver.probe(
            context_initializer,
            port="/dev/ttyUSB0",
            baudrate=115200,
        )

        probe_mock.assert_awaited_once()
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
