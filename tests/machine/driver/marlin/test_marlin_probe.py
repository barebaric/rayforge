from unittest.mock import AsyncMock

import pytest
from raygeo.driver.marlin.probe import describe

from rayforge.machine.driver.marlin.marlin_probe import (
    build_marlin_profile,
)
from rayforge.machine.driver.marlin.marlin_serial import MarlinSerialDriver
from rayforge.machine.transport import TransportStatus
from rayforge.shared.units.system import UnitSystem

FULL_M115_LINES = [
    ("FIRMWARE_NAME:Marlin 2.1.2.7 MACHINE_TYPE:CustomLaser EXTRUDER_COUNT:1")
]
FULL_M211_LINES = ["X: 400.00 Y: 300.00 Z: 200.00 S: 1 (Enabled)"]
FULL_M503_LINES = [
    "echo: M203 X300.00 Y300.00 Z5.00 E25.00",
    "echo: M204 S3000.00 T3000.00",
]


class TestBuildMarlinProfile:
    def test_full_profile(self):
        info = describe(
            FULL_M115_LINES,
            FULL_M211_LINES,
            FULL_M503_LINES,
            [],
            ["Marlin 2.1.2.7"],
        )
        profile, warnings = build_marlin_profile(info)
        assert profile.meta.name == "CustomLaser"
        assert profile.machine_config.driver is None
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["firmware_version"] == "2.1.2.7"
        assert profile.machine_config.axis_extents == (400.0, 300.0)
        assert profile.machine_config.max_travel_speed == 18000
        assert profile.machine_config.max_cut_speed == 18000
        assert profile.machine_config.acceleration == 3000
        assert profile.machine_config.single_axis_homing_enabled is True
        assert profile.machine_config.supports_arcs is True
        assert profile.machine_config.unit_system == UnitSystem.METRIC
        assert warnings == []

    def test_minimal_profile(self):
        profile, _warnings = build_marlin_profile(describe([], [], [], []))
        assert profile.meta.name == "Unknown Marlin Device"
        assert profile.machine_config.axis_extents is None
        assert profile.machine_config.max_travel_speed is None
        assert profile.machine_config.acceleration is None
        assert profile.machine_config.heads is None

    def test_different_feedrates_picks_lower(self):
        m503_lines = [
            "echo: M203 X500.00 Y300.00 Z5.00 E25.00",
        ]
        info = describe([], [], m503_lines, [])
        profile, _ = build_marlin_profile(info)
        assert profile.machine_config.max_travel_speed == 18000

    def test_no_boot_lines(self):
        m115_lines = ["FIRMWARE_NAME:Marlin 2.1.2.7 MACHINE_TYPE:TestDevice"]
        info = describe(m115_lines, [], [], [], None)
        profile, _ = build_marlin_profile(info)
        assert profile.meta.name == "TestDevice"

    def test_unit_system_metric(self):
        m149_lines = ["echo: M149 Units in mm"]
        info = describe([], [], [], m149_lines)
        profile, _ = build_marlin_profile(info)
        assert profile.machine_config.unit_system == UnitSystem.METRIC

    def test_unit_system_imperial(self):
        m149_lines = ["echo: M149 Units in inches"]
        info = describe([], [], [], m149_lines)
        profile, _ = build_marlin_profile(info)
        assert profile.machine_config.unit_system == UnitSystem.IMPERIAL

    def test_unit_system_unknown_defaults_to_metric(self):
        profile, _ = build_marlin_profile(describe([], [], [], []))
        assert profile.machine_config.unit_system == UnitSystem.METRIC


class TestDriverProbe:
    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
        m149_lines = ["echo: M149 Units in mm"]

        def fake_setup(self, **kwargs):
            self.did_setup = True

        async def fake_connect(self):
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTED, message=None
            )

        mocker.patch.object(
            MarlinSerialDriver, "_setup_implementation", fake_setup
        )
        mocker.patch.object(
            MarlinSerialDriver,
            "_connect_implementation",
            fake_connect,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(MarlinSerialDriver, "cleanup", mock_cleanup)
        fake_session = mocker.MagicMock()
        mocker.patch.object(
            MarlinSerialDriver,
            "session",
            new_callable=mocker.PropertyMock,
            return_value=fake_session,
        )
        info = describe(
            FULL_M115_LINES,
            FULL_M211_LINES,
            FULL_M503_LINES,
            m149_lines,
        )
        probe_mock = mocker.patch.object(
            fake_session,
            "probe_device",
            AsyncMock(return_value=info),
        )

        profile, _warnings = await MarlinSerialDriver.probe(
            context_initializer,
            port="/dev/ttyUSB0",
            baudrate=115200,
        )

        probe_mock.assert_awaited_once()
        assert probe_mock.await_args.args == ([],)
        assert profile.meta.name == "CustomLaser"
        driver_config = profile.machine_config.driver_config
        assert driver_config is not None
        assert driver_config["firmware_version"] == "Marlin"
        assert profile.machine_config.axis_extents == (400.0, 300.0)
        assert profile.machine_config.max_travel_speed == 300 * 60
        assert profile.machine_config.acceleration == 3000
        assert profile.machine_config.unit_system == UnitSystem.METRIC
        mock_cleanup.assert_awaited_once()
