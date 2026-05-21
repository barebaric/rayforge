from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from blinker import Signal

from rayforge.machine.driver.marlin.marlin_probe import build_marlin_profile
from rayforge.machine.driver.marlin.marlin_serial import MarlinSerialDriver
from rayforge.machine.transport import TransportStatus


class TestBuildMarlinProfile:
    def test_full_profile(self):
        m115_lines = [
            "FIRMWARE_NAME:Marlin 2.1.2.7 "
            "MACHINE_TYPE:CustomLaser EXTRUDER_COUNT:1"
        ]
        m211_lines = ["X: 400.00 Y: 300.00 Z: 200.00 S: 1 (Enabled)"]
        m503_lines = [
            "echo: M203 X300.00 Y300.00 Z5.00 E25.00",
            "echo: M204 S3000.00 T3000.00",
        ]
        boot_lines = ["Marlin 2.1.2.7"]
        profile, warnings = build_marlin_profile(
            m115_lines, m211_lines, m503_lines, boot_lines
        )
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
        assert warnings == []

    def test_minimal_profile(self):
        profile, warnings = build_marlin_profile([], [], [])
        assert profile.meta.name == "Unknown Marlin Device"
        assert profile.machine_config.axis_extents is None
        assert profile.machine_config.max_travel_speed is None
        assert profile.machine_config.acceleration is None
        assert profile.machine_config.heads is None

    def test_different_feedrates_picks_lower(self):
        m503_lines = [
            "echo: M203 X500.00 Y300.00 Z5.00 E25.00",
        ]
        profile, _ = build_marlin_profile([], [], m503_lines)
        assert profile.machine_config.max_travel_speed == 18000

    def test_no_boot_lines(self):
        m115_lines = ["FIRMWARE_NAME:Marlin 2.1.2.7 MACHINE_TYPE:TestDevice"]
        profile, _ = build_marlin_profile(m115_lines, [], [], None)
        assert profile.meta.name == "TestDevice"


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
    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
        mock_serial = _make_mock_serial()
        mocker.patch(
            "rayforge.machine.driver.marlin.marlin_serial.SerialTransport",
            return_value=mock_serial,
        )
        mocker.patch.object(
            mock_serial,
            "is_connected",
            new_callable=PropertyMock,
            return_value=True,
        )

        m115_lines = [
            "FIRMWARE_NAME:Marlin 2.1.2.7 "
            "MACHINE_TYPE:CustomLaser EXTRUDER_COUNT:1"
        ]
        m211_lines = ["X: 400.00 Y: 300.00 Z: 200.00 S: 1 (Enabled)"]
        m503_lines = [
            "echo: M203 X300.00 Y300.00 Z5.00 E25.00",
            "echo: M204 S3000.00 T3000.00",
        ]

        async def fake_connect(self):
            self._handshake_event.set()
            self._update_connection_status(TransportStatus.CONNECTED)

        async def fake_interactive(self, command):
            if command == "M115":
                return m115_lines
            if command == "M211":
                return m211_lines
            if command == "M503":
                return m503_lines
            return []

        mocker.patch.object(
            MarlinSerialDriver,
            "_connect_implementation",
            fake_connect,
        )
        mocker.patch.object(
            MarlinSerialDriver,
            "execute_interactive_command",
            fake_interactive,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(MarlinSerialDriver, "cleanup", mock_cleanup)

        profile, warnings = await MarlinSerialDriver.probe(
            context_initializer,
            port="/dev/ttyUSB0",
            baudrate=115200,
        )

        mock_cleanup.assert_awaited_once()
        assert profile.meta.name == "CustomLaser"
        assert profile.machine_config.driver == "MarlinSerialDriver"
        assert profile.machine_config.driver_args == {
            "port": "/dev/ttyUSB0",
            "baudrate": 115200,
        }
        assert profile.machine_config.axis_extents == (
            400.0,
            300.0,
        )
        assert profile.machine_config.max_travel_speed == 18000
        assert profile.machine_config.acceleration == 3000

    @pytest.mark.asyncio
    async def test_probe_cleanup_on_error(self, context_initializer, mocker):
        mock_serial = _make_mock_serial()
        mocker.patch(
            "rayforge.machine.driver.marlin.marlin_serial.SerialTransport",
            return_value=mock_serial,
        )

        async def failing_connect(self):
            raise ConnectionError("Port not found")

        mocker.patch.object(
            MarlinSerialDriver,
            "_connect_implementation",
            failing_connect,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(MarlinSerialDriver, "cleanup", mock_cleanup)

        with pytest.raises(ConnectionError):
            await MarlinSerialDriver.probe(
                context_initializer,
                port="/dev/ttyUSB0",
            )

        mock_cleanup.assert_awaited_once()
