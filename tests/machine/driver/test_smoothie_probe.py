from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from blinker import Signal

from rayforge.machine.driver.smoothie import SmoothieDriver
from rayforge.machine.driver.smoothie.smoothie_util import (
    SmoothieProbeResult,
    build_smoothie_profile,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.transport import TransportStatus


class TestBuildSmoothieProfile:
    def test_full_profile(self):
        profile, warnings = build_smoothie_profile(
            SmoothieProbeResult(
                version=["Smoothieware edge-1234abc1"],
                alpha_max=["200.000"],
                beta_max=["300.000"],
                alpha_max_rate=["500.000"],
                beta_max_rate=["600.000"],
                acceleration=["1000.000"],
            )
        )
        assert profile.meta.name == "Smoothieware"
        assert profile.machine_config.driver is None
        dc = profile.machine_config.driver_config
        assert dc is not None
        assert dc["firmware_version"] == "Smoothieware edge-1234abc1"
        assert profile.machine_config.axis_extents == (200.0, 300.0)
        # min(500,600) mm/s -> mm/min
        assert profile.machine_config.max_travel_speed == 30000
        assert profile.machine_config.max_cut_speed == 30000
        assert profile.machine_config.acceleration == 1000
        assert profile.machine_config.home_on_start is None
        assert profile.machine_config.single_axis_homing_enabled is True
        assert profile.machine_config.supports_arcs is True
        assert warnings == []

    def test_minimal_profile(self):
        profile, warnings = build_smoothie_profile(SmoothieProbeResult())
        assert profile.meta.name == "Smoothieware"
        assert profile.machine_config.axis_extents is None
        assert profile.machine_config.max_travel_speed is None
        assert profile.machine_config.acceleration is None
        assert profile.machine_config.heads is None
        assert profile.machine_config.driver_config is None
        # Three missing-value warnings (extents, feed rates, accel).
        assert len(warnings) == 3

    def test_different_feedrates_picks_lower(self):
        profile, _ = build_smoothie_profile(
            SmoothieProbeResult(
                alpha_max=["200.000"],
                beta_max=["200.000"],
                alpha_max_rate=["500.000"],
                beta_max_rate=["300.000"],
                acceleration=["1000.000"],
            )
        )
        assert profile.machine_config.max_travel_speed == 18000

    def test_non_positive_extents_warning(self):
        profile, warnings = build_smoothie_profile(
            SmoothieProbeResult(
                alpha_max=["0.000"],
                beta_max=["300.000"],
                alpha_max_rate=["500.000"],
                beta_max_rate=["500.000"],
                acceleration=["1000.000"],
            )
        )
        assert profile.machine_config.axis_extents is None
        assert len(warnings) == 1
        assert "work-area" in warnings[0].lower()

    def test_malformed_value_is_ignored(self):
        profile, warnings = build_smoothie_profile(
            SmoothieProbeResult(
                alpha_max=["not-a-number"],
                beta_max=["300.000"],
                alpha_max_rate=["500.000"],
                beta_max_rate=["500.000"],
                acceleration=["1000.000"],
            )
        )
        assert profile.machine_config.axis_extents is None
        assert len(warnings) == 1


def _make_mock_telnet():
    mock = AsyncMock()
    mock.received = Signal()
    mock.status_changed = Signal()
    mock.is_connected = True
    return mock


class TestDriverProbe:
    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
        mock_telnet = _make_mock_telnet()
        mocker.patch(
            "rayforge.machine.driver.smoothie.smoothie_driver.TelnetTransport",
            return_value=mock_telnet,
        )
        mocker.patch.object(
            mock_telnet,
            "is_connected",
            new_callable=PropertyMock,
            return_value=True,
        )

        async def fake_connect(self):
            self.on_telnet_status_changed(self, TransportStatus.CONNECTED)

        replies = {
            "version": ["Smoothieware edge-1234abc1"],
            "config-get alpha_max": ["200.000"],
            "config-get beta_max": ["300.000"],
            "config-get alpha_max_rate": ["500.000"],
            "config-get beta_max_rate": ["600.000"],
            "config-get acceleration": ["1000.000"],
        }

        async def fake_interactive(self, command):
            return list(replies.get(command, []))

        mocker.patch.object(
            SmoothieDriver, "_connect_implementation", fake_connect
        )
        mocker.patch.object(
            SmoothieDriver,
            "execute_interactive_command",
            fake_interactive,
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(SmoothieDriver, "cleanup", mock_cleanup)

        profile, _warnings = await SmoothieDriver.probe(
            context_initializer,
            host="127.0.0.1",
            port=23,
        )

        mock_cleanup.assert_awaited_once()
        assert profile.meta.name == "Smoothieware"
        assert profile.machine_config.driver == "SmoothieDriver"
        assert profile.machine_config.driver_args == {
            "host": "127.0.0.1",
            "port": 23,
        }
        assert profile.machine_config.axis_extents == (200.0, 300.0)
        assert profile.machine_config.max_travel_speed == 30000
        assert profile.machine_config.acceleration == 1000

    @pytest.mark.asyncio
    async def test_probe_cleanup_on_error(self, context_initializer, mocker):
        mock_telnet = _make_mock_telnet()
        mocker.patch(
            "rayforge.machine.driver.smoothie.smoothie_driver.TelnetTransport",
            return_value=mock_telnet,
        )

        async def failing_connect(self):
            raise ConnectionError("Connection refused")

        mocker.patch.object(
            SmoothieDriver, "_connect_implementation", failing_connect
        )
        mock_cleanup = AsyncMock()
        mocker.patch.object(SmoothieDriver, "cleanup", mock_cleanup)

        with pytest.raises(ConnectionError):
            await SmoothieDriver.probe(
                context_initializer,
                host="127.0.0.1",
                port=23,
            )

        mock_cleanup.assert_awaited_once()


class TestExecuteInteractiveCommand:
    """Verifies the driver collects framed reply lines until ``ok``
    and suppresses status-poll noise while a command is in flight."""

    @pytest.mark.asyncio
    async def test_collects_lines_until_ok(self, context_initializer):
        machine = Machine(context_initializer)
        machine.dialect_uid = "smoothieware"
        driver = SmoothieDriver(context_initializer, machine)

        mock_telnet = MagicMock()
        mock_telnet.received = Signal()
        mock_telnet.status_changed = Signal()
        mock_telnet.is_connected = True
        mock_telnet.send = AsyncMock()
        driver.telnet = mock_telnet

        async def fake_send(cmd: bytes, wait_for_ok: bool = True):
            await mock_telnet.send(cmd)
            # Simulate the device replying with the value then ok.
            payload = b"200.000\nok\n"
            driver.on_telnet_data_received(mock_telnet, payload)

        driver._send_and_wait = fake_send  # type: ignore

        lines = await driver.execute_interactive_command(
            "config-get alpha_max"
        )
        assert lines == ["200.000"]
        # The ok line is filtered out and not collected.
        assert "ok" not in lines

    @pytest.mark.asyncio
    async def test_status_reports_not_collected(self, context_initializer):
        machine = Machine(context_initializer)
        machine.dialect_uid = "smoothieware"
        driver = SmoothieDriver(context_initializer, machine)

        mock_telnet = MagicMock()
        mock_telnet.received = Signal()
        mock_telnet.status_changed = Signal()
        mock_telnet.is_connected = True
        mock_telnet.send = AsyncMock()
        driver.telnet = mock_telnet

        async def fake_send(cmd: bytes, wait_for_ok: bool = True):
            await mock_telnet.send(cmd)
            # A status report sneaks in before the real reply + ok.
            payload = b"<Idle|MPos:0,0,0|FS:0,0>\n200.000\nok\n"
            driver.on_telnet_data_received(mock_telnet, payload)

        driver._send_and_wait = fake_send  # type: ignore

        lines = await driver.execute_interactive_command(
            "config-get alpha_max"
        )
        assert lines == ["200.000"]
        assert "<Idle" not in lines

    @pytest.mark.asyncio
    async def test_not_connected_raises(self, context_initializer):
        machine = Machine(context_initializer)
        machine.dialect_uid = "smoothieware"
        driver = SmoothieDriver(context_initializer, machine)
        driver.telnet = None
        with pytest.raises(ConnectionError):
            await driver.execute_interactive_command("version")
