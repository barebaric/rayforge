import asyncio
from unittest.mock import AsyncMock

import pytest

from rayforge.machine.driver.driver import DeviceStatus
from rayforge.machine.driver.smoothie import SmoothieDriver
from rayforge.machine.driver.smoothie.smoothie_util import (
    SmoothieProbeResult,
    build_smoothie_profile,
)
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


class TestDriverProbe:
    @pytest.mark.asyncio
    async def test_probe_connects_and_queries(
        self, context_initializer, mocker
    ):
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

        def fake_setup(self, **kwargs):
            self.did_setup = True

        async def fake_connect(self):
            self.connection_status_changed.send(
                self, status=TransportStatus.CONNECTED, message=None
            )

        mocker.patch.object(
            SmoothieDriver, "_setup_implementation", fake_setup
        )
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
        async def failing_connect(self):
            raise ConnectionError("Connection refused")

        def fake_setup(self, **kwargs):
            self.did_setup = True

        mocker.patch.object(
            SmoothieDriver, "_setup_implementation", fake_setup
        )
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


STATUS = b"<Idle|MPos:0.000,0.000,0.000|WPos:0.000,0.000,0.000|FS:0,0>\r\n"

FAST_CONFIG = {
    "handshake_timeout": 2.0,
    "handshake_poll_interval": 0.02,
    "status_poll_interval": 0.05,
    "reconnect_delay": 0.2,
    "command_timeout": 2.0,
}


class TestExecuteInteractiveCommand:
    """Verifies the driver collects reply lines until ``ok``,
    filters the ``ok`` terminator and status-poll noise, and raises
    when the driver is not connected.  Driven through the real Rust
    session against a mute scripted device."""

    async def mute_driver(self, context_initializer, machine):
        from raydriver.core.transport import MockTransport
        from raydriver.smoothie import SmoothieSession

        machine.dialect_uid = "smoothieware"
        drv = SmoothieDriver(context_initializer, machine)
        drv.did_setup = True
        mock = MockTransport()
        session = SmoothieSession.with_transport(
            mock,
            config=FAST_CONFIG,
            dialect=drv._dialect_templates(),
            event_callback=drv._on_session_event,
        )
        drv._session = session
        await drv.connect()
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            if mock.sent():
                break
            await asyncio.sleep(0.01)
        mock.push(STATUS)
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            if drv.state.status == DeviceStatus.IDLE:
                break
            await asyncio.sleep(0.01)
        mock.clear_sent()
        return drv, mock

    @pytest.mark.asyncio
    async def test_collects_lines_until_ok(self, context_initializer, machine):
        drv, mock = await self.mute_driver(context_initializer, machine)
        try:
            command = asyncio.ensure_future(
                drv.execute_interactive_command("config-get alpha_max")
            )
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                if b"config-get alpha_max" in b"".join(mock.sent()):
                    break
                await asyncio.sleep(0.01)
            mock.push(b"200.000\nok\n")
            lines = await asyncio.wait_for(command, timeout=5.0)
            assert lines == ["200.000"]
        finally:
            await drv.cleanup()

    @pytest.mark.asyncio
    async def test_status_reports_not_collected(
        self, context_initializer, machine
    ):
        drv, mock = await self.mute_driver(context_initializer, machine)
        try:
            command = asyncio.ensure_future(
                drv.execute_interactive_command("config-get alpha_max")
            )
            deadline = asyncio.get_running_loop().time() + 5.0
            while asyncio.get_running_loop().time() < deadline:
                if b"config-get alpha_max" in b"".join(mock.sent()):
                    break
                await asyncio.sleep(0.01)
            mock.push(b"<Idle|MPos:0,0,0|FS:0,0>\n200.000\nok\n")
            lines = await asyncio.wait_for(command, timeout=5.0)
            assert lines == ["200.000"]
            assert not any(line.startswith("<") for line in lines)
        finally:
            await drv.cleanup()

    @pytest.mark.asyncio
    async def test_not_connected_raises(self, context_initializer, machine):
        from raydriver.core.transport import MockTransport
        from raydriver.smoothie import SmoothieSession

        machine.dialect_uid = "smoothieware"
        drv = SmoothieDriver(context_initializer, machine)
        drv.did_setup = True
        drv._session = SmoothieSession.with_transport(
            MockTransport(),
            config=FAST_CONFIG,
            dialect=drv._dialect_templates(),
            event_callback=drv._on_session_event,
        )
        with pytest.raises(ConnectionError):
            await drv.execute_interactive_command("version")
