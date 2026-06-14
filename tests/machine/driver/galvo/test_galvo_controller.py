"""
Tests for GalvoController.

Tests the low-level protocol implementation against MockGalvoConnection.
"""

import pytest

from rayforge.machine.driver.galvo.galvo_consts import (
    SOURCE_CO2,
    SOURCE_FIBER,
    listJumpTo,
    listEndOfList,
    listReadyMark,
    EnableLaser,
    SetControlMode,
    SetLaserMode,
    SetDelayMode,
    Fiber_SetMo,
)
from rayforge.machine.driver.galvo.galvo_controller import (
    GalvoController,
    DRIVER_STATE_IDLE,
    DRIVER_STATE_MARKING,
)
from rayforge.machine.driver.galvo.galvo_mock_connection import (
    MockGalvoConnection,
)


class TestGalvoController:
    def _controller_with_mock(self, source=SOURCE_FIBER):
        """Create a controller with a mock connection."""
        conn = MockGalvoConnection()
        ctrl = GalvoController(connection=conn, source=source)
        return ctrl, conn

    @pytest.mark.asyncio
    async def test_connect_initializes_controller(self):
        """Test that connect sends initialization sequence."""
        ctrl, conn = self._controller_with_mock()

        await ctrl.connect()

        assert ctrl.is_connected
        assert len(conn.sent_singles) > 0
        init_commands = {c.cmd for c in conn.sent_singles}
        assert EnableLaser in init_commands
        assert SetControlMode in init_commands
        assert SetLaserMode in init_commands
        assert SetDelayMode in init_commands

    @pytest.mark.asyncio
    async def test_connect_sends_fiber_mo_for_fiber(self):
        """Test that fiber source sends Fiber_SetMo."""
        ctrl, conn = self._controller_with_mock(SOURCE_FIBER)

        await ctrl.connect()

        cmds = [c.cmd for c in conn.sent_singles]
        assert Fiber_SetMo in cmds

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test that disconnect closes connection."""
        ctrl, conn = self._controller_with_mock()
        await ctrl.connect()

        await ctrl.disconnect()

        assert not ctrl.is_connected

    @pytest.mark.asyncio
    async def test_enter_marking_mode(self):
        """Test entering marking mode sends correct setup."""
        ctrl, conn = self._controller_with_mock()
        await ctrl.connect()

        await ctrl.enter_marking_mode()

        assert ctrl.state == DRIVER_STATE_MARKING

        ctrl._list_write(listJumpTo, 100, 200, 0, 500)
        await ctrl.enter_idle_mode()

    @pytest.mark.asyncio
    async def test_enter_idle_mode(self):
        """Test entering idle mode cleans up."""
        ctrl, conn = self._controller_with_mock()
        await ctrl.connect()

        await ctrl.enter_marking_mode()
        ctrl._list_write(listEndOfList)
        await ctrl.enter_idle_mode()

        assert ctrl.state == DRIVER_STATE_IDLE

    def test_goto(self):
        """Test that goto adds a jump command."""
        ctrl, _ = self._controller_with_mock()
        ctrl._list_write(listReadyMark)

        ctrl.goto(0x8000, 0x8000)

        assert ctrl._last_x == 0x8000
        assert ctrl._last_y == 0x8000

        ctrl.goto(0x9000, 0xA000)

        assert ctrl._last_x == 0x9000
        assert ctrl._last_y == 0xA000

    def test_goto_same_position(self):
        """Test that goto to same position does nothing."""
        ctrl, _ = self._controller_with_mock()

        ctrl.goto(0x8000, 0x8000)
        index_after = ctrl._active_index

        ctrl.goto(0x8000, 0x8000)
        assert ctrl._active_index == index_after

    def test_goto_clamps_coordinates(self):
        """Test that goto clamps to valid galvo range."""
        ctrl, _ = self._controller_with_mock()

        ctrl.goto(-10, 0x10000)
        assert ctrl._last_x == 0
        assert ctrl._last_y == 0xFFFF

    def test_mark(self):
        """Test that mark adds a mark command."""
        ctrl, _ = self._controller_with_mock()

        ctrl.mark(0x8000, 0x8000)
        assert ctrl._last_x == 0x8000
        assert ctrl._last_y == 0x8000

        ctrl.mark(0x9000, 0xA000)
        assert ctrl._last_x == 0x9000
        assert ctrl._last_y == 0xA000

    def test_goto_xy_sends_single(self):
        """Test that goto_xy sends a single command."""
        ctrl, conn = self._controller_with_mock()

        ctrl.goto_xy(0x8000, 0x8000)

        assert len(conn.sent_singles) > 0
        last = conn.sent_singles[-1]
        assert last.cmd == 0x000D

    def test_set_mark_speed(self):
        """Test that set_mark_speed adds list command."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_mark_speed(100.0)

    def test_set_travel_speed(self):
        """Test that set_travel_speed adds list command."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_travel_speed(2000.0)
        assert ctrl._travel_speed == 2000.0

        ctrl.set_travel_speed(2000.0)
        assert ctrl._travel_speed == 2000.0

    def test_set_power_fiber(self):
        """Test set_power for fiber source."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_power(50.0)
        assert ctrl._power == 50.0

    def test_set_power_co2(self):
        """Test set_power for CO2 source."""
        ctrl, _ = self._controller_with_mock(SOURCE_CO2)
        ctrl.set_frequency(30.0)

        ctrl.set_power(50.0)
        assert ctrl._power == 50.0

    def test_set_frequency_fiber(self):
        """Test set_frequency for fiber source."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_frequency(30.0)
        assert ctrl._frequency == 30.0

    def test_set_frequency_co2(self):
        """Test set_frequency for CO2 source."""
        ctrl, _ = self._controller_with_mock(SOURCE_CO2)

        ctrl.set_frequency(30.0)

    def test_set_laser_on_delay(self):
        """Test set_laser_on_delay."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_laser_on_delay(100.0)
        assert ctrl._delay_on == 100.0

    def test_set_laser_off_delay(self):
        """Test set_laser_off_delay."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_laser_off_delay(100.0)
        assert ctrl._delay_off == 100.0

    def test_set_polygon_delay(self):
        """Test set_polygon_delay."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_polygon_delay(100.0)
        assert ctrl._delay_poly == 100.0

    def test_set_pulse_width(self):
        """Test set_pulse_width."""
        ctrl, _ = self._controller_with_mock()

        ctrl.set_pulse_width(10)
        assert ctrl._pulse_width == 10

    def test_dwell(self):
        """Test dwell command."""
        ctrl, _ = self._controller_with_mock()
        initial_index = ctrl._active_index

        ctrl.dwell(10)

        assert ctrl._active_index > initial_index

    def test_wait(self):
        """Test wait command."""
        ctrl, _ = self._controller_with_mock()
        initial_index = ctrl._active_index

        ctrl.wait(10)

        assert ctrl._active_index > initial_index

    def test_port_on_off(self):
        """Test port manipulation."""
        ctrl, _ = self._controller_with_mock()

        ctrl._port_on(0)
        assert ctrl._port_bits & 1

        ctrl._port_off(0)
        assert not (ctrl._port_bits & 1)

    def test_source_property(self):
        """Test source property getter/setter."""
        ctrl, _ = self._controller_with_mock(SOURCE_FIBER)

        assert ctrl.source == SOURCE_FIBER
        ctrl.source = SOURCE_CO2
        assert ctrl.source == SOURCE_CO2

    def test_default_connection_on_connect(self):
        """Test that connect uses no connection initially."""
        ctrl = GalvoController(source=SOURCE_FIBER)
        assert ctrl._connection is None

    def test_is_connected_false_initially(self):
        """Test is_connected returns False before connect."""
        ctrl, _ = self._controller_with_mock()
        assert not ctrl.is_connected

    def test_convert_speed(self):
        """Test speed conversion."""
        ctrl, _ = self._controller_with_mock()
        result = ctrl._convert_speed(100.0)
        expected = int(100.0 * 500 / 1000.0)
        assert result == expected

    def test_convert_frequency(self):
        """Test frequency conversion."""
        ctrl, _ = self._controller_with_mock()
        result = ctrl._convert_frequency(30.0, base=20000.0)
        expected = int(round(20000.0 / 30.0)) & 0xFFFF
        assert result == expected

    def test_convert_power(self):
        """Test power conversion."""
        ctrl, _ = self._controller_with_mock()
        result = ctrl._convert_power(50.0)
        expected = int(round(50.0 * 0xFFF / 100.0))
        assert result == expected

    def test_set_fpk_for_co2(self):
        """Test set_fpk works for CO2."""
        ctrl, _ = self._controller_with_mock(SOURCE_CO2)
        ctrl.set_frequency(30.0)

        ctrl.set_fpk(50.0)
        assert ctrl._fpk == 50.0

    def test_set_fpk_ignored_for_fiber(self):
        """Test set_fpk is ignored for fiber."""
        ctrl, _ = self._controller_with_mock(SOURCE_FIBER)

        ctrl.set_fpk(50.0)
        assert ctrl._fpk is None
