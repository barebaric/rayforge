"""
Tests for RuidaClient command generation and sending.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from rayforge.machine.driver.ruida.ruida_client import RuidaClient


class TestRuidaClient:
    """Tests for RuidaClient methods."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connect delegates to transport."""
        await self.client.connect()
        self.mock_transport.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect delegates to transport."""
        await self.client.disconnect()
        self.mock_transport.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_command(self):
        """Test send_command delegates to transport."""
        command = b"\xda\x00"
        await self.client.send_command(command)
        self.mock_transport.send_command.assert_called_once_with(command)

    @pytest.mark.asyncio
    async def test_move_abs(self):
        """Test move_abs builds correct command."""
        await self.client.move_abs(10000, 20000)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0] == 0x88

    @pytest.mark.asyncio
    async def test_cut_abs(self):
        """Test cut_abs builds correct command."""
        await self.client.cut_abs(10000, 20000)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0] == 0xA8

    @pytest.mark.asyncio
    async def test_home_xy(self):
        """Test home_xy builds correct command."""
        await self.client.home_xy()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd8\x2a"

    @pytest.mark.asyncio
    async def test_start_process(self):
        """Test start_process builds correct command."""
        await self.client.start_process()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd8\x00"

    @pytest.mark.asyncio
    async def test_stop_process(self):
        """Test stop_process builds correct command."""
        await self.client.stop_process()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd8\x01"

    @pytest.mark.asyncio
    async def test_pause_process(self):
        """Test pause_process builds correct command."""
        await self.client.pause_process()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd8\x02"

    @pytest.mark.asyncio
    async def test_resume_process(self):
        """Test resume_process builds correct command."""
        await self.client.resume_process()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd8\x03"


class TestRuidaClientAirAssist:
    """Tests for air assist commands."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_air_assist_on(self):
        """Test air_assist_on sends correct command."""
        await self.client.air_assist_on()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xca\x13"

    @pytest.mark.asyncio
    async def test_air_assist_off(self):
        """Test air_assist_off sends correct command."""
        await self.client.air_assist_off()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xca\x12"


class TestRuidaClientSelectLayer:
    """Tests for layer selection commands."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("layer_index", [0, 5, 10, 15])
    async def test_select_layer_valid(self, layer_index):
        """Test select_layer with valid indices."""
        await self.client.select_layer(layer_index)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == bytes([0xCA, layer_index])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("layer_index", [-1, 16, 100])
    async def test_select_layer_invalid(self, layer_index):
        """Test select_layer with invalid indices raises error."""
        with pytest.raises(ValueError, match="Layer index must be 0-15"):
            await self.client.select_layer(layer_index)


class TestRuidaClientSendRaw:
    """Tests for send_raw command."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_send_raw(self):
        """Test send_raw sends data directly to transport."""
        data = b"\xda\x00\x05\x7e"
        await self.client.send_raw(data)
        self.mock_transport.send.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_send_raw_empty(self):
        """Test send_raw with empty data."""
        await self.client.send_raw(b"")
        self.mock_transport.send.assert_called_once_with(b"")


class TestRuidaClientJogCommands:
    """Tests for jog commands via main transport."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_jog_move_x(self):
        """Test jog_move_x sends rapid move X via main transport."""
        await self.client.jog_move_x(10000)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0] == 0xD9
        assert sent[1] == 0x00
        assert sent[2] == 0x02

    @pytest.mark.asyncio
    async def test_jog_move_y(self):
        """Test jog_move_y sends rapid move Y via main transport."""
        await self.client.jog_move_y(20000)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0] == 0xD9
        assert sent[1] == 0x01
        assert sent[2] == 0x02


class TestRuidaClientPowerCommands:
    """Tests for power commands."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("laser", [1, 2, 3, 4])
    async def test_set_power_immediate_valid_laser(self, laser):
        """Test set_power_immediate with valid laser numbers."""
        await self.client.set_power_immediate(laser, 50.0)
        self.mock_transport.send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_power_immediate_invalid_laser(self):
        """Test set_power_immediate with invalid laser number."""
        with pytest.raises(ValueError, match="Invalid laser"):
            await self.client.set_power_immediate(5, 50.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("laser", [1, 2, 3, 4])
    async def test_set_power_end_valid_laser(self, laser):
        """Test set_power_end with valid laser numbers."""
        await self.client.set_power_end(laser, 50.0)
        self.mock_transport.send_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_power_end_invalid_laser(self):
        """Test set_power_end with invalid laser number."""
        with pytest.raises(ValueError, match="Invalid laser"):
            await self.client.set_power_end(5, 50.0)


class TestRuidaClientSpeedCommands:
    """Tests for speed commands."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_set_speed(self):
        """Test set_speed builds correct command."""
        await self.client.set_speed(100.0)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0:2] == b"\xc9\x02"

    @pytest.mark.asyncio
    async def test_set_axis_speed(self):
        """Test set_axis_speed builds correct command."""
        await self.client.set_axis_speed(100.0)
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent[0:2] == b"\xc9\x03"


class TestRuidaClientEndOfFile:
    """Tests for end of file command."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "send_command",
                "is_connected",
                "received",
                "decoded_received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.send_command = AsyncMock()
        self.mock_transport.is_connected = False
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.decoded_received = MagicMock()
        self.mock_transport.decoded_received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

        self.client = RuidaClient(self.mock_transport)

    @pytest.mark.asyncio
    async def test_end_of_file(self):
        """Test end_of_file sends correct command."""
        await self.client.end_of_file()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xd7"

    @pytest.mark.asyncio
    async def test_keep_alive(self):
        """Test keep_alive sends correct command."""
        await self.client.keep_alive()
        self.mock_transport.send_command.assert_called_once()
        sent = self.mock_transport.send_command.call_args[0][0]
        assert sent == b"\xce"
