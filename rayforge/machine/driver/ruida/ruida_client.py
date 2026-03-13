"""
Ruida Client Protocol - Client-side command generation and sending.

Handles generation of commands to send to a Ruida laser controller,
sending them via transport, and parsing of responses.
"""

import asyncio
import logging
from typing import Dict, Optional, TYPE_CHECKING

from blinker import Signal

from ...transport.transport import Transport
from .ruida_maps import (
    REF_POINT_COMMANDS,
    REF_POINT_MODE_TO_NAME,
    REF_POINT_OFFSET_ADDRESSES,
)
from .ruida_protocol import RuidaResponse, RuidaState
from .ruida_util import encode14, encode35, decode35

if TYPE_CHECKING:
    from .ruida_transport import RuidaTransport

logger = logging.getLogger(__name__)


class RuidaClient:
    """
    Ruida client-side protocol handler.

    Generates commands to send to a Ruida controller, sends them via
    the transport layer, and parses responses.

    Usage:
        transport = RuidaTransport(UdpTransport(host, port))
        client = RuidaClient(transport)
        await client.connect()
        await client.home_xy()
        await client.move_abs(10000, 20000)  # in micrometers
    """

    def __init__(
        self,
        transport: "RuidaTransport",
        state: Optional[RuidaState] = None,
        jog_transport: Optional[Transport] = None,
    ):
        self._transport = transport
        self._jog_transport = jog_transport
        self.state = state or RuidaState()
        self._pending_mem_reads: Dict[int, asyncio.Future] = {}
        self.position_updated = Signal()
        self.state_changed = Signal()

        self._transport.decoded_received.connect(self._handle_response)

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    def _handle_response(self, sender, data: bytes) -> None:
        """
        Handle decoded data from the transport layer.

        Parses DA memory read responses and emits signals.
        Also resolves any pending synchronous memory reads.

        Args:
            sender: The signal sender (unused)
            data: The decoded response data
        """
        pending = list(self._pending_mem_reads.keys())
        logger.debug(f"handle_response: {data.hex()} (pending: {pending})")
        if len(data) >= 14 and data[0] == 0xDA and data[1] == 0x01:
            mem_address = (data[2] << 8) | data[3]
            value = decode35(data[4:9])

            if mem_address in self._pending_mem_reads:
                future = self._pending_mem_reads.pop(mem_address)
                if not future.done():
                    future.set_result(value)

            axis = None
            if mem_address == 0x0421:
                axis = "x"
                self.state.x = value
            elif mem_address == 0x0431:
                axis = "y"
                self.state.y = value
            elif mem_address == 0x0441:
                axis = "z"
                self.state.z = value

            if axis:
                logger.debug(
                    f"Position response: {axis}={value}um "
                    f"(mem 0x{mem_address:04X})"
                )
                self.position_updated.send(self, axis=axis, value_um=value)

        self.state_changed.send(self)

    async def connect(self) -> None:
        """Establish connection to the Ruida controller."""
        await self._transport.connect()
        if self._jog_transport:
            await self._jog_transport.connect()

    async def disconnect(self) -> None:
        """Close connection to the Ruida controller."""
        if self._jog_transport:
            await self._jog_transport.disconnect()
        await self._transport.disconnect()

    def parse_response(self, data: bytes) -> RuidaResponse:
        return RuidaResponse.from_bytes(data)

    async def send_command(self, command: bytes) -> None:
        """
        Send a raw command to the controller.

        Args:
            command: Raw command bytes (will be swizzled and framed)
        """
        await self._transport.send_command(command)

    async def send_jog_command(self, command: bytes) -> None:
        """
        Send a raw jog command to the controller.

        Jog commands are sent without swizzling or framing.

        Args:
            command: Raw command bytes
        """
        if self._jog_transport:
            await self._jog_transport.send(command)
        else:
            await self._transport.send(command)

    async def move_abs(self, x: int, y: int) -> None:
        """
        Move to absolute position (traversal, laser off).

        Args:
            x: X coordinate in micrometers
            y: Y coordinate in micrometers
        """
        await self.send_command(self._build_move_abs(x, y))

    async def move_rel(self, dx: int, dy: int) -> None:
        """
        Move by relative offset (traversal, laser off).

        Args:
            dx: X offset in micrometers
            dy: Y offset in micrometers
        """
        await self.send_command(self._build_move_rel(dx, dy))

    async def cut_abs(self, x: int, y: int) -> None:
        """
        Move to absolute position (cutting, laser on).

        Args:
            x: X coordinate in micrometers
            y: Y coordinate in micrometers
        """
        await self.send_command(self._build_cut_abs(x, y))

    async def cut_rel(self, dx: int, dy: int) -> None:
        """
        Move by relative offset (cutting, laser on).

        Args:
            dx: X offset in micrometers
            dy: Y offset in micrometers
        """
        await self.send_command(self._build_cut_rel(dx, dy))

    async def move_rel_x(self, dx: int) -> None:
        """
        Move X axis by relative offset (traversal, laser off).

        Args:
            dx: X offset in micrometers
        """
        await self.send_command(self._build_move_rel_x(dx))

    async def move_rel_y(self, dy: int) -> None:
        """
        Move Y axis by relative offset (traversal, laser off).

        Args:
            dy: Y offset in micrometers
        """
        await self.send_command(self._build_move_rel_y(dy))

    async def cut_rel_x(self, dx: int) -> None:
        """
        Move X axis by relative offset (cutting, laser on).

        Args:
            dx: X offset in micrometers
        """
        await self.send_command(self._build_cut_rel_x(dx))

    async def cut_rel_y(self, dy: int) -> None:
        """
        Move Y axis by relative offset (cutting, laser on).

        Args:
            dy: Y offset in micrometers
        """
        await self.send_command(self._build_cut_rel_y(dy))

    async def rapid_move_xy(
        self, x: int, y: int, origin: bool = False, light: bool = False
    ) -> None:
        """
        Rapid move to absolute XY position.

        Args:
            x: X coordinate in micrometers
            y: Y coordinate in micrometers
            origin: Move relative to stored origin point
            light: Enable laser pointer during move
        """
        await self.send_command(self._build_rapid_move_xy(x, y, origin, light))

    async def rapid_move_axis(
        self,
        axis: int,
        coord: int,
        origin: bool = False,
        light: bool = False,
    ) -> None:
        """
        Rapid move on a single axis.

        Args:
            axis: Axis number (0x10=X, 0x11=Y, 0x12=Z, 0x13=U)
            coord: Coordinate in micrometers
            origin: Move relative to stored origin point
            light: Enable laser pointer during move
        """
        await self.send_command(
            self._build_rapid_move_axis(axis, coord, origin, light)
        )

    async def home_xy(self) -> None:
        """Home the X and Y axes."""
        await self.send_command(self._build_home_xy())

    async def home_z(self) -> None:
        """Home the Z axis."""
        await self.send_command(self._build_home_z())

    async def home_u(self) -> None:
        """Home the U axis."""
        await self.send_command(self._build_home_u())

    async def start_process(self) -> None:
        """Start the laser cutting process."""
        await self.send_command(self._build_start_process())

    async def stop_process(self) -> None:
        """Stop the laser cutting process."""
        await self.send_command(self._build_stop_process())

    async def pause_process(self) -> None:
        """Pause the laser cutting process."""
        await self.send_command(self._build_pause_process())

    async def resume_process(self) -> None:
        """Resume the paused laser cutting process."""
        await self.send_command(self._build_resume_process())

    async def set_ref_point_0(self) -> None:
        """Set reference point 0 (current position)."""
        await self.send_command(self._build_ref_point_0())

    async def set_ref_point_1(self) -> None:
        """Set reference point 1 (current position)."""
        await self.send_command(self._build_ref_point_1())

    async def set_ref_point_2(self) -> None:
        """Set reference point 2 (machine zero/absolute position)."""
        await self.send_command(self._build_ref_point_2())

    async def set_absolute_mode(self) -> None:
        """Set absolute coordinate mode."""
        await self.send_command(b"\xe6\x01")

    async def commit_ref_point(self) -> None:
        """Commit the current reference point setting."""
        await self.send_command(b"\xf0")

    async def jog_start(self, axis: str, direction: int) -> None:
        """
        Start continuous jog on an axis.

        Args:
            axis: Axis name ('x', 'y', 'z', or 'u')
            direction: Direction (1 for positive, -1 for negative)
        """
        await self.send_command(self._build_jog_keydown(axis, direction))

    async def jog_stop(self, axis: str) -> None:
        """
        Stop continuous jog on an axis.

        Args:
            axis: Axis name ('x', 'y', 'z', or 'u')
        """
        await self.send_command(self._build_jog_keyup(axis))

    async def jog_rel_x(self, dx: int) -> None:
        """
        Jog X axis by relative offset using D8 command.

        Args:
            dx: X offset in micrometers
        """
        await self.send_jog_command(self._build_jog_rel_x(dx))

    async def jog_rel_y(self, dy: int) -> None:
        """
        Jog Y axis by relative offset using D8 command.

        Args:
            dy: Y offset in micrometers
        """
        await self.send_jog_command(self._build_jog_rel_y(dy))

    async def set_power_immediate(
        self, laser: int, power_percent: float
    ) -> None:
        """
        Set laser power immediately (takes effect right away).

        Args:
            laser: Laser number (1-4)
            power_percent: Power level (0.0 to 100.0)
        """
        await self.send_command(
            self._build_power_immediate(laser, power_percent)
        )

    async def set_power_end(self, laser: int, power_percent: float) -> None:
        """
        Set laser power at end of move (takes effect after current move).

        Args:
            laser: Laser number (1-4)
            power_percent: Power level (0.0 to 100.0)
        """
        await self.send_command(self._build_power_end(laser, power_percent))

    async def set_speed(self, speed_mm_s: float) -> None:
        """
        Set movement speed.

        Args:
            speed_mm_s: Speed in millimeters per second
        """
        await self.send_command(self._build_speed(speed_mm_s))

    async def set_axis_speed(self, speed_mm_s: float) -> None:
        """
        Set axis-specific speed.

        Args:
            speed_mm_s: Speed in millimeters per second
        """
        await self.send_command(self._build_axis_speed(speed_mm_s))

    async def end_of_file(self) -> None:
        """Send end-of-file marker."""
        await self.send_command(self._build_end_of_file())

    async def keep_alive(self) -> None:
        """Send keep-alive packet to maintain connection."""
        await self.send_command(self._build_keep_alive())

    def _build_move_abs(self, x: int, y: int) -> bytes:
        return b"\x88" + encode35(x) + encode35(y)

    def _build_move_rel(self, dx: int, dy: int) -> bytes:
        return b"\x89" + encode14(dx) + encode14(dy)

    def _build_cut_abs(self, x: int, y: int) -> bytes:
        return b"\xa8" + encode35(x) + encode35(y)

    def _build_cut_rel(self, dx: int, dy: int) -> bytes:
        return b"\xa9" + encode14(dx) + encode14(dy)

    def _build_move_rel_x(self, dx: int) -> bytes:
        return b"\x8a" + encode14(dx)

    def _build_move_rel_y(self, dy: int) -> bytes:
        return b"\x8b" + encode14(dy)

    def _build_cut_rel_x(self, dx: int) -> bytes:
        return b"\xaa" + encode14(dx)

    def _build_cut_rel_y(self, dy: int) -> bytes:
        return b"\xab" + encode14(dy)

    def _build_rapid_move_xy(
        self, x: int, y: int, origin: bool = False, light: bool = False
    ) -> bytes:
        opts = self._build_move_opts(origin, light)
        return b"\xd9\x60" + bytes([opts]) + encode35(x) + encode35(y)

    def _build_rapid_move_axis(
        self,
        axis: int,
        coord: int,
        origin: bool = False,
        light: bool = False,
    ) -> bytes:
        opts = self._build_move_opts(origin, light)
        return b"\xd9" + bytes([axis & 0x0F]) + bytes([opts]) + encode35(coord)

    def _build_move_opts(self, origin: bool, light: bool) -> int:
        if origin and light:
            return 0x01
        elif origin:
            return 0x00
        elif light:
            return 0x03
        return 0x02

    def _build_home_xy(self) -> bytes:
        return b"\xd8\x2a"

    def _build_home_z(self) -> bytes:
        return b"\xd8\x2c"

    def _build_home_u(self) -> bytes:
        return b"\xd8\x2d"

    def _build_start_process(self) -> bytes:
        return b"\xd8\x00"

    def _build_stop_process(self) -> bytes:
        return b"\xd8\x01"

    def _build_pause_process(self) -> bytes:
        return b"\xd8\x02"

    def _build_resume_process(self) -> bytes:
        return b"\xd8\x03"

    def _build_ref_point_0(self) -> bytes:
        return b"\xd8\x12"

    def _build_ref_point_1(self) -> bytes:
        return b"\xd8\x11"

    def _build_ref_point_2(self) -> bytes:
        return b"\xd8\x10"

    def _build_jog_keydown(self, axis: str, direction: int) -> bytes:
        axis_map = {
            ("x", -1): 0x20,
            ("x", 1): 0x21,
            ("y", 1): 0x22,
            ("y", -1): 0x23,
            ("z", 1): 0x24,
            ("z", -1): 0x25,
            ("u", 1): 0x26,
            ("u", -1): 0x27,
        }
        key = (axis.lower(), direction)
        if key not in axis_map:
            raise ValueError(f"Invalid axis/direction: {axis}, {direction}")
        return b"\xd8" + bytes([axis_map[key]])

    def _build_jog_keyup(self, axis: str) -> bytes:
        axis_map = {
            "x": 0x30,
            "y": 0x32,
            "z": 0x34,
            "u": 0x36,
        }
        if axis.lower() not in axis_map:
            raise ValueError(f"Invalid axis: {axis}")
        return b"\xd8" + bytes([axis_map[axis.lower()]])

    def _build_jog_rel_x(self, dx: int) -> bytes:
        """Build D8 command for relative X jog."""
        return b"\xd9\x10\x02" + encode35(dx)

    def _build_jog_rel_y(self, dy: int) -> bytes:
        """Build D8 command for relative Y jog."""
        return b"\xd9\x11\x02" + encode35(dy)

    def _build_power_immediate(
        self, laser: int, power_percent: float
    ) -> bytes:
        power_val = int(power_percent * 163.84)
        laser_map = {1: 0xC7, 2: 0xC0, 3: 0xC2, 4: 0xC3}
        if laser not in laser_map:
            raise ValueError(f"Invalid laser: {laser}")
        return bytes([laser_map[laser]]) + encode14(power_val)

    def _build_power_end(self, laser: int, power_percent: float) -> bytes:
        power_val = int(power_percent * 163.84)
        laser_map = {1: 0xC8, 2: 0xC1, 3: 0xC4, 4: 0xC5}
        if laser not in laser_map:
            raise ValueError(f"Invalid laser: {laser}")
        return bytes([laser_map[laser]]) + encode14(power_val)

    def _build_speed(self, speed_mm_s: float) -> bytes:
        speed_val = int(speed_mm_s * 1000)
        return b"\xc9\x02" + encode35(speed_val)

    def _build_axis_speed(self, speed_mm_s: float) -> bytes:
        speed_val = int(speed_mm_s * 1000)
        return b"\xc9\x03" + encode35(speed_val)

    def _build_end_of_file(self) -> bytes:
        return b"\xd7"

    def _build_keep_alive(self) -> bytes:
        return b"\xce"

    def _build_ack(self) -> bytes:
        return b"\xcc"

    def _build_error(self) -> bytes:
        return b"\xcd"

    async def air_assist_on(self) -> None:
        """Enable air assist."""
        await self.send_command(b"\xca\x13")

    async def air_assist_off(self) -> None:
        """Disable air assist."""
        await self.send_command(b"\xca\x12")

    async def select_layer(self, layer_index: int) -> None:
        """
        Select layer by index (0-15).

        Args:
            layer_index: Layer index (0-15).
        """
        if not 0 <= layer_index <= 15:
            raise ValueError(f"Layer index must be 0-15, got {layer_index}")
        await self.send_command(bytes([0xCA, layer_index]))

    async def send_raw(self, data: bytes) -> None:
        """
        Send raw binary data (already framed/swizzled).

        Args:
            data: Raw binary data to send.
        """
        await self._transport.send(data)

    def _build_read_memory(self, mem_address: int) -> bytes:
        """
        Build command to read from controller memory.

        Args:
            mem_address: Memory address (e.g., 0x0421 for Current X)

        Returns:
            Command bytes to send
        """
        mem_high = (mem_address >> 8) & 0xFF
        mem_low = mem_address & 0xFF
        return bytes([0xDA, 0x00, mem_high, mem_low])

    async def _read_memory(self, mem_address: int) -> None:
        """
        Send a memory read request to the controller.

        Args:
            mem_address: Memory address to read (e.g., 0x0421 for Current X)
        """
        await self.send_command(self._build_read_memory(mem_address))

    async def get_position(self) -> tuple[int, int, int]:
        """
        Request current X, Y, Z position from controller.

        Sends memory read commands for position registers.
        Position values will be returned asynchronously via the
        decoded_received signal.

        Returns:
            Tuple of (x, y, z) in micrometers
            (may be stale until response received)
        """
        await self._read_memory(0x0421)
        await self._read_memory(0x0431)
        await self._read_memory(0x0441)
        return (self.state.x, self.state.y, self.state.z)

    def _build_write_memory(self, mem_address: int, value: int) -> bytes:
        """
        Build command to write to controller memory.

        Args:
            mem_address: Memory address (e.g., 0x0224 for Position Point 0 X)
            value: Value to write in micrometers

        Returns:
            Command bytes to send
        """
        mem_high = (mem_address >> 8) & 0xFF
        mem_low = mem_address & 0xFF
        encoded_value = encode35(value)
        return (
            bytes([0xDA, 0x01, mem_high, mem_low])
            + encoded_value
            + encoded_value
        )

    async def _write_memory(self, mem_address: int, value: int) -> None:
        """
        Write a value to controller memory.

        Args:
            mem_address: Memory address to write
            value: Value to write (will be encoded as 35-bit signed)
        """
        await self.send_command(self._build_write_memory(mem_address, value))

    async def _read_memory_wait(
        self, mem_address: int, timeout: float = 2.0
    ) -> Optional[int]:
        """
        Read a value from controller memory and wait for response.

        Args:
            mem_address: Memory address to read
            timeout: Maximum time to wait for response in seconds

        Returns:
            Decoded value or None if timeout
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_mem_reads[mem_address] = future

        try:
            await self._read_memory(mem_address)
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self._pending_mem_reads.pop(mem_address, None)
            logger.warning(f"Timeout reading memory 0x{mem_address:04X}")
            return None

    async def set_ref_point_offset(
        self, ref_point: str, x_um: int, y_um: int
    ) -> None:
        """
        Set the offset for a reference point.

        Args:
            ref_point: "REF0" or "REF1"
            x_um: X offset in micrometers
            y_um: Y offset in micrometers
        """
        if ref_point not in REF_POINT_OFFSET_ADDRESSES:
            raise ValueError(f"Unknown reference point: {ref_point}")

        x_addr, y_addr = REF_POINT_OFFSET_ADDRESSES[ref_point]
        await self._write_memory(x_addr, x_um)
        await self._write_memory(y_addr, y_um)

    async def get_ref_point_offset(
        self, ref_point: str
    ) -> Optional[tuple[int, int]]:
        """
        Get the offset for a reference point.

        Args:
            ref_point: "REF0" or "REF1"

        Returns:
            Tuple of (x_um, y_um) or None if read failed
        """
        if ref_point not in REF_POINT_OFFSET_ADDRESSES:
            raise ValueError(f"Unknown reference point: {ref_point}")

        x_addr, y_addr = REF_POINT_OFFSET_ADDRESSES[ref_point]
        x_um = await self._read_memory_wait(x_addr)
        y_um = await self._read_memory_wait(y_addr)

        if x_um is None or y_um is None:
            return None
        return (x_um, y_um)

    @property
    def ref_points(self) -> tuple[str, ...]:
        """Return tuple of valid reference point names including MACHINE."""
        return ("MACHINE",) + tuple(REF_POINT_OFFSET_ADDRESSES.keys())

    async def select_ref_point(self, ref_point: str) -> None:
        """
        Select a reference point mode on the controller.

        Args:
            ref_point: "MACHINE", "REF0", or "REF1"
        """
        if ref_point not in REF_POINT_COMMANDS:
            raise ValueError(f"Unknown reference point: {ref_point}")
        await self.send_command(REF_POINT_COMMANDS[ref_point])

    async def get_ref_point_mode(self) -> Optional[str]:
        """
        Get the current reference point mode from the controller.

        Returns:
            "MACHINE", "REF0", "REF1", or None if read failed
        """
        mode = await self._read_memory_wait(0x04F0)
        if mode is None:
            return None

        return REF_POINT_MODE_TO_NAME.get(mode, "UNKNOWN")
