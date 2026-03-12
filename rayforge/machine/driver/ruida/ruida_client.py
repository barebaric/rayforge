"""
Ruida Client Protocol - Client-side command generation and sending.

Handles generation of commands to send to a Ruida laser controller,
sending them via transport, and parsing of responses.
"""

import logging
from typing import Optional, TYPE_CHECKING

from .ruida_protocol import RuidaResponse, RuidaState
from .ruida_util import encode14, encode35

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
    ):
        self._transport = transport
        self.state = state or RuidaState()

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    async def connect(self) -> None:
        """Establish connection to the Ruida controller."""
        await self._transport.connect()

    async def disconnect(self) -> None:
        """Close connection to the Ruida controller."""
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
        """Set reference point 2 (current position)."""
        await self.send_command(self._build_ref_point_2())

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
        return b"\xd9\x10" + bytes([opts]) + encode35(x) + encode35(y)

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
