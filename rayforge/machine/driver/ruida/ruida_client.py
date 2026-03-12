"""
Ruida Client Protocol - Client-side command generation.

Handles generation of commands to send to a Ruida laser controller
and parsing of responses.
"""

import logging
from typing import Optional

from .ruida_protocol import RuidaResponse, RuidaState
from .ruida_util import (
    encode14,
    encode35,
)

logger = logging.getLogger(__name__)


class RuidaClient:
    """
    Ruida client-side protocol handler.

    Generates commands to send to a Ruida controller and parses responses.
    Use this for implementing a Ruida driver/client.
    """

    def __init__(self, state: Optional[RuidaState] = None):
        self.state = state or RuidaState()

    def parse_response(self, data: bytes) -> RuidaResponse:
        """Parse a response from the controller."""
        return RuidaResponse.from_bytes(data)

    def build_move_abs(self, x: int, y: int) -> bytes:
        """Build absolute move command (0x88)."""
        return b"\x88" + encode35(x) + encode35(y)

    def build_move_rel(self, dx: int, dy: int) -> bytes:
        """Build relative move command (0x89)."""
        return b"\x89" + encode14(dx) + encode14(dy)

    def build_cut_abs(self, x: int, y: int) -> bytes:
        """Build absolute cut command (0xA8)."""
        return b"\xa8" + encode35(x) + encode35(y)

    def build_cut_rel(self, dx: int, dy: int) -> bytes:
        """Build relative cut command (0xA9)."""
        return b"\xa9" + encode14(dx) + encode14(dy)

    def build_move_rel_x(self, dx: int) -> bytes:
        """Build relative X move command (0x8A)."""
        return b"\x8a" + encode14(dx)

    def build_move_rel_y(self, dy: int) -> bytes:
        """Build relative Y move command (0x8B)."""
        return b"\x8b" + encode14(dy)

    def build_cut_rel_x(self, dx: int) -> bytes:
        """Build relative X cut command (0xAA)."""
        return b"\xaa" + encode14(dx)

    def build_cut_rel_y(self, dy: int) -> bytes:
        """Build relative Y cut command (0xAB)."""
        return b"\xab" + encode14(dy)

    def build_rapid_move_xy(
        self, x: int, y: int, origin: bool = False, light: bool = False
    ) -> bytes:
        """Build rapid XY move command (0xD9)."""
        opts = self._build_move_opts(origin, light)
        return b"\xd9\x10" + bytes([opts]) + encode35(x) + encode35(y)

    def build_rapid_move_axis(
        self,
        axis: int,
        coord: int,
        origin: bool = False,
        light: bool = False,
    ) -> bytes:
        """Build rapid single-axis move command (0xD9)."""
        opts = self._build_move_opts(origin, light)
        return b"\xd9" + bytes([axis & 0x0F]) + bytes([opts]) + encode35(coord)

    def _build_move_opts(self, origin: bool, light: bool) -> int:
        """Build move options byte."""
        if origin and light:
            return 0x01
        elif origin:
            return 0x00
        elif light:
            return 0x03
        return 0x02

    def build_home_xy(self) -> bytes:
        """Build home XY command."""
        return b"\xd8\x2a"

    def build_home_z(self) -> bytes:
        """Build home Z command."""
        return b"\xd8\x2c"

    def build_home_u(self) -> bytes:
        """Build home U command."""
        return b"\xd8\x2d"

    def build_start_process(self) -> bytes:
        """Build start process command."""
        return b"\xd8\x00"

    def build_stop_process(self) -> bytes:
        """Build stop process command."""
        return b"\xd8\x01"

    def build_pause_process(self) -> bytes:
        """Build pause process command."""
        return b"\xd8\x02"

    def build_resume_process(self) -> bytes:
        """Build resume process command."""
        return b"\xd8\x03"

    def build_ref_point_0(self) -> bytes:
        """Build set ref point 0 command."""
        return b"\xd8\x12"

    def build_ref_point_1(self) -> bytes:
        """Build set ref point 1 command."""
        return b"\xd8\x11"

    def build_ref_point_2(self) -> bytes:
        """Build set ref point 2 command."""
        return b"\xd8\x10"

    def build_jog_keydown(self, axis: str, direction: int) -> bytes:
        """
        Build jog keydown command.

        Args:
            axis: One of 'x', 'y', 'z', 'u'
            direction: 1 for positive, -1 for negative
        """
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

    def build_jog_keyup(self, axis: str) -> bytes:
        """Build jog keyup command."""
        axis_map = {
            "x": 0x30,
            "y": 0x32,
            "z": 0x34,
            "u": 0x36,
        }
        if axis.lower() not in axis_map:
            raise ValueError(f"Invalid axis: {axis}")
        return b"\xd8" + bytes([axis_map[axis.lower()]])

    def build_power_immediate(self, laser: int, power_percent: float) -> bytes:
        """Build immediate power command."""
        power_val = int(power_percent * 163.84)
        laser_map = {1: 0xC7, 2: 0xC0, 3: 0xC2, 4: 0xC3}
        if laser not in laser_map:
            raise ValueError(f"Invalid laser: {laser}")
        return bytes([laser_map[laser]]) + encode14(power_val)

    def build_power_end(self, laser: int, power_percent: float) -> bytes:
        """Build end power command."""
        power_val = int(power_percent * 163.84)
        laser_map = {1: 0xC8, 2: 0xC1, 3: 0xC4, 4: 0xC5}
        if laser not in laser_map:
            raise ValueError(f"Invalid laser: {laser}")
        return bytes([laser_map[laser]]) + encode14(power_val)

    def build_speed(self, speed_mm_s: float) -> bytes:
        """Build speed command."""
        speed_val = int(speed_mm_s * 1000)
        return b"\xc9\x02" + encode35(speed_val)

    def build_axis_speed(self, speed_mm_s: float) -> bytes:
        """Build axis speed command."""
        speed_val = int(speed_mm_s * 1000)
        return b"\xc9\x03" + encode35(speed_val)

    def build_end_of_file(self) -> bytes:
        """Build end of file command."""
        return b"\xd7"

    def build_keep_alive(self) -> bytes:
        """Build keep alive command."""
        return b"\xce"

    def build_ack(self) -> bytes:
        """Build ACK response."""
        return b"\xcc"

    def build_error(self) -> bytes:
        """Build error response."""
        return b"\xcd"
