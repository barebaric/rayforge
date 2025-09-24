# rayforge/importer/ruida/parser.py

import struct
from typing import Dict, Tuple, Callable, Union

from .job import RuidaJob, RuidaLayer, RuidaCommand

# Conversion factor from micrometers (device units) to millimeters.
UM_PER_MM = 1000.0

# A type alias for a command handler, defined at the module level for
# correct type checking. A handler is a tuple of:
# (payload_length, handler_function).
HandlerType = Tuple[int, Callable[[RuidaJob, bytes], None]]


def _unscramble(byte_val: int) -> int:
    """
    Decodes a single byte from a Ruida file stream. This is a required
    proprietary decryption step.
    """
    # The scrambling algorithm is a simple byte-level substitution cipher.
    # This function reverses it.
    temp_val = byte_val - 1
    if temp_val < 0:
        temp_val += 0x100
    temp_val ^= 0x88
    msb = temp_val & 0x80
    lsb = temp_val & 1
    temp_val = temp_val - msb - lsb
    temp_val |= lsb << 7
    temp_val |= msb >> 7
    return temp_val


class RuidaParser:
    """
    Parses a Ruida .rd file content into a structured RuidaJob object.
    It handles the proprietary unscrambling and decodes the binary
    command stream.
    """

    def __init__(self, data: bytes):
        """
        Initializes the parser with the raw .rd file data.

        Args:
            data: The byte content of the .rd file.
        """
        if data.startswith(b"RDWORKV"):
            # Standard .rd files have a 10-byte header to skip.
            raw_data = data[10:]
        else:
            raw_data = data

        self.data = bytes([_unscramble(b) for b in raw_data])
        self.index = 0
        self.current_color = 0
        self.x, self.y = 0.0, 0.0

        # The command table maps a command byte to either a handler
        # or a nested dictionary of sub-command bytes to handlers.
        self.COMMAND_TABLE: Dict[
            int, Union[HandlerType, Dict[int, HandlerType]]
        ] = self._build_command_table()

    def parse(self) -> RuidaJob:
        """
        Parses the entire data buffer and returns a complete RuidaJob.
        """
        job = RuidaJob()
        while self.index < len(self.data):
            self._process_one_command(job)
        return job

    def _process_one_command(self, job: RuidaJob) -> None:
        """
        Reads, decodes, and handles a single command from the data stream.
        """
        command_byte = self.data[self.index]
        handler_entry = self.COMMAND_TABLE.get(command_byte)

        if handler_entry is None:
            self.index += 1
            return

        self.index += 1
        handler = None
        length = 0

        if isinstance(handler_entry, dict):
            if self.index >= len(self.data):
                return
            subcommand_byte = self.data[self.index]
            found_handler = handler_entry.get(subcommand_byte)
            if found_handler:
                self.index += 1
                length, handler = found_handler
        else:
            length, handler = handler_entry

        if handler:
            if self.index + length > len(self.data):
                return  # Avoid reading past the end of the buffer
            payload = self.data[self.index:self.index + length]
            self.index += length
            handler(job, payload)

    def _handle_set_color(self, job: RuidaJob, payload: bytes) -> None:
        self.current_color = payload[0]

    def _handle_set_speed(self, job: RuidaJob, payload: bytes) -> None:
        color_index = payload[0]
        speed = struct.unpack("<f", payload[1:5])[0]
        self._ensure_layer(job, color_index).speed = speed

    def _handle_set_power(self, job: RuidaJob, payload: bytes) -> None:
        color_index = payload[0]
        # Power is a short, scaled by 10
        power = struct.unpack("<H", payload[1:3])[0] / 10.0
        self._ensure_layer(job, color_index).power = power

    def _handle_move_abs(self, job: RuidaJob, payload: bytes) -> None:
        self.x, self.y = self._decode_abs_coords(payload)
        cmd = RuidaCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_abs(self, job: RuidaJob, payload: bytes) -> None:
        self.x, self.y = self._decode_abs_coords(payload)
        cmd = RuidaCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_xy(self, job: RuidaJob, payload: bytes) -> None:
        dx, dy = self._decode_rel_coords(payload)
        self.x += dx
        self.y += dy
        cmd = RuidaCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_xy(self, job: RuidaJob, payload: bytes) -> None:
        dx, dy = self._decode_rel_coords(payload)
        self.x += dx
        self.y += dy
        cmd = RuidaCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_x(self, job: RuidaJob, payload: bytes) -> None:
        dx = self._decode_rel_coord(payload)
        self.x += dx
        cmd = RuidaCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_x(self, job: RuidaJob, payload: bytes) -> None:
        dx = self._decode_rel_coord(payload)
        self.x += dx
        cmd = RuidaCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_y(self, job: RuidaJob, payload: bytes) -> None:
        dy = self._decode_rel_coord(payload)
        self.y += dy
        cmd = RuidaCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_y(self, job: RuidaJob, payload: bytes) -> None:
        dy = self._decode_rel_coord(payload)
        self.y += dy
        cmd = RuidaCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_end(self, job: RuidaJob, payload: bytes) -> None:
        job.commands.append(RuidaCommand("End"))

    def _build_command_table(self):
        """Constructs the mapping from command bytes to handlers."""
        return {
            0x88: (10, self._handle_move_abs),
            0x89: (4, self._handle_move_rel_xy),
            0x8A: (2, self._handle_move_rel_x),
            0x8B: (2, self._handle_move_rel_y),
            0xA8: (10, self._handle_cut_abs),
            0xA9: (4, self._handle_cut_rel_xy),
            0xAA: (2, self._handle_cut_rel_x),
            0xAB: (2, self._handle_cut_rel_y),
            0xD7: (0, self._handle_end),
            # Nested commands
            0xCA: {0x06: (5, self._handle_set_color)},
            0xC9: {0x04: (5, self._handle_set_speed)},
            0xC6: {0x32: (3, self._handle_set_power)},
        }

    def _ensure_layer(self, job: RuidaJob, color: int) -> RuidaLayer:
        """
        Gets the layer for a given color, creating it if it doesn't exist.
        """
        if color not in job.layers:
            job.layers[color] = RuidaLayer(color_index=color, speed=0, power=0)
        return job.layers[color]

    def _decode_abs_coords(self, payload: bytes) -> Tuple[float, float]:
        """Decodes a 10-byte absolute coordinate pair."""

        def _decode_val(val_bytes: bytes) -> int:
            """
            Decodes a 5-byte, big-endian, variable-length integer into a
            35-bit signed int.
            """
            val = 0
            for i in range(5):
                val <<= 7
                val |= val_bytes[i] & 0x7F

            # The value is 35 bits (5 * 7). The sign bit is the 35th bit.
            # Sign bit mask: 1 << 34 = 0x400000000
            # Range of a 35-bit number: 1 << 35 = 0x800000000
            if val & 0x400000000:
                val -= 0x800000000
            return val

        x_um = _decode_val(payload[:5])
        y_um = _decode_val(payload[5:10])
        return x_um / UM_PER_MM, y_um / UM_PER_MM

    def _decode_rel_coord(self, payload: bytes) -> float:
        """
        Decodes a 2-byte, big-endian, variable-length relative coordinate.
        Each byte contributes 7 bits of data, forming a 14-bit signed integer.
        """
        # Big-Endian: payload[0] is MSB, payload[1] is LSB.
        val = ((payload[0] & 0x7F) << 7) | (payload[1] & 0x7F)

        # Check the 14th bit (the sign bit) and apply two's complement.
        if val & 0x2000:  # 0x2000 = 1 << 13
            val -= 0x4000  # 0x4000 = 1 << 14
        return val / UM_PER_MM

    def _decode_rel_coords(self, payload: bytes) -> Tuple[float, float]:
        """Decodes a 4-byte relative coordinate pair."""
        dx = self._decode_rel_coord(payload[:2])
        dy = self._decode_rel_coord(payload[2:4])
        return dx, dy
