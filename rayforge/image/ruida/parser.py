# rayforge/importer/ruida/parser.py

import struct
from typing import Dict, Tuple, Callable, Union

from ...machine.driver.ruida.ruida_util import (
    UM_PER_MM,
    unswizzle_byte,
    decode14,
    decode_abs_coords,
    decode_rel_coords,
)
from .job import RuidaJob, RuidaLayer, RuidaGeoCommand

# A type alias for a command handler, defined at the module level for
# correct type checking. A handler is a tuple of:
# (payload_length, handler_function).
HandlerType = Tuple[int, Callable[[RuidaJob, bytes], None]]


class RuidaParseError(Exception):
    """Custom exception for errors during Ruida file parsing."""

    pass


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

        self.data = bytes([unswizzle_byte(b, magic=0x88) for b in raw_data])
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
                raise RuidaParseError(
                    f"Unexpected end of file after command "
                    f"0x{command_byte:02X}."
                )
            subcommand_byte = self.data[self.index]
            found_handler = handler_entry.get(subcommand_byte)
            if found_handler:
                self.index += 1
                length, handler = found_handler
        else:
            length, handler = handler_entry

        if handler:
            if self.index + length > len(self.data):
                raise RuidaParseError(
                    f"Incomplete payload for command 0x{command_byte:02X}. "
                    f"Expected {length} bytes, "
                    f"found {len(self.data) - self.index}."
                )
            payload = self.data[self.index : self.index + length]
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
        self.x, self.y = decode_abs_coords(payload)
        cmd = RuidaGeoCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_abs(self, job: RuidaJob, payload: bytes) -> None:
        self.x, self.y = decode_abs_coords(payload)
        cmd = RuidaGeoCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_xy(self, job: RuidaJob, payload: bytes) -> None:
        dx, dy = decode_rel_coords(payload)
        self.x += dx
        self.y += dy
        cmd = RuidaGeoCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_xy(self, job: RuidaJob, payload: bytes) -> None:
        dx, dy = decode_rel_coords(payload)
        self.x += dx
        self.y += dy
        cmd = RuidaGeoCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_x(self, job: RuidaJob, payload: bytes) -> None:
        dx = decode14(payload) / UM_PER_MM
        self.x += dx
        cmd = RuidaGeoCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_x(self, job: RuidaJob, payload: bytes) -> None:
        dx = decode14(payload) / UM_PER_MM
        self.x += dx
        cmd = RuidaGeoCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_move_rel_y(self, job: RuidaJob, payload: bytes) -> None:
        dy = decode14(payload) / UM_PER_MM
        self.y += dy
        cmd = RuidaGeoCommand("Move_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_cut_rel_y(self, job: RuidaJob, payload: bytes) -> None:
        dy = decode14(payload) / UM_PER_MM
        self.y += dy
        cmd = RuidaGeoCommand("Cut_Abs", [self.x, self.y], self.current_color)
        job.commands.append(cmd)

    def _handle_end(self, job: RuidaJob, payload: bytes) -> None:
        job.commands.append(RuidaGeoCommand("End"))

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
