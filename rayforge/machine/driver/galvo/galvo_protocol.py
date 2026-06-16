"""
EzCad2/LMC protocol data structures and helpers.
"""

import struct
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GalvoResponse:
    """Response from a single command (4 x uint16)."""

    v0: int
    v1: int
    v2: int
    v3: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "GalvoResponse":
        if len(data) != 8:
            return cls(0, 0, 0, 0)
        words = struct.unpack("<4H", data[:8])
        return cls(words[0], words[1], words[2], words[3])


@dataclass
class GalvoCommand:
    """
    A raw 12-byte command (6 x uint16 LE).
    """

    cmd: int
    v1: int = 0
    v2: int = 0
    v3: int = 0
    v4: int = 0
    v5: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(
            "<6H", self.cmd, self.v1, self.v2, self.v3, self.v4, self.v5
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "GalvoCommand":
        words = struct.unpack("<6H", data[:12])
        return cls(words[0], words[1], words[2], words[3], words[4], words[5])

    @property
    def is_list_command(self) -> bool:
        return self.cmd >= 0x8000

    @property
    def is_single_command(self) -> bool:
        return self.cmd < 0x8000

    @property
    def name(self) -> str:
        from .galvo_consts import LIST_COMMAND_NAMES, SINGLE_COMMAND_NAMES

        if self.is_list_command:
            return LIST_COMMAND_NAMES.get(self.cmd, f"0x{self.cmd:04X}")
        return SINGLE_COMMAND_NAMES.get(self.cmd, f"0x{self.cmd:04X}")


def build_nop() -> bytes:
    """Build a NOP command (filler for empty list slots)."""
    return struct.pack("<6H", 0x8002, 0, 0, 0, 0, 0)


def build_list_packet(commands: bytes) -> bytes:
    """Build a 3072-byte list packet from concatenated command bytes."""
    if len(commands) > 0xC00:
        commands = commands[:0xC00]
    nop = build_nop()
    result = bytearray(commands)
    result.extend(nop * ((0xC00 - len(commands)) // 12))
    return bytes(result)


def parse_response(data: bytes) -> GalvoResponse:
    """Parse an 8-byte response into a GalvoResponse."""
    return GalvoResponse.from_bytes(data)


def _bytes_to_words(data: bytes) -> Tuple[int, int, int, int]:
    """Convert 8 bytes to 4 uint16 LE words."""
    return struct.unpack("<4H", data[:8])
