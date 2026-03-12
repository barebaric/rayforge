"""
Ruida protocol utility functions for encoding, decoding, and swizzling.

Based on:
- https://edutechwiki.unige.ch/en/Ruida
- https://github.com/meerk40t/meerk40t/tree/main/meerk40t/ruida
- https://github.com/StevenIsaacs/ruida-protocol-analyzer
"""

from typing import Tuple


UM_PER_MM = 1000.0


def swizzle_byte(b: int, magic: int = 0x88) -> int:
    """Swizzle a single byte for transmission."""
    b ^= (b >> 7) & 0xFF
    b ^= (b << 7) & 0xFF
    b ^= (b >> 7) & 0xFF
    b ^= magic
    b = (b + 1) & 0xFF
    return b


def unswizzle_byte(b: int, magic: int = 0x88) -> int:
    """Unswizzle a single byte after reception."""
    b = (b - 1) & 0xFF
    b ^= magic
    b ^= (b >> 7) & 0xFF
    b ^= (b << 7) & 0xFF
    b ^= (b >> 7) & 0xFF
    return b


def build_swizzle_lut(magic: int) -> Tuple[bytes, bytes]:
    """Build lookup tables for swizzling and unswizzling."""
    swizzle = bytes([swizzle_byte(i, magic) for i in range(256)])
    unswizzle = bytes([unswizzle_byte(i, magic) for i in range(256)])
    return swizzle, unswizzle


def encode14(v: int) -> bytes:
    """Encode a 14-bit value."""
    v = int(v) & 0x3FFF
    return bytes([(v >> 7) & 0x7F, v & 0x7F])


def encode32(v: int) -> bytes:
    """Encode a 32-bit value as 5 bytes (35-bit encoding)."""
    v = int(v) & 0xFFFFFFFF
    return bytes(
        [
            (v >> 28) & 0x7F,
            (v >> 21) & 0x7F,
            (v >> 14) & 0x7F,
            (v >> 7) & 0x7F,
            v & 0x7F,
        ]
    )


def decode14(data: bytes) -> int:
    """Decode a 14-bit value from 2 bytes."""
    val = ((data[0] & 0x7F) << 7) | (data[1] & 0x7F)
    if val & 0x2000:
        val -= 0x4000
    return val


def decodeu14(data: bytes) -> int:
    """Decode an unsigned 14-bit value from 2 bytes."""
    return ((data[0] & 0x7F) << 7) | (data[1] & 0x7F)


def decode32(data: bytes) -> int:
    """Decode a 32-bit value from 5 bytes."""
    val = (
        ((data[0] & 0x7F) << 28)
        | ((data[1] & 0x7F) << 21)
        | ((data[2] & 0x7F) << 14)
        | ((data[3] & 0x7F) << 7)
        | (data[4] & 0x7F)
    )
    if val & 0x40000000:
        val -= 0x80000000
    return val


def decodeu35(data: bytes) -> int:
    """Decode an unsigned 35-bit value from 5 bytes."""
    return (
        ((data[0] & 0x7F) << 28)
        | ((data[1] & 0x7F) << 21)
        | ((data[2] & 0x7F) << 14)
        | ((data[3] & 0x7F) << 7)
        | (data[4] & 0x7F)
    )


def parse_mem(data: bytes) -> int:
    """Parse memory address from 2 bytes (big-endian)."""
    return (data[0] << 8) | data[1]


def calculate_checksum(data: bytes) -> int:
    """Calculate 16-bit checksum (sum of all bytes)."""
    return sum(data) & 0xFFFF


def decode_abs_coords(data: bytes) -> Tuple[float, float]:
    """
    Decode absolute X,Y coordinates from 10 bytes.
    Returns coordinates in millimeters.
    """
    x_um = decode32(data[:5])
    y_um = decode32(data[5:10])
    return x_um / UM_PER_MM, y_um / UM_PER_MM


def decode_rel_coords(data: bytes) -> Tuple[float, float]:
    """
    Decode relative X,Y coordinates from 4 bytes.
    Returns coordinates in millimeters.
    """
    dx_um = decode14(data[:2])
    dy_um = decode14(data[2:4])
    return dx_um / UM_PER_MM, dy_um / UM_PER_MM
