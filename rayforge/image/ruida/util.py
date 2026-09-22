"""
Ruida .rd file encoding/decoding utilities.

Based on:
- https://edutechwiki.unige.ch/en/Ruida
- https://github.com/meerk40t/meerk40t/tree/main/meerk40t/ruida
- https://github.com/StevenIsaacs/ruida-protocol-analyzer
"""

UM_PER_MM = 1000.0


def unswizzle_byte(b: int, magic: int = 0x88) -> int:
    """Unswizzle a single byte after reception."""
    b = (b - 1) & 0xFF
    b ^= magic
    b ^= (b >> 7) & 0xFF
    b ^= (b << 7) & 0xFF
    b ^= (b >> 7) & 0xFF
    return b


def encode14(v: int) -> bytes:
    """Encode a 14-bit value."""
    v = int(v) & 0x3FFF
    return bytes([(v >> 7) & 0x7F, v & 0x7F])


def encode35(v: int) -> bytes:
    """Encode a signed 35-bit coordinate as 5 bytes."""
    v = int(v) & 0x7FFFFFFFF
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


def decode35(data: bytes) -> int:
    """Decode a signed 35-bit coordinate from 5 bytes."""
    val = (
        ((data[0] & 0x7F) << 28)
        | ((data[1] & 0x7F) << 21)
        | ((data[2] & 0x7F) << 14)
        | ((data[3] & 0x7F) << 7)
        | (data[4] & 0x7F)
    )
    if val & 0x400000000:
        val -= 0x800000000
    return val


def decode_abs_coords(data: bytes) -> tuple[float, float]:
    """
    Decode absolute X,Y coordinates from 10 bytes.
    Returns coordinates in millimeters.
    """
    x_um = decode35(data[:5])
    y_um = decode35(data[5:10])
    return x_um / UM_PER_MM, y_um / UM_PER_MM


def decode_rel_coords(data: bytes) -> tuple[float, float]:
    """
    Decode relative X,Y coordinates from 4 bytes.
    Returns coordinates in millimeters.
    """
    dx_um = decode14(data[:2])
    dy_um = decode14(data[2:4])
    return dx_um / UM_PER_MM, dy_um / UM_PER_MM
