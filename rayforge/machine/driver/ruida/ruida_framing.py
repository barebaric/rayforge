"""
Layer 2 (Data Link/Framing) for Ruida protocol.

Handles packet boundaries, checksum validation, and accumulation
for streaming transports like serial.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Tuple

from .ruida_maps import (
    DA_4_BYTE_RESPONSE_SUBCOMMANDS,
    DA_VARIABLE_4_BYTE_SUBCOMMANDS,
)
from .ruida_util import calculate_checksum, decode14

logger = logging.getLogger(__name__)


class PacketStatus(Enum):
    VALID = "valid"
    INVALID_CHECKSUM = "invalid_checksum"
    INCOMPLETE = "incomplete"


@dataclass
class FramedPacket:
    status: PacketStatus
    payload: bytes
    raw: bytes
    expected_checksum: int = 0
    actual_checksum: int = 0


class PacketFramer:
    """
    Handles framing of Ruida protocol packets.

    Ruida packets have a 2-byte checksum prefix followed by payload:
    [checksum_high][checksum_low][payload...]

    The checksum is the sum of all payload bytes (mod 65536).
    """

    def __init__(self):
        self._buffer: bytes = b""

    def reset(self) -> None:
        """Clear the internal buffer."""
        self._buffer = b""

    def add_data(self, data: bytes) -> None:
        """Add incoming data to the buffer."""
        self._buffer += data

    def extract_packets(self) -> Generator[FramedPacket, None, None]:
        """
        Extract all complete packets from the buffer.

        Yields FramedPacket objects. Consumes extracted data from buffer.
        """
        while len(self._buffer) >= 2:
            checksum_received = (self._buffer[0] << 8) | self._buffer[1]
            payload = self._buffer[2:]

            estimated_len = estimate_packet_length(payload)
            if estimated_len < 0:
                break
            if len(payload) < estimated_len:
                break

            packet_payload = payload[:estimated_len]
            checksum_calculated = calculate_checksum(packet_payload)

            raw_packet = self._buffer[: 2 + estimated_len]
            self._buffer = self._buffer[2 + estimated_len :]

            if checksum_received != checksum_calculated:
                yield FramedPacket(
                    status=PacketStatus.INVALID_CHECKSUM,
                    payload=packet_payload,
                    raw=raw_packet,
                    expected_checksum=checksum_received,
                    actual_checksum=checksum_calculated,
                )
            else:
                yield FramedPacket(
                    status=PacketStatus.VALID,
                    payload=packet_payload,
                    raw=raw_packet,
                    expected_checksum=checksum_received,
                    actual_checksum=checksum_calculated,
                )

    @property
    def buffer_size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)


def frame_packet(payload: bytes) -> bytes:
    """
    Create a framed packet with checksum prefix.

    Args:
        payload: The payload bytes to frame

    Returns:
        Complete packet with 2-byte checksum prefix + payload
    """
    checksum = calculate_checksum(payload)
    return bytes([checksum >> 8, checksum & 0xFF]) + payload


def validate_packet(data: bytes) -> Tuple[bool, bytes, int, int]:
    """
    Validate a complete packet and extract payload.

    Args:
        data: Complete packet with checksum prefix

    Returns:
        Tuple of (is_valid, payload, expected_checksum, actual_checksum)
    """
    if len(data) < 2:
        return False, b"", 0, 0

    checksum_received = (data[0] << 8) | data[1]
    payload = data[2:]
    checksum_calculated = calculate_checksum(payload)

    return (
        checksum_received == checksum_calculated,
        payload,
        checksum_received,
        checksum_calculated,
    )


def estimate_packet_length(payload: bytes) -> int:
    """
    Estimate the expected packet length from the payload.

    Args:
        payload: Payload bytes (without checksum prefix)

    Returns:
        Expected payload length, or -1 if unknown/insufficient data
    """
    if len(payload) < 1:
        return -1

    cmd = payload[0]

    if cmd == 0xCC or cmd == 0xCD or cmd == 0xCE:
        return 1

    if cmd == 0xD0:
        if len(payload) < 2:
            return -1
        return 2

    if cmd == 0xD7:
        return 1

    if cmd == 0xD8:
        if len(payload) < 2:
            return -1
        return 2

    if cmd == 0xD9:
        if len(payload) < 2:
            return -1
        return 2

    if cmd == 0xDA:
        if len(payload) < 4:
            return -1
        sub = payload[1]
        if sub == 0x00:
            return 4
        if sub == 0x01:
            return 4
        if sub == 0x04:
            if len(payload) < 7:
                return -1
            return 7
        if sub == 0x05:
            if len(payload) < 8:
                return -1
            return 8
        if sub == 0x06:
            if len(payload) < 6:
                return -1
            return 6
        if sub == 0x07:
            if len(payload) < 7:
                return -1
            return 7
        if sub == 0x10:
            if len(payload) < 7:
                return -1
            extra = decode14(payload[4:])
            return 7 + extra
        if sub in DA_4_BYTE_RESPONSE_SUBCOMMANDS:
            return 4
        if sub == 0x54:
            if len(payload) < 6:
                return -1
            return 6
        if sub == 0x55:
            if len(payload) < 5:
                return -1
            return 5
        if sub in DA_VARIABLE_4_BYTE_SUBCOMMANDS:
            return 4
        return 4

    if cmd == 0xA5:
        if len(payload) < 3:
            return -1
        return 3

    if cmd == 0xA7:
        return 2

    if cmd in (0xC3, 0xC6, 0xC7):
        if len(payload) < 5:
            return -1
        extra = decode14(payload[3:])
        return 5 + extra

    if cmd == 0xCA:
        if len(payload) < 11:
            return -1
        return 11

    if cmd in (0xE5, 0xE7, 0xE8):
        if len(payload) < 2:
            return -1
        return 2

    if cmd == 0x88:
        if len(payload) < 7:
            return -1
        return 7

    if cmd == 0x89:
        if len(payload) < 11:
            return -1
        return 11

    return len(payload)
