"""
Ruida codec for swizzle encoding/decoding with magic key management.

The magic key determines the swizzle transformation. It can be
detected from certain packets (card ID queries) or set explicitly.
"""

import logging
from typing import Dict, Optional

from .ruida_maps import CARD_ID_TO_MAGIC
from .ruida_util import build_swizzle_lut, parse_mem

logger = logging.getLogger(__name__)


class RuidaCodec:
    """
    Handles swizzle encoding/decoding with magic key management.

    The magic key determines the swizzle transformation. It can be
    detected from certain packets (card ID queries) or set explicitly.
    """

    def __init__(self, magic: int = 0x88):
        self.magic = magic
        self._swizzle_lut, self._unswizzle_lut = build_swizzle_lut(magic)
        self._magic_keys = self._build_magic_keys()

    def _build_magic_keys(self) -> Dict[bytes, int]:
        """Build lookup table for magic key detection from 4-byte packets."""
        keys = {}
        for g in range(256):
            swiz, _ = build_swizzle_lut(g)
            keys[bytes([swiz[b] for b in b"\xda\x00\x05\x7e"])] = g
        return keys

    def set_magic(self, magic: int) -> bool:
        """
        Set the magic key for swizzle encoding.

        Returns True if magic changed.
        """
        if magic != self.magic:
            self.magic = magic
            self._swizzle_lut, self._unswizzle_lut = build_swizzle_lut(magic)
            logger.info(f"Magic key changed to 0x{magic:02X}")
            return True
        return False

    def swizzle(self, data: bytes) -> bytes:
        """Encode data for transmission."""
        return bytes([self._swizzle_lut[b] for b in data])

    def unswizzle(self, data: bytes) -> bytes:
        """Decode received data."""
        return bytes([self._unswizzle_lut[b] for b in data])

    def detect_magic_from_payload(self, payload: bytes) -> Optional[int]:
        """
        Try to detect magic key from a swizzled payload.

        Returns detected magic or None.
        """
        if len(payload) == 4:
            return self._magic_keys.get(payload)
        return None

    def detect_magic_from_mem_request(
        self, unswizzled: bytes
    ) -> Optional[int]:
        """
        Detect magic key from DA memory read requests.

        This is a secondary detection mechanism for certain edge cases
        where the memory address itself encodes card ID information.

        Returns detected magic or None.
        """
        if len(unswizzled) >= 4 and unswizzled[0] == 0xDA:
            if unswizzled[1] == 0x00:
                mem = parse_mem(unswizzled[2:4])
                if mem in CARD_ID_TO_MAGIC:
                    return CARD_ID_TO_MAGIC[mem]
        return None
