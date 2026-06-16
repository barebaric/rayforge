"""
Connection interface for EzCad2/LMC controllers.

Defines a Protocol (structural typing) for USB-like connections
so both MockGalvoConnection and GalvoUSBConnection are interchangeable.
"""

from typing import Optional, Protocol


class GalvoConnection(Protocol):
    """Protocol for EzCad2 USB-like connections."""

    @property
    def is_connected(self) -> bool: ...

    def open(self, index: int = 0) -> int: ...

    def close(self, index: int = 0) -> None: ...

    def write(
        self, index: int = 0, packet: Optional[bytes] = None
    ) -> None: ...

    def read(self, index: int = 0) -> bytes: ...
