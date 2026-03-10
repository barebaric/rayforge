"""
Layer 4 (Application/Protocol) shared structures for Ruida protocol.

Contains shared state model and command/response definitions used by both
server and client implementations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .ruida_maps import DEFAULT_MEMORY_MAP, DYNAMIC_MEMORY_KEYS


class RuidaState:
    """
    Machine state for Ruida controller.
    """

    CARD_ID = 0x65106510
    DEFAULT_BED_X = 320000
    DEFAULT_BED_Y = 220000

    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.u = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.bed_x = self.DEFAULT_BED_X
        self.bed_y = self.DEFAULT_BED_Y
        self.z_range = 0
        self.u_range = 0
        self.machine_status = 22
        self.filename: Optional[str] = None
        self.program_mode = False
        self.file_checksum = 0
        self.file_checksum_accumulator = 0
        self.checksum_enabled = True
        self.jog_speed = 10000
        self.jog_active: Dict[str, int] = {"x": 0, "y": 0, "z": 0, "u": 0}
        self.memory_values: Dict[int, int] = {}
        self.ref_point_mode = 0

    def mem_lookup(self, mem: int) -> Tuple[str, int]:
        """Look up memory address and return (name, value)."""
        if mem in self.memory_values:
            name = "Written Value"
            if mem in DEFAULT_MEMORY_MAP:
                name = DEFAULT_MEMORY_MAP[mem][0]
            elif mem in DYNAMIC_MEMORY_KEYS:
                name = DYNAMIC_MEMORY_KEYS[mem][0]
            return name, self.memory_values[mem]

        if mem == 0x057E:
            return "Card ID", self.CARD_ID

        if mem in DYNAMIC_MEMORY_KEYS:
            entry = DYNAMIC_MEMORY_KEYS[mem]
            name = entry[0]
            attr = entry[1]
            value = getattr(self, attr, 0)
            return name, value

        if mem in DEFAULT_MEMORY_MAP:
            entry = DEFAULT_MEMORY_MAP[mem]
            return entry[0], entry[1]

        return f"Unknown mem 0x{mem:04X}", 0


@dataclass
class RuidaCommand:
    """
    Represents a parsed Ruida command.

    Attributes:
        cmd: Primary command byte
        subcmd: Secondary command byte (if applicable)
        data: Raw command data
        name: Human-readable command name
        params: Parsed parameters
    """

    cmd: int
    subcmd: Optional[int] = None
    data: bytes = b""
    name: str = ""
    params: Optional[dict] = None

    @property
    def length(self) -> int:
        """Return the length of the raw command data."""
        return len(self.data)


@dataclass
class RuidaResponse:
    """
    Represents a Ruida response.

    Attributes:
        data: Raw response bytes
        success: Whether the command was successful
        ack: True for ACK (0xCC), False for error (0xCD)
    """

    data: bytes = b""
    success: bool = True

    @property
    def ack(self) -> bool:
        return self.success

    @classmethod
    def ack_response(cls) -> "RuidaResponse":
        """Create a standard ACK response."""
        return cls(data=b"\xcc", success=True)

    @classmethod
    def error_response(cls) -> "RuidaResponse":
        """Create a standard error response."""
        return cls(data=b"\xcd", success=False)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RuidaResponse":
        """Create response from raw bytes."""
        if not data:
            return cls.ack_response()
        if data[0] == 0xCD:
            return cls(data=data, success=False)
        return cls(data=data, success=True)
