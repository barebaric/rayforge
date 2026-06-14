"""
Mock connection for Galvo/EzCad2 controllers.

Provides a fake USB connection for testing purposes.
Records sent commands and returns configurable responses.
"""

import random
import struct
from typing import Callable, Optional

from .galvo_consts import (
    COMMAND_SIZE,
    LIST_BUFFER_SIZE,
    GetSerialNo,
    GetVersion,
    ReadPort,
    SINGLE_COMMAND_NAMES,
    LIST_COMMAND_NAMES,
)
from .galvo_protocol import GalvoCommand


class MockGalvoConnection:
    """
    A mock USB connection for EzCad2/LMC controllers.

    Records all sent commands for later verification and returns
    configurable responses. Useful for testing without hardware.
    """

    def __init__(
        self,
        on_send: Optional[Callable[[str], None]] = None,
        on_recv: Optional[Callable[[str], None]] = None,
    ):
        self.on_send = on_send
        self.on_recv = on_recv
        self._is_open = False
        self.sent_singles: list[GalvoCommand] = []
        self.sent_lists: list[bytes] = []
        self._implied_response: Optional[bytes] = None
        self._serial_number: str = "MOCK-GALVO-001"
        self._version: int = 0x0020
        self._list_executed: bool = False
        self._position_x: int = 0x8000
        self._position_y: int = 0x8000

    @property
    def is_connected(self) -> bool:
        return self._is_open

    def open(self, index: int = 0) -> int:
        self._is_open = True
        self.sent_singles.clear()
        self.sent_lists.clear()
        return index

    def close(self, index: int = 0) -> None:
        self._is_open = False

    def write(self, index: int = 0, packet: Optional[bytes] = None) -> None:
        if packet is None:
            return
        length = len(packet)
        if length == COMMAND_SIZE:
            self._process_single(packet)
        elif length == LIST_BUFFER_SIZE:
            self._process_list(packet)

    def read(self, index: int = 0) -> bytes:
        if self._implied_response is not None:
            resp = self._implied_response
            self._implied_response = None
            return resp
        read = bytearray(8)
        for r in range(len(read)):
            read[r] = random.randint(0, 255)
        return struct.pack("8B", *read)

    def set_implied_response(self, data: Optional[bytes]) -> None:
        self._implied_response = data

    def _process_single(self, packet: bytes) -> None:
        cmd = GalvoCommand.from_bytes(packet)
        self.sent_singles.append(cmd)
        if self.on_send:
            name = SINGLE_COMMAND_NAMES.get(cmd.cmd, f"0x{cmd.cmd:04X}")
            self.on_send(
                f"{cmd.cmd:04X}:{cmd.v1:04X}:{cmd.v2:04X}:"
                f"{cmd.v3:04X}:{cmd.v4:04X}:{cmd.v5:04X} {name}"
            )

        if cmd.cmd == GetSerialNo:
            self._implied_response = self._serial_number.encode("ascii")[:8]
        elif cmd.cmd == GetVersion:
            self._implied_response = struct.pack("<4H", 0, 0, 0, self._version)
        elif cmd.cmd == ReadPort:
            pinvalue = 0
            self._implied_response = (
                b"\x00\x00" + struct.pack("<H", pinvalue) + b"\x00\x00\x00\x00"
            )
        else:
            self._implied_response = struct.pack("<4H", 0, 0, 0, 0x20)

    def _process_list(self, packet: bytes) -> None:
        self.sent_lists.append(packet)
        self._list_executed = True

        if self.on_send:
            commands = []
            last_cmd = None
            repeats = 0
            for i in range(0, LIST_BUFFER_SIZE, COMMAND_SIZE):
                chunk = packet[i : i + COMMAND_SIZE]
                cmd = GalvoCommand.from_bytes(chunk)
                name = LIST_COMMAND_NAMES.get(cmd.cmd, f"0x{cmd.cmd:04X}")
                line = (
                    f"{cmd.cmd:04X}:{cmd.v1:04X}:{cmd.v2:04X}:"
                    f"{cmd.v3:04X}:{cmd.v4:04X}:{cmd.v5:04X}"
                    f" {name}"
                )
                if line == last_cmd:
                    repeats += 1
                    continue
                if repeats:
                    commands.append(f"... repeated {repeats} times ...")
                repeats = 0
                commands.append(line)
                last_cmd = line
            if repeats:
                commands.append(f"... repeated {repeats} times ...")
            self.on_send("\n".join(commands))

    @property
    def was_list_executed(self) -> bool:
        return self._list_executed

    def reset(self) -> None:
        self.sent_singles.clear()
        self.sent_lists.clear()
        self._list_executed = False
        self._implied_response = None
