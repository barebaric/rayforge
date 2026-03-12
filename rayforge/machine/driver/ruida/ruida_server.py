"""
Ruida Server Protocol - Server-side command handling.

Handles parsing of incoming commands and generation of responses
for the Ruida laser controller simulator.
"""

import logging
from typing import Callable, Optional, Tuple

from .ruida_maps import (
    A7_KEYPRESS_COMMANDS,
    C6_DELAY_COMMANDS,
    C6_PART_POWER_COMMANDS,
    C6_POWER_COMMANDS,
    CA_MODE_COMMANDS,
    CHECKSUM_COMMANDS,
    DA_COMMANDS,
    D8_COMMANDS,
    D8_KEYDOWN_AXIS_MAP,
    D8_KEYUP_AXIS_MAP,
    E5_COMMANDS,
    E8_FILE_ACTIONS,
    INTERFACE_COMMANDS,
)
from .ruida_protocol import RuidaState
from .ruida_util import (
    decode14,
    decode35,
    decodeu14,
    decodeu35,
    encode35,
    parse_mem,
)

logger = logging.getLogger(__name__)


class RuidaServer:
    """
    Ruida server-side protocol handler.

    Parses incoming commands, updates machine state, and generates responses.
    Use this for implementing a Ruida controller simulator.
    """

    def __init__(
        self,
        state: Optional[RuidaState] = None,
        on_command: Optional[Callable[[str, bytes], None]] = None,
        model: str = "644XG",
    ):
        self.state = state or RuidaState()
        self.on_command = on_command
        self.model = model

    def process_commands(self, data: bytes) -> bytes:
        """Process unswizzled commands and return response."""
        response = b""
        pos = 0

        while pos < len(data):
            cmd = data[pos]
            if cmd < 0x80:
                pos += 1
                continue

            cmd_response, cmd_len = self._process_single_command(data[pos:])
            if cmd_response:
                response += cmd_response
            pos += cmd_len if cmd_len > 0 else 1

        return response

    def _process_single_command(self, data: bytes) -> Tuple[bytes, int]:
        """
        Process a single command and return (response, length consumed).

        Returns:
            Tuple of (response_bytes, bytes_consumed)
        """
        if len(data) < 1:
            return b"", 0

        cmd = data[0]
        s = self.state

        self._accumulate_checksum(data)

        if cmd == 0xCC:
            self._log_command("ACK from machine", data[:1])
            return b"", 1

        if cmd == 0xCD:
            self._log_command("ERR from machine", data[:1])
            return b"", 1

        if cmd == 0xCE:
            self._log_command("Keep Alive", data[:1])
            return b"\xcc", 1

        if cmd == 0xD0:
            return self._handle_d0_command(data)

        if cmd == 0xD7:
            self._log_command("End Of File", data[:1])
            s.program_mode = False
            return b"", 1

        if cmd == 0xD8:
            return self._handle_d8_command(data)

        if cmd == 0xD9:
            return self._handle_d9_command(data)

        if cmd == 0xDA:
            return self._handle_da_command(data)

        if cmd == 0xA5:
            return self._handle_a5_command(data)

        if cmd == 0xA7:
            return self._handle_a7_command(data)

        if cmd == 0xE5:
            return self._handle_e5_command(data)

        if cmd == 0xE7:
            return self._handle_e7_command(data)

        if cmd == 0xE8:
            return self._handle_e8_command(data)

        if cmd == 0x88:
            return self._handle_move_abs(data)

        if cmd == 0x89:
            return self._handle_move_rel(data)

        if cmd == 0xA8:
            return self._handle_cut_abs(data)

        if cmd == 0xA9:
            return self._handle_cut_rel(data)

        if cmd == 0xC6:
            return self._handle_c6_command(data)

        if cmd == 0xC7:
            return self._handle_power_command("Imd Power 1", data, 3)

        if cmd == 0xC0:
            return self._handle_power_command("Imd Power 2", data, 3)

        if cmd == 0xC2:
            return self._handle_power_command("Imd Power 3", data, 3)

        if cmd == 0xC3:
            return self._handle_power_command("Imd Power 4", data, 3)

        if cmd == 0xC8:
            return self._handle_power_command("End Power 1", data, 3)

        if cmd == 0xC1:
            return self._handle_power_command("End Power 2", data, 3)

        if cmd == 0xC4:
            return self._handle_power_command("End Power 3", data, 3)

        if cmd == 0xC5:
            return self._handle_power_command("End Power 4", data, 3)

        if cmd == 0xC9:
            return self._handle_c9_command(data)

        if cmd == 0xCA:
            return self._handle_ca_command(data)

        if cmd == 0x8A:
            return self._handle_move_rel_x(data)

        if cmd == 0x8B:
            return self._handle_move_rel_y(data)

        if cmd == 0xAA:
            return self._handle_cut_rel_x(data)

        if cmd == 0xAB:
            return self._handle_cut_rel_y(data)

        if cmd == 0x80:
            return self._handle_axis_move(data)

        if cmd == 0xA0:
            return self._handle_axis_move_a0(data)

        if cmd == 0xEA:
            index = data[1] if len(data) > 1 else 0
            self._log_command(f"Array Start ({index})", data[:2])
            return b"", 2

        if cmd == 0xEB:
            self._log_command("Array End", data[:1])
            return b"", 1

        if cmd == 0xF0:
            self._log_command("Ref Point Set", data[:1])
            return b"", 1

        if cmd == 0xF1:
            return self._handle_f1_command(data)

        if cmd == 0xF2:
            return self._handle_f2_command(data)

        if cmd == 0xE6:
            if len(data) > 1 and data[1] == 0x01:
                self._log_command("Set Absolute", data[:2])
                return b"", 2
            return b"", 1

        self._log_command(f"Unknown command 0x{cmd:02X}", data[:1])
        return b"", 1

    def _handle_d0_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle D0 set inhale zone command."""
        if len(data) < 2:
            return b"", 1

        zone = data[1]
        self._log_command(f"Set Inhale Zone: {zone}", data[:2])
        return b"", 2

    def _handle_d8_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle D8 realtime commands."""
        if len(data) < 2:
            return b"", 1

        s = self.state
        subcmd = data[1]
        desc = D8_COMMANDS.get(subcmd, f"Unknown D8 subcommand 0x{subcmd:02X}")
        self._log_command(desc, data[:2])

        if subcmd == 0x00:
            s.program_mode = True
            s.machine_status = 21
        elif subcmd == 0x01:
            s.program_mode = False
            s.machine_status = 22
        elif subcmd == 0x02:
            s.machine_status = 23
        elif subcmd == 0x03:
            if s.program_mode:
                s.machine_status = 21
            else:
                s.machine_status = 22
        elif subcmd in (0x10, 0x11, 0x12):
            s.ref_point_mode = {0x10: 2, 0x11: 1, 0x12: 0}.get(subcmd, 0)
        elif subcmd in range(0x20, 0x28):
            if subcmd in D8_KEYDOWN_AXIS_MAP:
                axis, direction = D8_KEYDOWN_AXIS_MAP[subcmd]
                s.jog_active[axis] = direction * s.jog_speed
        elif subcmd == 0x2A:
            s.x = 0
            s.y = 0
        elif subcmd == 0x2C:
            s.z = 0
        elif subcmd == 0x2D:
            s.u = 0
        elif subcmd == 0x2E:
            pass
        elif subcmd in range(0x30, 0x38):
            if subcmd in D8_KEYUP_AXIS_MAP:
                s.jog_active[D8_KEYUP_AXIS_MAP[subcmd]] = 0
        elif subcmd in range(0x40, 0x48):
            pass
        elif subcmd in range(0x48, 0x50):
            pass
        elif subcmd == 0x39:
            s.a = 0
        elif subcmd == 0x3A:
            s.b = 0
        elif subcmd == 0x3B:
            s.c = 0
        elif subcmd == 0x3C:
            s.d = 0
        elif subcmd == 0x51:
            pass

        return b"", 2

    def _handle_d9_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle D9 rapid move commands."""
        if len(data) < 2:
            return b"", 1

        s = self.state
        subcmd = data[1]

        def get_opt_desc(opts: int) -> str:
            if opts == 0x00:
                return "Origin"
            elif opts == 0x01:
                return "Light/Origin"
            elif opts == 0x02:
                return ""
            elif opts == 0x03:
                return "Light"
            return f"opts={opts}"

        if subcmd in (0x00, 0x01, 0x02, 0x03, 0x50, 0x51, 0x52, 0x53):
            if len(data) < 8:
                return b"", 1
            opts = data[2]
            coord = decode35(data[3:8])
            base_axis = subcmd & 0x0F
            axis = {0x00: "X", 0x01: "Y", 0x02: "Z", 0x03: "U"}.get(
                base_axis, "?"
            )
            opt_desc = get_opt_desc(opts)
            self._log_command(
                f"Rapid move {opt_desc} {axis}: {coord:+d}um (rel)", data[:8]
            )
            if base_axis == 0x00:
                s.x += coord
            elif base_axis == 0x01:
                s.y += coord
            elif base_axis == 0x02:
                s.z += coord
            elif base_axis == 0x03:
                s.u += coord
            return b"", 8

        if subcmd == 0x0F:
            if len(data) < 8:
                return b"", 1
            opts = data[2]
            opt_desc = get_opt_desc(opts)
            self._log_command(f"Rapid Feed Axis Move {opt_desc}", data[:8])
            return b"", 8

        if subcmd == 0x10:
            if len(data) < 8:
                return b"", 1
            opts = data[2]
            coord = decode35(data[3:8])
            opt_desc = get_opt_desc(opts)
            self._log_command(
                f"Rapid move {opt_desc} X: {coord:+d}um (rel)", data[:8]
            )
            s.x += coord
            return b"", 8

        if subcmd == 0x11:
            if len(data) < 8:
                return b"", 1
            opts = data[2]
            coord = decode35(data[3:8])
            opt_desc = get_opt_desc(opts)
            self._log_command(
                f"Rapid move {opt_desc} Y: {coord:+d}um (rel)", data[:8]
            )
            s.y += coord
            return b"", 8

        if subcmd == 0x60:
            if len(data) < 13:
                return b"", 1
            opts = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            opt_desc = get_opt_desc(opts)
            self._log_command(
                f"Rapid move {opt_desc} XY: ({x}um, {y}um)", data[:13]
            )
            s.x = x
            s.y = y
            return b"", 13

        if subcmd in (0x30, 0x70):
            if len(data) < 18:
                return b"", 1
            opts = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            u = decode35(data[13:18])
            opt_desc = get_opt_desc(opts)
            self._log_command(
                f"Rapid move {opt_desc} XYU: ({x}um, {y}um, {u}um)",
                data[:18],
            )
            s.x = x
            s.y = y
            s.u = u
            return b"", 18

        return b"", 2

    def _handle_da_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle DA memory commands."""
        if len(data) < 4:
            return b"", 1

        s = self.state
        subcmd = data[1]
        mem = parse_mem(data[2:4])

        if subcmd == 0x00:
            name, value = s.mem_lookup(mem)
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(f"{desc} {name} (mem: 0x{mem:04X})", data[:4])
            if isinstance(value, bytes):
                encoded = value
            else:
                encoded = encode35(value)
            response = b"\xda\x01" + data[2:4] + encoded + encoded
            return response, 4

        if subcmd == 0x01:
            if len(data) < 14:
                return b"", 1
            v0 = decodeu35(data[4:9])
            v1 = decodeu35(data[9:14])
            name, _ = s.mem_lookup(mem)
            s.memory_values[mem] = v0
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(
                f"{desc} {name} (mem: 0x{mem:04X}) = {v0} (0x{v0:08x}) "
                f"{v1} (0x{v1:08x})",
                data[:14],
            )
            return b"", 14

        if subcmd == 0x04:
            self._log_command("OEM On/Off, CardIO On/Off", data[:4])
            return b"\xda\x04" + b"\x00" * 10, 4

        if subcmd in (0x05, 0x54):
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(desc, data[:4])
            return b"\xda" + bytes([subcmd]) + b"\x00" * 20, 4

        if subcmd in (0x06, 0x52):
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(desc, data[:4])
            return b"", 4

        if subcmd in (0x10, 0x53):
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(desc, data[:4])
            return b"", 4

        if subcmd in (0x30, 0x31):
            desc = DA_COMMANDS.get(subcmd, f"Unknown DA 0x{subcmd:02X}")
            self._log_command(desc, data[:4])
            return b"\xda" + bytes([subcmd]) + b"\x00" * 20, 4

        if subcmd == 0x60:
            if len(data) < 4:
                return b"", 1
            v = decode14(data[2:4])
            self._log_command(f"RD-FUNCTION-UNK1 {v}", data[:4])
            return b"", 4

        return b"", 4

    def _handle_a5_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle A5 interface commands (also on jog port)."""
        if len(data) < 3:
            return b"", 1

        if data[1] in (0x50, 0x51):
            key_type = "Down" if data[1] == 0x50 else "Up"
            desc = INTERFACE_COMMANDS.get(data[2], f"Unknown(0x{data[2]:02X})")
            self._log_command(f"Interface {key_type}: {desc}", data[:3])
            return b"", 3

        if data[1] == 0x53:
            self._log_command("Interface Frame", data[:3])
            return b"", 3

        return b"", 3

    def _handle_a7_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle A7 keypress commands."""
        if len(data) < 2:
            return b"", 1

        key = data[1]
        desc = A7_KEYPRESS_COMMANDS.get(key, f"KeyPress Unknown(0x{key:02X})")
        self._log_command(desc, data[:2])
        return b"", 2

    def _handle_e5_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle E5 document commands."""
        s = self.state
        if len(data) < 1:
            return b"", 0

        if len(data) == 1:
            self._log_command("Lightburn Swizzle Modulation E5", data[:1])
            return b"", 1

        subcmd = data[1]

        if subcmd == 0x00:
            if len(data) < 4:
                return b"", 2
            filenumber = decodeu14(data[2:4])
            desc = E5_COMMANDS.get(subcmd, f"Unknown E5 0x{subcmd:02X}")
            self._log_command(f"{desc} {filenumber}", data[:4])
            return b"\xe5\x00" + encode35(0) + encode35(0), 4

        if subcmd == 0x02:
            desc = E5_COMMANDS.get(subcmd, f"Unknown E5 0x{subcmd:02X}")
            self._log_command(desc, data[:2])
            return b"", 2

        if subcmd == 0x03:
            desc = E5_COMMANDS.get(subcmd, f"Unknown E5 0x{subcmd:02X}")
            self._log_command(desc, data[:2])
            return b"", 2

        if subcmd == 0x04:
            self._log_command("Chunk Write Check", data[:2])
            return b"", 2

        if subcmd == 0x05:
            if len(data) < 7:
                return b"", 2
            checksum = decodeu35(data[2:7])
            desc = E5_COMMANDS.get(subcmd, f"Unknown E5 0x{subcmd:02X}")
            self._log_command(f"{desc}: {checksum}", data[:7])
            if s.checksum_enabled:
                if checksum != s.file_checksum_accumulator:
                    logger.warning(
                        f"File checksum mismatch: received {checksum}, "
                        f"calculated {s.file_checksum_accumulator}"
                    )
                else:
                    logger.debug(f"File checksum verified: {checksum}")
            s.file_checksum = checksum
            s.file_checksum_accumulator = 0
            return b"", 7

        return b"", 2

    def _handle_e7_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle E7 file layout commands."""
        if len(data) < 2:
            return b"", 1

        s = self.state
        subcmd = data[1]

        if subcmd == 0x00:
            self._log_command("Block End", data[:2])
            return b"", 2

        if subcmd == 0x01:
            name_end = data.find(b"\x00", 2)
            if name_end == -1:
                name_end = min(len(data), 12)
            filename = data[2:name_end].decode("ascii", errors="replace")
            s.filename = filename
            self._log_command(f"Filename: {filename}", data[:name_end])
            return b"", name_end + 1

        if subcmd == 0x03:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Process TopLeft ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x04:
            if len(data) < 16:
                return b"", 2
            v0 = decode14(data[2:4])
            v1 = decode14(data[4:6])
            v2 = decode14(data[6:8])
            v3 = decode14(data[8:10])
            v4 = decode14(data[10:12])
            v5 = decode14(data[12:14])
            v6 = decode14(data[14:16])
            self._log_command(
                f"Process Repeat ({v0}, {v1}, {v2}, {v3}, {v4}, {v5}, {v6})",
                data[:16],
            )
            return b"", 16

        if subcmd == 0x05:
            if len(data) < 3:
                return b"", 2
            direction = data[2]
            self._log_command(f"Array Direction: {direction}", data[:3])
            return b"", 3

        if subcmd == 0x06:
            if len(data) < 12:
                return b"", 2
            v0 = decode35(data[2:7])
            v1 = decode35(data[7:12])
            self._log_command(f"Feed Repeat ({v0}, {v1})", data[:12])
            return b"", 12

        if subcmd == 0x07:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Process BottomRight ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x08:
            if len(data) < 16:
                return b"", 2
            v0 = decode14(data[2:4])
            v1 = decode14(data[4:6])
            v2 = decode14(data[6:8])
            v3 = decode14(data[8:10])
            v4 = decode14(data[10:12])
            v5 = decode14(data[12:14])
            v6 = decode14(data[14:16])
            self._log_command(
                f"Array Repeat ({v0}, {v1}, {v2}, {v3}, {v4}, {v5}, {v6})",
                data[:16],
            )
            return b"", 16

        if subcmd == 0x09:
            if len(data) < 7:
                return b"", 2
            length = decode35(data[2:7])
            self._log_command(f"Feed Length: {length}um", data[:7])
            return b"", 7

        if subcmd == 0x0A:
            if len(data) < 7:
                return b"", 2
            v = decodeu35(data[2:7])
            self._log_command(f"Feed Info: {v}", data[:7])
            return b"", 7

        if subcmd == 0x0B:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Array En Mirror Cut: {v}", data[:3])
            return b"", 3

        if subcmd == 0x0C:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Array Mirror Cut Distance: {v}", data[:3])
            return b"", 3

        if subcmd == 0x13:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Array TopLeft ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x17:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Array BottomRight ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x23:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Array Add ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x24:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Array Mirror: {v}", data[:3])
            return b"", 3

        if subcmd == 0x32:
            if len(data) < 7:
                return b"", 2
            v = decodeu35(data[2:7])
            self._log_command(f"Set Tick Count: {v}", data[:7])
            return b"", 7

        if subcmd == 0x35:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Block X Size ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x36:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Set File Empty: {v}", data[:3])
            return b"", 3

        if subcmd == 0x37:
            if len(data) < 12:
                return b"", 2
            v0 = decodeu35(data[2:7])
            v1 = decodeu35(data[7:12])
            self._log_command(f"Array Even Distance ({v0}, {v1})", data[:12])
            return b"", 12

        if subcmd == 0x38:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Set Feed Auto Pause: {v}", data[:3])
            return b"", 3

        if subcmd == 0x3A:
            self._log_command("Union Block Property", data[:2])
            return b"", 2

        if subcmd == 0x3B:
            if len(data) < 3:
                return b"", 2
            v = data[2]
            self._log_command(f"Set File Property: {v}", data[:3])
            return b"", 3

        if subcmd == 0x46:
            if len(data) < 7:
                return b"", 2
            v = decodeu35(data[2:7])
            self._log_command(f"BY Test: 0x{v:08X}", data[:7])
            return b"", 7

        if subcmd == 0x50:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Document Min Point ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x51:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Document Max Point ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x52:
            if len(data) < 13:
                return b"", 2
            part = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            self._log_command(f"Part {part} TopLeft ({x}um, {y}um)", data[:13])
            return b"", 13

        if subcmd == 0x53:
            if len(data) < 13:
                return b"", 2
            part = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            self._log_command(
                f"Part {part} BottomRight ({x}um, {y}um)", data[:13]
            )
            return b"", 13

        if subcmd == 0x54:
            if len(data) < 8:
                return b"", 2
            axis_id = data[2]
            coord = decode35(data[3:8])
            self._log_command(
                f"Pen Offset Axis={axis_id}: {coord}um", data[:8]
            )
            return b"", 8

        if subcmd == 0x55:
            if len(data) < 8:
                return b"", 2
            axis_id = data[2]
            coord = decode35(data[3:8])
            self._log_command(
                f"Layer Offset Axis={axis_id}: {coord}um", data[:8]
            )
            return b"", 8

        if subcmd == 0x57:
            self._log_command("PList Feed", data[:2])
            return b"", 2

        if subcmd == 0x60:
            if len(data) < 3:
                return b"", 2
            index = data[2]
            self._log_command(f"Set Current Element Index: {index}", data[:3])
            return b"", 3

        if subcmd == 0x61:
            if len(data) < 13:
                return b"", 2
            part = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            self._log_command(
                f"Part {part} Ex TopLeft ({x}um, {y}um)", data[:13]
            )
            return b"", 13

        if subcmd == 0x62:
            if len(data) < 13:
                return b"", 2
            part = data[2]
            x = decode35(data[3:8])
            y = decode35(data[8:13])
            self._log_command(
                f"Part {part} Ex BottomRight ({x}um, {y}um)", data[:13]
            )
            return b"", 13

        return b"", 2

    def _handle_e8_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle E8 file interaction commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd == 0x00:
            if len(data) < 4:
                return b"", 2
            filenumber = parse_mem(data[2:4])
            self._log_command(
                f"{E8_FILE_ACTIONS.get(subcmd, '?')} Document {filenumber}",
                data[:4],
            )
            return b"\xe8\x00" + b"\x00" * 10, 4

        if subcmd == 0x02:
            self._log_command("File transfer", data[:2])
            return b"", 2

        if subcmd in (0x01, 0x03, 0x04):
            if len(data) < 4:
                return b"", 2
            filenumber = parse_mem(data[2:4])
            self._log_command(
                f"{E8_FILE_ACTIONS.get(subcmd, '?')} Document {filenumber}",
                data[:4],
            )
            if subcmd == 0x01:
                name = f"FILE{filenumber:04d}"
                return data[:4] + name.encode()[:8] + b"\x00", 4
            return b"", 4

        return b"", 2

    def _handle_move_abs(self, data: bytes) -> Tuple[bytes, int]:
        """Handle absolute move command (0x88)."""
        s = self.state
        if len(data) < 11:
            return b"", 1
        x = decode35(data[1:6])
        y = decode35(data[6:11])
        self._log_command(f"Move Abs ({x}um, {y}um)", data[:11])
        s.x = x
        s.y = y
        return b"", 11

    def _handle_move_rel(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative move command (0x89)."""
        s = self.state
        if len(data) < 5:
            return b"", 1
        dx = decode14(data[1:3])
        dy = decode14(data[3:5])
        self._log_command(f"Move Rel ({dx:+d}um, {dy:+d}um)", data[:5])
        s.x += dx
        s.y += dy
        return b"", 5

    def _handle_move_rel_x(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative X move command (0x8A)."""
        s = self.state
        if len(data) < 3:
            return b"", 1
        dx = decode14(data[1:3])
        self._log_command(f"Move Rel X ({dx:+d}um)", data[:3])
        s.x += dx
        return b"", 3

    def _handle_move_rel_y(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative Y move command (0x8B)."""
        s = self.state
        if len(data) < 3:
            return b"", 1
        dy = decode14(data[1:3])
        self._log_command(f"Move Rel Y ({dy:+d}um)", data[:3])
        s.y += dy
        return b"", 3

    def _handle_cut_abs(self, data: bytes) -> Tuple[bytes, int]:
        """Handle absolute cut command (0xA8)."""
        s = self.state
        if len(data) < 11:
            return b"", 1
        x = decode35(data[1:6])
        y = decode35(data[6:11])
        self._log_command(f"Cut Abs ({x}um, {y}um)", data[:11])
        s.x = x
        s.y = y
        return b"", 11

    def _handle_cut_rel(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative cut command (0xA9)."""
        s = self.state
        if len(data) < 5:
            return b"", 1
        dx = decode14(data[1:3])
        dy = decode14(data[3:5])
        self._log_command(f"Cut Rel ({dx:+d}um, {dy:+d}um)", data[:5])
        s.x += dx
        s.y += dy
        return b"", 5

    def _handle_cut_rel_x(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative X cut command (0xAA)."""
        s = self.state
        if len(data) < 3:
            return b"", 1
        dx = decode14(data[1:3])
        self._log_command(f"Cut Rel X ({dx:+d}um)", data[:3])
        s.x += dx
        return b"", 3

    def _handle_cut_rel_y(self, data: bytes) -> Tuple[bytes, int]:
        """Handle relative Y cut command (0xAB)."""
        s = self.state
        if len(data) < 3:
            return b"", 1
        dy = decode14(data[1:3])
        self._log_command(f"Cut Rel Y ({dy:+d}um)", data[:3])
        s.y += dy
        return b"", 3

    def _handle_c6_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle C6 power/delay commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd in C6_POWER_COMMANDS:
            if len(data) < 4:
                return b"", 2
            power = decodeu14(data[2:4]) / 163.84
            self._log_command(
                f"{C6_POWER_COMMANDS[subcmd]}: {power:.1f}%", data[:4]
            )
            return b"", 4

        if subcmd in C6_DELAY_COMMANDS:
            if len(data) < 7:
                return b"", 2
            delay = decodeu35(data[2:7]) / 1000.0
            self._log_command(
                f"{C6_DELAY_COMMANDS[subcmd]}: {delay}ms", data[:7]
            )
            return b"", 7

        if subcmd in C6_PART_POWER_COMMANDS:
            if len(data) < 5:
                return b"", 2
            part = data[2]
            power = decodeu14(data[3:5]) / 163.84
            self._log_command(
                f"Part {part} {C6_PART_POWER_COMMANDS[subcmd]}: {power:.1f}%",
                data[:5],
            )
            return b"", 5

        if subcmd == 0x60:
            if len(data) < 9:
                return b"", 2
            laser = data[2]
            part = data[3]
            freq = decodeu35(data[4:9])
            self._log_command(
                f"Part {part} Laser {laser} Frequency: {freq}Hz", data[:9]
            )
            return b"", 9

        return b"", 2

    def _handle_power_command(
        self, name: str, data: bytes, length: int
    ) -> Tuple[bytes, int]:
        """Handle immediate/end power commands."""
        if len(data) < length:
            return b"", 1
        power = decodeu14(data[1:3]) / 163.84 if len(data) >= 3 else 0
        self._log_command(f"{name}: {power:.1f}%", data[:length])
        return b"", length

    def _handle_c9_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle C9 speed commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd == 0x02:
            if len(data) < 7:
                return b"", 2
            speed = decode35(data[2:7]) / 1000.0
            self._log_command(f"Speed Laser 1: {speed}mm/s", data[:7])
            return b"", 7

        if subcmd == 0x03:
            if len(data) < 7:
                return b"", 2
            speed = decode35(data[2:7]) / 1000.0
            self._log_command(f"Axis Speed: {speed}mm/s", data[:7])
            return b"", 7

        if subcmd == 0x04:
            if len(data) < 8:
                return b"", 2
            part = data[2]
            speed = decode35(data[3:8]) / 1000.0
            self._log_command(f"Part {part} Speed: {speed}mm/s", data[:8])
            return b"", 8

        if subcmd in (0x05, 0x06):
            if len(data) < 7:
                return b"", 2
            speed = decode35(data[2:7]) / 1000.0
            desc = "Force Eng Speed" if subcmd == 0x05 else "Axis Move Speed"
            self._log_command(f"{desc}: {speed}mm/s", data[:7])
            return b"", 7

        return b"", 2

    def _handle_ca_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle CA layer/mode commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd == 0x01:
            if len(data) < 3:
                return b"", 2
            desc = CA_MODE_COMMANDS.get(data[2], f"Mode 0x{data[2]:02X}")
            self._log_command(desc, data[:3])
            return b"", 3

        if subcmd == 0x02:
            if len(data) < 3:
                return b"", 2
            part = data[2]
            self._log_command(f"Part {part} Layer Number", data[:3])
            return b"", 3

        if subcmd == 0x03:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"EnLaserTube Start: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x04:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"X Sign Map: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x05:
            if len(data) < 7:
                return b"", 2
            c = decodeu35(data[2:7])
            r = c & 0xFF
            g = (c >> 8) & 0xFF
            b = (c >> 16) & 0xFF
            self._log_command(f"Layer Color: #{b:02X}{g:02X}{r:02X}", data[:7])
            return b"", 7

        if subcmd == 0x06:
            if len(data) < 8:
                return b"", 2
            part = data[2]
            c = decodeu35(data[3:8])
            r = c & 0xFF
            g = (c >> 8) & 0xFF
            b = (c >> 16) & 0xFF
            self._log_command(
                f"Part {part} Color: #{b:02X}{g:02X}{r:02X}", data[:8]
            )
            return b"", 8

        if subcmd == 0x10:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"EnExIO Start: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x22:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"Max Layer Part: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x30:
            if len(data) < 4:
                return b"", 2
            file_id = decodeu14(data[2:4])
            self._log_command(f"U File ID: {file_id}", data[:4])
            return b"", 4

        if subcmd == 0x40:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"ZU Map: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x41:
            if len(data) < 4:
                return b"", 2
            part = data[2]
            mode = data[3]
            self._log_command(
                f"Layer Select Part {part} Mode {mode}", data[:4]
            )
            return b"", 4

        return b"", 2

    def _handle_axis_move(self, data: bytes) -> Tuple[bytes, int]:
        """Handle 0x80 axis move commands."""
        s = self.state
        if len(data) < 2:
            return b"", 1

        if data[1] == 0x00:
            if len(data) < 7:
                return b"", 2
            coord = decode35(data[2:7])
            self._log_command(f"Axis X Move: {coord}um", data[:7])
            s.x = coord
            return b"", 7

        if data[1] == 0x08:
            if len(data) < 7:
                return b"", 2
            coord = decode35(data[2:7])
            self._log_command(f"Axis Z Move: {coord}um", data[:7])
            s.z = coord
            return b"", 7

        return b"", 2

    def _handle_axis_move_a0(self, data: bytes) -> Tuple[bytes, int]:
        """Handle 0xA0 axis move commands."""
        s = self.state
        if len(data) < 2:
            return b"", 1

        if data[1] == 0x00:
            if len(data) < 7:
                return b"", 2
            coord = decode35(data[2:7])
            self._log_command(f"Axis Y Move: {coord}um", data[:7])
            s.y = coord
            return b"", 7

        if data[1] == 0x08:
            if len(data) < 7:
                return b"", 2
            coord = decode35(data[2:7])
            self._log_command(f"Axis U Move: {coord}um", data[:7])
            s.u = coord
            return b"", 7

        return b"", 2

    def _handle_f1_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle F1 commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd == 0x00:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"Element Max Index: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x01:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"Element Name Max Index: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x02:
            if len(data) < 3:
                return b"", 2
            self._log_command(f"Enable Block Cutting: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x03:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Display Offset ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x04:
            if len(data) < 7:
                return b"", 2
            v = decodeu35(data[2:7])
            self._log_command(f"Feed Auto Calc: {v}", data[:7])
            return b"", 7

        if subcmd == 0x10:
            if len(data) < 4:
                return b"", 2
            v0 = data[2]
            v1 = data[3]
            self._log_command(f"Unknown Common ({v0}, {v1})", data[:4])
            return b"", 4

        if subcmd == 0x20:
            if len(data) < 4:
                return b"", 2
            v0 = data[2]
            v1 = data[3]
            self._log_command(f"Unknown F1 0x20 ({v0}, {v1})", data[:4])
            return b"", 4

        return b"", 2

    def _handle_f2_command(self, data: bytes) -> Tuple[bytes, int]:
        """Handle F2 element commands."""
        if len(data) < 2:
            return b"", 1

        subcmd = data[1]

        if subcmd in (0x00, 0x01):
            if len(data) < 3:
                return b"", 2
            label = "Index" if subcmd == 0 else "Name Index"
            self._log_command(f"Element {label}: {data[2]}", data[:3])
            return b"", 3

        if subcmd == 0x02:
            name_end = min(len(data), 12)
            self._log_command("Element Name", data[:name_end])
            return b"", name_end

        if subcmd in (0x03, 0x04, 0x06):
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            desc = {0x03: "Min Point", 0x04: "Max Point", 0x06: "Add"}.get(
                subcmd, "?"
            )
            self._log_command(
                f"Element Array {desc} ({x}um, {y}um)", data[:12]
            )
            return b"", 12

        if subcmd == 0x05:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(f"Element Array ({x}um, {y}um)", data[:12])
            return b"", 12

        if subcmd == 0x07:
            if len(data) < 12:
                return b"", 2
            x = decode35(data[2:7])
            y = decode35(data[7:12])
            self._log_command(
                f"Element Array Mirror ({x}um, {y}um)", data[:12]
            )
            return b"", 12

        return b"", 2

    def _log_command(self, desc: str, data: bytes) -> None:
        """Log a command description."""
        hex_data = data.hex() if data else ""
        logger.debug(f"--> {hex_data}\t({desc})")
        if self.on_command:
            self.on_command(desc, data)

    def _accumulate_checksum(self, data: bytes) -> None:
        """Accumulate bytes into the file checksum for relevant commands."""
        s = self.state
        if s.checksum_enabled and data and data[0] in CHECKSUM_COMMANDS:
            s.file_checksum_accumulator += sum(data)
