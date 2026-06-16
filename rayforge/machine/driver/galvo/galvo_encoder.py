"""
Galvo Encoder - Converts Ops commands to EzCad2 binary protocol.

Produces both binary output (list buffer commands) for the controller
and human-readable text representation for UI display.
"""

import logging
import struct
from typing import TYPE_CHECKING, List, Optional

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.types import CommandType

from ....pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
    OpsEncoder,
)
from .galvo_consts import GALVO_CENTER

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ....machine.models.machine import Machine

logger = logging.getLogger(__name__)


class GalvoEncoder(OpsEncoder):
    """
    Converts Ops commands to EzCad2 binary protocol for galvo controllers.
    """

    def __init__(
        self,
        source: str = "fiber",
        galvos_per_mm: int = 500,
        default_mark_speed: float = 100.0,
        default_jump_speed: float = 2000.0,
        default_frequency: float = 30.0,
        default_power: float = 50.0,
    ):
        self._source = source
        self._galvos_per_mm = galvos_per_mm
        self._default_mark_speed = default_mark_speed
        self._default_jump_speed = default_jump_speed
        self._default_frequency = default_frequency
        self._default_power = default_power
        self._reset_state()

    def _reset_state(self) -> None:
        self._binary_chunks: List[bytes] = []
        self._text_lines: List[str] = []
        self._current_pos: Point3D = (0.0, 0.0, 0.0)
        self._last_x: int = GALVO_CENTER
        self._last_y: int = GALVO_CENTER
        self._power: Optional[float] = None
        self._cut_speed: Optional[float] = None
        self._travel_speed: Optional[float] = None
        self._frequency: Optional[float] = None
        self._in_marking_mode: bool = False

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        self._reset_state()
        op_map = MachineCodeOpMap()

        for i in range(ops.len()):
            start_line = len(self._text_lines)
            self._handle_command(ops, i, machine)
            end_line = len(self._text_lines)

            if end_line > start_line:
                line_indices = list(range(start_line, end_line))
                op_map.op_to_machine_code[i] = line_indices
                for line_num in line_indices:
                    op_map.machine_code_to_op[line_num] = i
            else:
                op_map.op_to_machine_code[i] = []

        binary_data = b"".join(self._binary_chunks)

        if self._text_lines and not self._text_lines[-1]:
            self._text_lines = self._text_lines[:-1]

        return EncodedOutput(
            text="\n".join(self._text_lines),
            op_map=op_map,
            driver_data={"binary": binary_data},
        )

    def _mm_to_galvo(self, mm: float, field_size_mm: float = 200.0) -> int:
        centered = (mm / field_size_mm) * 32768.0 + 32768.0
        return int(max(0, min(0xFFFF, centered)))

    def _power_to_int(self, power_normalized: float) -> int:
        return int(round(power_normalized * 100.0))

    def _speed_to_int(self, speed_mm_s: float) -> int:
        return int(speed_mm_s * self._galvos_per_mm / 1000.0)

    def _frequency_to_period(
        self, freq_khz: float, base: float = 20000.0
    ) -> int:
        return int(round(base / freq_khz)) & 0xFFFF

    def _add_binary(
        self,
        cmd: int,
        v1: int = 0,
        v2: int = 0,
        v3: int = 0,
        v4: int = 0,
        v5: int = 0,
    ) -> None:
        self._binary_chunks.append(struct.pack("<6H", cmd, v1, v2, v3, v4, v5))

    def _handle_command(self, ops: Ops, idx: int, machine: "Machine") -> None:
        ct = ops.command_type(idx)

        if ct == CommandType.SET_POWER:
            power = ops.power(idx)
            self._power = power
            pct = power * 100.0
            self._text_lines.append(f"POWER {pct:.1f}%")
            if self._source == "fiber":
                current = int(round(pct * 0xFFF / 100.0))
                self._add_binary(0x8012, current)
            elif self._source in ("co2", "uv"):
                freq = self._frequency or self._default_frequency
                ratio = int(round(200 * pct / freq))
                self._add_binary(0x800B, ratio)

        elif ct == CommandType.SET_CUT_SPEED:
            speed = ops.speed(idx)
            self._cut_speed = speed
            speed_val = self._speed_to_int(speed)
            if speed_val > 0xFFFF:
                speed_val = 0xFFFF
            self._add_binary(0x800C, speed_val)
            self._text_lines.append(f"MARK_SPEED {speed:.1f}")

        elif ct == CommandType.SET_TRAVEL_SPEED:
            speed = ops.speed(idx)
            self._travel_speed = speed
            speed_val = self._speed_to_int(speed)
            if speed_val > 0xFFFF:
                speed_val = 0xFFFF
            self._add_binary(0x8006, speed_val)
            self._text_lines.append(f"JUMP_SPEED {speed:.1f}")

        elif ct == CommandType.SET_FREQUENCY:
            freq = ops.frequency(idx)
            self._frequency = float(freq)
            if self._source == "fiber":
                period = self._frequency_to_period(float(freq), base=20000.0)
                self._add_binary(0x801B, period)
            elif self._source in ("co2", "uv"):
                period = self._frequency_to_period(float(freq), base=10000.0)
                self._add_binary(0x800A, period)
            self._text_lines.append(f"FREQUENCY {freq}")

        elif ct == CommandType.SET_PULSE_WIDTH:
            pw = ops.pulse_width(idx)
            self._add_binary(0x8026, int(pw))
            self._text_lines.append(f"PULSE_WIDTH {pw:.1f}")

        elif ct == CommandType.MOVE_TO:
            end = ops.endpoint(idx)
            x = self._mm_to_galvo(end[0])
            y = self._mm_to_galvo(end[1])
            distance = int(
                abs(complex(x, y) - complex(self._last_x, self._last_y))
            )
            if distance > 0xFFFF:
                distance = 0xFFFF
            self._add_binary(0x8001, x, y, 0, distance)
            self._text_lines.append(f"JUMP X:{end[0]:.3f} Y:{end[1]:.3f}")
            self._last_x = x
            self._last_y = y
            self._current_pos = end

        elif ct == CommandType.LINE_TO:
            end = ops.endpoint(idx)
            x = self._mm_to_galvo(end[0])
            y = self._mm_to_galvo(end[1])
            distance = int(
                abs(complex(x, y) - complex(self._last_x, self._last_y))
            )
            if distance > 0xFFFF:
                distance = 0xFFFF
            self._add_binary(0x8005, x, y, 0, distance)
            self._text_lines.append(f"MARK X:{end[0]:.3f} Y:{end[1]:.3f}")
            self._last_x = x
            self._last_y = y
            self._current_pos = end

        elif ct == CommandType.ARC_TO:
            end = ops.endpoint(idx)
            i_val, j_val, cw = ops.arc_params(idx)
            self._text_lines.append(
                f"; ARC ({end[0]:.3f}, {end[1]:.3f}) {'CW' if cw else 'CCW'}"
            )
            sub_ops = ops.linearize(idx, self._current_pos)
            for j in range(sub_ops.len()):
                self._handle_command(sub_ops, j, machine)

        elif ct == CommandType.SCAN_LINE:
            end = ops.endpoint(idx)
            self._text_lines.append(
                f"; SCAN_LINE to ({end[0]:.3f}, {end[1]:.3f})"
            )
            sub_ops = ops.linearize(idx, self._current_pos)
            for j in range(sub_ops.len()):
                self._handle_command(sub_ops, j, machine)

        elif ct == CommandType.JOB_START:
            self._add_binary(0x8051)
            self._text_lines.append("; JOB START (READY)")
            active_wcs = machine.active_wcs
            self._text_lines.append(f"; WCS: {active_wcs}")

        elif ct == CommandType.JOB_END:
            self._add_binary(0x8002)
            self._text_lines.append("; JOB END")

        elif ct == CommandType.LAYER_START:
            uid = ops.layer_uid(idx)
            self._text_lines.append(f"; --- Layer {uid[:8]} ---")

        elif ct == CommandType.LAYER_END:
            self._text_lines.append("; --- End Layer ---")

        elif ct == CommandType.WORKPIECE_START:
            uid = ops.workpiece_uid(idx)
            self._text_lines.append(f"; --- Workpiece {uid[:8]} ---")

        elif ct == CommandType.WORKPIECE_END:
            self._text_lines.append("; --- End Workpiece ---")

        elif ct == CommandType.ENABLE_AIR_ASSIST:
            self._add_binary(0x8011, 1)
            self._text_lines.append("AIR_ASSIST ON")

        elif ct == CommandType.DISABLE_AIR_ASSIST:
            self._add_binary(0x8011, 0)
            self._text_lines.append("AIR_ASSIST OFF")

        elif ct == CommandType.SET_LASER:
            self._text_lines.append("; LASER SELECT")
