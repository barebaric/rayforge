"""
Ruida Encoder - Converts Ops commands to Ruida binary protocol.

Produces both binary output for the controller and human-readable
text representation for UI display.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode
from raygeo.ops.types import CommandType

from ....pipeline.encoder.base import (
    EncodedOutput,
    MachineCodeOpMap,
    OpsEncoder,
)
from .ruida_maps import REF_POINT_COMMANDS
from .ruida_util import encode14, encode35

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ....machine.models.machine import Machine

logger = logging.getLogger(__name__)


class RuidaEncoder(OpsEncoder):
    """
    Converts Ops commands to Ruida binary protocol.

    This encoder produces:
    - Binary data for transmission to Ruida controllers
    - Human-readable text for UI display

    Coordinates are converted from mm to micrometers (µm) internally.
    Power is converted from normalized (0.0-1.0) to percentage (0-100)
    and then to the 14-bit value expected by Ruida (0-16384).
    """

    UM_PER_MM = 1000.0
    POWER_SCALE = 16384.0

    def __init__(self):
        self.power: Optional[float] = None
        self.cut_speed: Optional[float] = None
        self.travel_speed: Optional[float] = None
        self.air_assist: bool = False
        self.current_pos: Point3D = (0.0, 0.0, 0.0)
        self.active_laser: int = 1

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        """
        Encode Ops commands to Ruida binary format.

        Args:
            ops: The Ops object containing commands to encode
            machine: The machine configuration
            doc: The document being processed

        Returns:
            EncodedOutput with binary in driver_data["binary"],
            text representation, and op_map
        """
        self._reset_state()

        binary_chunks: List[bytes] = []
        text_lines: List[str] = []
        op_map = MachineCodeOpMap()

        for i in range(ops.len()):
            start_line = len(text_lines)
            self._handle_command(ops, i, machine, binary_chunks, text_lines)
            end_line = len(text_lines)

            if end_line > start_line:
                line_indices = list(range(start_line, end_line))
                op_map.op_to_machine_code[i] = line_indices
                for line_num in line_indices:
                    op_map.machine_code_to_op[line_num] = i
            else:
                op_map.op_to_machine_code[i] = []

        binary_data = b"".join(binary_chunks)

        if text_lines and not text_lines[-1]:
            text_lines = text_lines[:-1]

        return EncodedOutput(
            text="\n".join(text_lines),
            op_map=op_map,
            driver_data={"binary": binary_data},
        )

    def _reset_state(self) -> None:
        """Reset encoder state for a new encoding pass."""
        self.power = None
        self.cut_speed = None
        self.travel_speed = None
        self.air_assist = False
        self.current_pos = (0.0, 0.0, 0.0)
        self.active_laser = 1

    def _mm_to_um(self, mm: float) -> int:
        """Convert millimeters to micrometers."""
        return int(mm * self.UM_PER_MM)

    def _power_to_ruida(self, power_normalized: float) -> int:
        """Convert normalized power (0.0-1.0) to Ruida 14-bit value."""
        return int(power_normalized * self.POWER_SCALE) & 0x3FFF

    def _handle_command(
        self,
        ops: Ops,
        idx: int,
        machine: "Machine",
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Dispatch command to appropriate handler."""
        ct = ops.command_type(idx)

        if ct == CommandType.SET_POWER:
            self._handle_set_power(ops, idx, binary, text)
        elif ct == CommandType.SET_FEED_RATE:
            self._handle_set_cut_speed(ops, idx, binary, text)
        elif ct == CommandType.SET_RAPID_RATE:
            self._handle_set_travel_speed(ops, idx, binary, text)
        elif ct == CommandType.SET_FREQUENCY:
            self._handle_set_frequency(ops, idx, binary, text)
        elif ct == CommandType.SET_PULSE_WIDTH:
            self._handle_set_pulse_width(ops, idx, binary, text)
        elif ct == CommandType.SET_AIR_ASSIST:
            self._handle_air_assist(ops, idx, binary, text)
        elif ct == CommandType.SET_COOLANT:
            self._handle_coolant(ops, idx, binary, text)
        elif ct == CommandType.SET_HEAD:
            self._handle_set_laser(ops, idx, machine, binary, text)
        elif ct == CommandType.MOVE_TO:
            self._handle_move_to(ops, idx, binary, text)
            self.current_pos = ops.endpoint(idx)
        elif ct == CommandType.LINE_TO:
            self._handle_line_to(ops, idx, binary, text)
            self.current_pos = ops.endpoint(idx)
        elif ct == CommandType.ARC_TO:
            self._handle_arc_to(ops, idx, binary, text)
            self.current_pos = ops.endpoint(idx)
        elif ct == CommandType.SCAN_LINE:
            self._handle_scan_line(ops, idx, binary, text)
            self.current_pos = ops.endpoint(idx)
        elif ct == CommandType.JOB_START:
            self._handle_job_start(machine, binary, text)
        elif ct == CommandType.JOB_END:
            self._handle_job_end(binary, text)
        elif ct == CommandType.LAYER_START:
            self._handle_layer_start(ops, idx, binary, text)
        elif ct == CommandType.LAYER_END:
            self._handle_layer_end(ops, idx, binary, text)
        elif ct == CommandType.WORKPIECE_START:
            self._handle_workpiece_start(ops, idx, text)
        elif ct == CommandType.WORKPIECE_END:
            self._handle_workpiece_end(ops, idx, text)

    def _handle_set_power(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetPowerCommand - set laser power percentage."""
        power = ops.power(idx)
        self.power = power
        power_val = self._power_to_ruida(power)
        power_percent = power * 100.0

        laser_cmd = {1: 0xC8, 2: 0xC1, 3: 0xC4, 4: 0xC5}
        cmd_byte = laser_cmd.get(self.active_laser, 0xC8)
        binary.append(bytes([cmd_byte]) + encode14(power_val))
        text.append(f"POWER {power_percent:.1f}")

    def _handle_set_cut_speed(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetCutSpeedCommand - set cutting speed in mm/s."""
        speed = ops.rate(idx)
        self.cut_speed = speed
        speed_um = self._mm_to_um(speed)
        binary.append(b"\xc9\x02" + encode35(speed_um))
        text.append(f"SPEED {speed:.1f}")

    def _handle_set_travel_speed(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetTravelSpeedCommand - store for move operations."""
        speed = ops.rate(idx)
        self.travel_speed = speed
        if self.travel_speed is not None:
            speed_um = self._mm_to_um(speed)
            binary.append(b"\xc9\x02" + encode35(speed_um))
        text.append(f"TRAVEL_SPEED {speed:.1f}")

    def _handle_set_frequency(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetFrequencyCommand - emit 0xC6 0x60 frequency."""
        freq = ops.frequency(idx)
        binary.append(
            b"\xc6\x60" + bytes([self.active_laser, 0]) + encode35(freq)
        )
        text.append(f"FREQUENCY {freq}")

    def _handle_set_pulse_width(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetPulseWidthCommand - emit 0xC6 0x10 interval."""
        pw = ops.pulse_width(idx)
        pulse_us = int(pw)
        binary.append(
            b"\xc6\x10" + bytes([self.active_laser, 0]) + encode35(pulse_us)
        )
        text.append(f"PULSE_WIDTH {pw:.1f}")

    def _handle_air_assist(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetAirAssistCommand - update air assist state."""
        mode = ops.air_assist(idx)
        if mode == AirAssistMode.ON:
            if not self.air_assist:
                self.air_assist = True
                binary.append(b"\xca\x13")
                text.append("AIR_ASSIST ON")
        else:
            if self.air_assist:
                self.air_assist = False
                binary.append(b"\xca\x12")
                text.append("AIR_ASSIST OFF")

    def _handle_coolant(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetCoolantCommand - coolant not used on laser cutters."""

    def _handle_set_laser(
        self,
        ops: Ops,
        idx: int,
        machine: "Machine",
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetLaserCommand - select active laser head."""
        laser_uid = ops.head_uid(idx)
        laser_head = next(
            (head for head in machine.heads if head.uid == laser_uid),
            None,
        )

        if laser_head is None:
            logger.warning(
                f"Could not find laser with UID '{laser_uid}'. "
                "Using default laser 1."
            )
            self.active_laser = 1
        else:
            self.active_laser = laser_head.tool_number

        laser_select_cmd = 0xCA + self.active_laser - 1
        binary.append(bytes([0xCA, laser_select_cmd & 0x0F]))
        text.append(f"LASER {self.active_laser}")

    def _handle_move_to(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle MoveToCommand - rapid move with laser off."""
        end = ops.endpoint(idx)
        x_um = self._mm_to_um(end[0])
        y_um = self._mm_to_um(end[1])
        binary.append(b"\x88" + encode35(x_um) + encode35(y_um))
        text.append(f"MOVE_ABS X:{end[0]:.3f} Y:{end[1]:.3f}")

    def _handle_line_to(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LineToCommand - cutting move with laser on."""
        end = ops.endpoint(idx)
        x_um = self._mm_to_um(end[0])
        y_um = self._mm_to_um(end[1])
        binary.append(b"\xa8" + encode35(x_um) + encode35(y_um))
        text.append(f"CUT_ABS X:{end[0]:.3f} Y:{end[1]:.3f}")

    def _handle_arc_to(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle ArcToCommand - linearize arc to series of cuts."""
        end = ops.endpoint(idx)
        i_val, j_val, cw = ops.arc_params(idx)
        text.append(
            f"; ARC to ({end[0]:.3f}, {end[1]:.3f}) {'CW' if cw else 'CCW'}"
        )

        sub_ops = ops.linearize(idx, self.current_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                self._handle_line_to(sub_ops, j, binary, text)
            elif sub_ct == CommandType.SET_POWER:
                self._handle_set_power(sub_ops, j, binary, text)

    def _handle_scan_line(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle ScanLinePowerCommand - linearize to power/line segments."""
        end = ops.endpoint(idx)
        power_mv = ops.scanline_data(idx)
        text.append(
            f"; SCAN_LINE to ({end[0]:.3f}, {end[1]:.3f}) "
            f"({len(power_mv)} samples)"
        )

        sub_ops = ops.linearize(idx, self.current_pos)
        for j in range(sub_ops.len()):
            sub_ct = sub_ops.command_type(j)
            if sub_ct == CommandType.LINE_TO:
                self._handle_line_to(sub_ops, j, binary, text)
            elif sub_ct == CommandType.SET_POWER:
                self._handle_set_power(sub_ops, j, binary, text)

    def _handle_job_start(
        self,
        machine: "Machine",
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """
        Handle JobStartCommand - select reference point and mark job start.

        Raises:
            ValueError: If active_wcs is not a valid Ruida reference point
        """
        active_wcs = machine.active_wcs
        if active_wcs not in REF_POINT_COMMANDS:
            raise ValueError(
                f"Unknown WCS slot '{active_wcs}'. "
                f"Valid options: {', '.join(REF_POINT_COMMANDS.keys())}"
            )
        binary.append(REF_POINT_COMMANDS[active_wcs])
        text.append(f"; Job Start - Ref Point: {active_wcs}")

    def _handle_job_end(
        self,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle JobEndCommand - send end-of-file marker."""
        binary.append(b"\xd7")
        text.append("; Job End")

    def _handle_layer_start(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LayerStartCommand - mark layer beginning."""
        uid = ops.layer_uid(idx)
        binary.append(b"\xca\x00")
        text.append(f"; --- Layer {uid[:8]} ---")

    def _handle_layer_end(
        self,
        ops: Ops,
        idx: int,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LayerEndCommand - mark layer end."""
        binary.append(b"\xca\x00")
        text.append("; --- End Layer ---")

    def _handle_workpiece_start(
        self,
        ops: Ops,
        idx: int,
        text: List[str],
    ) -> None:
        """Handle WorkpieceStartCommand - mark workpiece beginning."""
        uid = ops.workpiece_uid(idx)
        text.append(f"; --- Workpiece {uid[:8]} ---")

    def _handle_workpiece_end(
        self,
        ops: Ops,
        idx: int,
        text: List[str],
    ) -> None:
        """Handle WorkpieceEndCommand - mark workpiece end."""
        text.append("; --- End Workpiece ---")
