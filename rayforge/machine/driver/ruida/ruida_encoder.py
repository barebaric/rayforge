"""
Ruida Encoder - Converts Ops commands to Ruida binary protocol.

Produces both binary output for the controller and human-readable
text representation for UI display.
"""

import logging
from typing import TYPE_CHECKING, List, Optional

from ....core.ops import (
    Ops,
    Command,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    SetLaserCommand,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    ScanLinePowerCommand,
    JobStartCommand,
    JobEndCommand,
    LayerStartCommand,
    LayerEndCommand,
    WorkpieceStartCommand,
    WorkpieceEndCommand,
)
from ....core.geo.types import Point3D
from ....pipeline.encoder.base import (
    OpsEncoder,
    MachineCodeOpMap,
    EncodedOutput,
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

        for i, cmd in enumerate(ops):
            start_line = len(text_lines)
            self._handle_command(cmd, machine, binary_chunks, text_lines)
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
        cmd: Command,
        machine: "Machine",
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Dispatch command to appropriate handler."""
        match cmd:
            case SetPowerCommand():
                self._handle_set_power(cmd, binary, text)
            case SetCutSpeedCommand():
                self._handle_set_cut_speed(cmd, binary, text)
            case SetTravelSpeedCommand():
                self._handle_set_travel_speed(cmd, binary, text)
            case EnableAirAssistCommand():
                self._handle_enable_air_assist(binary, text)
            case DisableAirAssistCommand():
                self._handle_disable_air_assist(binary, text)
            case SetLaserCommand():
                self._handle_set_laser(cmd, machine, binary, text)
            case MoveToCommand():
                self._handle_move_to(cmd, binary, text)
                self.current_pos = cmd.end
            case LineToCommand():
                self._handle_line_to(cmd, binary, text)
                self.current_pos = cmd.end
            case ArcToCommand():
                self._handle_arc_to(cmd, binary, text)
                self.current_pos = cmd.end
            case ScanLinePowerCommand():
                self._handle_scan_line(cmd, binary, text)
                self.current_pos = cmd.end
            case JobStartCommand():
                self._handle_job_start(machine, binary, text)
            case JobEndCommand():
                self._handle_job_end(binary, text)
            case LayerStartCommand():
                self._handle_layer_start(cmd, binary, text)
            case LayerEndCommand():
                self._handle_layer_end(cmd, binary, text)
            case WorkpieceStartCommand():
                self._handle_workpiece_start(cmd, text)
            case WorkpieceEndCommand():
                self._handle_workpiece_end(cmd, text)

    def _handle_set_power(
        self,
        cmd: SetPowerCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetPowerCommand - set laser power percentage."""
        self.power = cmd.power
        power_val = self._power_to_ruida(cmd.power)
        power_percent = cmd.power * 100.0

        laser_cmd = {1: 0xC8, 2: 0xC1, 3: 0xC4, 4: 0xC5}
        cmd_byte = laser_cmd.get(self.active_laser, 0xC8)
        binary.append(bytes([cmd_byte]) + encode14(power_val))
        text.append(f"POWER {power_percent:.1f}")

    def _handle_set_cut_speed(
        self,
        cmd: SetCutSpeedCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetCutSpeedCommand - set cutting speed in mm/s."""
        self.cut_speed = cmd.speed
        speed_um = self._mm_to_um(cmd.speed)
        binary.append(b"\xc9\x02" + encode35(speed_um))
        text.append(f"SPEED {cmd.speed:.1f}")

    def _handle_set_travel_speed(
        self,
        cmd: SetTravelSpeedCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetTravelSpeedCommand - store for move operations."""
        self.travel_speed = cmd.speed
        if self.travel_speed is not None:
            speed_um = self._mm_to_um(cmd.speed)
            binary.append(b"\xc9\x02" + encode35(speed_um))
        text.append(f"TRAVEL_SPEED {cmd.speed:.1f}")

    def _handle_enable_air_assist(
        self,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle EnableAirAssistCommand - turn on air assist."""
        if not self.air_assist:
            self.air_assist = True
            binary.append(b"\xca\x13")
            text.append("AIR_ASSIST ON")

    def _handle_disable_air_assist(
        self,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle DisableAirAssistCommand - turn off air assist."""
        if self.air_assist:
            self.air_assist = False
            binary.append(b"\xca\x12")
            text.append("AIR_ASSIST OFF")

    def _handle_set_laser(
        self,
        cmd: SetLaserCommand,
        machine: "Machine",
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle SetLaserCommand - select active laser head."""
        laser_head = next(
            (head for head in machine.heads if head.uid == cmd.laser_uid),
            None,
        )

        if laser_head is None:
            logger.warning(
                f"Could not find laser with UID '{cmd.laser_uid}'. "
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
        cmd: MoveToCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle MoveToCommand - rapid move with laser off."""
        x_um = self._mm_to_um(cmd.end[0])
        y_um = self._mm_to_um(cmd.end[1])
        binary.append(b"\x88" + encode35(x_um) + encode35(y_um))
        text.append(f"MOVE_ABS X:{cmd.end[0]:.3f} Y:{cmd.end[1]:.3f}")

    def _handle_line_to(
        self,
        cmd: LineToCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LineToCommand - cutting move with laser on."""
        x_um = self._mm_to_um(cmd.end[0])
        y_um = self._mm_to_um(cmd.end[1])
        binary.append(b"\xa8" + encode35(x_um) + encode35(y_um))
        text.append(f"CUT_ABS X:{cmd.end[0]:.3f} Y:{cmd.end[1]:.3f}")

    def _handle_arc_to(
        self,
        cmd: ArcToCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle ArcToCommand - linearize arc to series of cuts."""
        text.append(
            f"; ARC to ({cmd.end[0]:.3f}, {cmd.end[1]:.3f}) "
            f"{'CW' if cmd.clockwise else 'CCW'}"
        )

        linear_cmds = cmd.linearize(self.current_pos)
        for linear_cmd in linear_cmds:
            if isinstance(linear_cmd, LineToCommand):
                self._handle_line_to(linear_cmd, binary, text)
            elif isinstance(linear_cmd, SetPowerCommand):
                self._handle_set_power(linear_cmd, binary, text)

    def _handle_scan_line(
        self,
        cmd: ScanLinePowerCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle ScanLinePowerCommand - linearize to power/line segments."""
        text.append(
            f"; SCAN_LINE to ({cmd.end[0]:.3f}, {cmd.end[1]:.3f}) "
            f"({len(cmd.power_values)} samples)"
        )

        linear_cmds = cmd.linearize(self.current_pos)
        for linear_cmd in linear_cmds:
            if isinstance(linear_cmd, LineToCommand):
                self._handle_line_to(linear_cmd, binary, text)
            elif isinstance(linear_cmd, SetPowerCommand):
                self._handle_set_power(linear_cmd, binary, text)

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
        cmd: LayerStartCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LayerStartCommand - mark layer beginning."""
        binary.append(b"\xca\x00")
        text.append(f"; --- Layer {cmd.layer_uid[:8]} ---")

    def _handle_layer_end(
        self,
        cmd: LayerEndCommand,
        binary: List[bytes],
        text: List[str],
    ) -> None:
        """Handle LayerEndCommand - mark layer end."""
        binary.append(b"\xca\x00")
        text.append("; --- End Layer ---")

    def _handle_workpiece_start(
        self,
        cmd: WorkpieceStartCommand,
        text: List[str],
    ) -> None:
        """Handle WorkpieceStartCommand - mark workpiece beginning."""
        text.append(f"; --- Workpiece {cmd.workpiece_uid[:8]} ---")

    def _handle_workpiece_end(
        self,
        cmd: WorkpieceEndCommand,
        text: List[str],
    ) -> None:
        """Handle WorkpieceEndCommand - mark workpiece end."""
        text.append("; --- End Workpiece ---")
