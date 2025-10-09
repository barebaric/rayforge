import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict
from ...core.ops import (
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
from ...machine.models.dialect import GcodeDialect, get_dialect
from ...machine.models.script import ScriptTrigger
from ...shared.util.template import TemplateFormatter
from .base import OpsEncoder
from .context import GcodeContext, JobInfo
from ...core.layer import Layer
from ...core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)


@dataclass
class GcodeOpMap:
    """
    A container for a bidirectional mapping between Ops command indices and
    G-code line numbers.

    Attributes:
        op_to_gcode: Maps an Ops command index to a list of G-code line
                     numbers it generated. An empty list means the command
                     produced no G-code.
        gcode_to_op: Maps a G-code line number back to the Ops command
                     index that generated it.
    """

    op_to_gcode: Dict[int, List[int]] = field(default_factory=dict)
    gcode_to_op: Dict[int, int] = field(default_factory=dict)


class GcodeEncoder(OpsEncoder):
    """Converts Ops commands to G-code using instance state tracking"""

    def __init__(self, dialect: GcodeDialect):
        """
        Initializes the GcodeEncoder.

        Args:
            dialect: The G-code dialect configuration to use for encoding.
        """
        self.dialect: GcodeDialect = dialect
        self.power: Optional[float] = None  # Current laser power (None = off)
        self.cut_speed: Optional[float] = None  # Current cut speed (mm/min)
        self.travel_speed: Optional[float] = (
            None  # Current travel speed (mm/min)
        )
        self.emitted_speed: Optional[float] = (
            None  # Last speed sent to the controller
        )
        self.air_assist: bool = False  # Air assist state
        self.laser_active: bool = False  # Laser on/off state
        self.active_laser_uid: Optional[str] = None
        self.current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._coord_format: str = "{:.3f}"  # Default format

    @classmethod
    def for_machine(cls, machine: "Machine") -> "GcodeEncoder":
        """
        Factory method to create a GcodeEncoder instance configured for a
        specific machine's dialect.
        """
        dialect = get_dialect(machine.dialect_name)
        return cls(dialect)

    def _get_current_laser_head(self, context: GcodeContext):
        if not self.active_laser_uid:
            raise ValueError("No active laser head is set")
        current_laser = context.machine.get_head_by_uid(self.active_laser_uid)
        if not current_laser:
            raise ValueError(
                f"Laser head with UID {self.active_laser_uid} not found"
            )
        return current_laser

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> Tuple[str, GcodeOpMap]:
        """Main encoding workflow"""
        # Set the coordinate format based on the machine's precision setting
        self._coord_format = f"{{:.{machine.gcode_precision}f}}"
        self.current_pos = (0.0, 0.0, 0.0)
        self.active_laser_uid = None

        context = GcodeContext(
            machine=machine, doc=doc, job=JobInfo(extents=ops.rect())
        )
        gcode: List[str] = []
        op_map = GcodeOpMap()

        # Include a bi-directional map from ops to line number.
        # Since this is an n:n mapping, this needs to be stored as
        # two separate maps.
        for i, cmd in enumerate(ops):
            start_line = len(gcode)
            self._handle_command(gcode, cmd, context)
            end_line = len(gcode)

            if end_line > start_line:
                line_indices = list(range(start_line, end_line))
                op_map.op_to_gcode[i] = line_indices
                for line_num in line_indices:
                    op_map.gcode_to_op[line_num] = i
            else:
                op_map.op_to_gcode[i] = []

        self._finalize(gcode)
        return "\n".join(gcode), op_map

    def _emit_scripts(
        self, context: GcodeContext, gcode: List[str], trigger: ScriptTrigger
    ):
        """
        Finds the script for a trigger and uses the TemplateFormatter to
        expand it.
        """
        script_action = context.machine.hookscripts.get(trigger)

        if script_action and script_action.enabled:
            formatter = TemplateFormatter(context.machine, context)
            expanded_lines = formatter.expand_script(script_action)
            gcode.extend(expanded_lines)
            return

        # If we get here, no user scripts were found, so use defaults.
        if trigger == ScriptTrigger.JOB_START:
            gcode.extend(self.dialect.default_preamble)
        elif trigger == ScriptTrigger.JOB_END:
            gcode.extend(self.dialect.default_postscript)

    def _handle_command(
        self, gcode: List[str], cmd: Command, context: GcodeContext
    ) -> None:
        """Dispatch command to appropriate handler"""
        match cmd:
            case SetPowerCommand():
                self._update_power(context, gcode, cmd.power)
            case SetCutSpeedCommand():
                # We limit to max travel speed, not max cut speed, to
                # allow framing operations to go faster. Cut limits should
                # should be kept by ensuring an Ops object is created
                # with limits in mind.
                self.cut_speed = min(
                    cmd.speed, context.machine.max_travel_speed
                )
            case SetTravelSpeedCommand():
                self.travel_speed = min(
                    cmd.speed, context.machine.max_travel_speed
                )
            case EnableAirAssistCommand():
                self._set_air_assist(context, gcode, True)
            case DisableAirAssistCommand():
                self._set_air_assist(context, gcode, False)
            case SetLaserCommand():
                self._handle_set_laser(context, gcode, cmd.laser_uid)
            case MoveToCommand():
                self._handle_move_to(context, gcode, *cmd.end)
                self.current_pos = cmd.end
            case LineToCommand():
                self._handle_line_to(context, gcode, *cmd.end)
                self.current_pos = cmd.end
            case ScanLinePowerCommand():
                # Deconstruct into simpler commands that the encoder already
                # understands.
                sub_commands = cmd.linearize(self.current_pos)
                for sub_cmd in sub_commands:
                    self._handle_command(gcode, sub_cmd, context)
                # To avoid float precision errors, explicitly set the final pos
                self.current_pos = cmd.end
            case ArcToCommand():
                self._handle_arc_to(
                    context, gcode, cmd.end, cmd.center_offset, cmd.clockwise
                )
                self.current_pos = cmd.end
            case JobStartCommand():
                self._emit_scripts(context, gcode, ScriptTrigger.JOB_START)
            case JobEndCommand():
                # This is the single point of truth for job cleanup.
                # First, perform guaranteed safety shutdowns. This emits the
                # first M5 and updates the internal state.
                self._laser_off(context, gcode)
                if self.air_assist:
                    self._set_air_assist(context, gcode, False)

                # Then, run the user script or the full default postscript.
                self._emit_scripts(context, gcode, ScriptTrigger.JOB_END)
            case LayerStartCommand(layer_uid=uid):
                descendant = context.doc.find_descendant_by_uid(uid)
                if isinstance(descendant, Layer):
                    context.layer = descendant
                elif descendant is not None:
                    logger.warning(
                        f"Expected Layer for UID {uid}, but "
                        f" found {type(descendant)}"
                    )
                self._emit_scripts(context, gcode, ScriptTrigger.LAYER_START)
            case LayerEndCommand():
                self._emit_scripts(context, gcode, ScriptTrigger.LAYER_END)
                context.layer = None
            case WorkpieceStartCommand(workpiece_uid=uid):
                descendant = context.doc.find_descendant_by_uid(uid)
                if isinstance(descendant, WorkPiece):
                    context.workpiece = descendant
                elif descendant is not None:
                    logger.warning(
                        f"Expected WorkPiece for UID {uid}, "
                        f" but found {type(descendant)}"
                    )
                self._emit_scripts(
                    context, gcode, ScriptTrigger.WORKPIECE_START
                )
            case WorkpieceEndCommand():
                self._emit_scripts(context, gcode, ScriptTrigger.WORKPIECE_END)
                context.workpiece = None

    def _emit_modal_speed(self, gcode: List[str], speed: float) -> None:
        """
        Emits a modal speed command if the dialect supports it and speed
        has changed.
        """
        if self.dialect.set_speed and speed != self.emitted_speed:
            gcode.append(self.dialect.set_speed.format(speed=speed))
            self.emitted_speed = speed

    def _handle_set_laser(
        self, context: GcodeContext, gcode: List[str], laser_uid: str
    ):
        """Handles a SetLaserCommand by emitting a tool change command."""
        if self.active_laser_uid == laser_uid:
            return

        laser_head = next(
            (head for head in context.machine.heads if head.uid == laser_uid),
            None,
        )

        if laser_head is None:
            logger.warning(
                f"Could not find laser with UID '{laser_uid}' on the "
                "current machine. Tool change command will not be emitted."
            )
            return

        cmd_str = self.dialect.tool_change.format(
            tool_number=laser_head.tool_number
        )
        gcode.append(cmd_str)
        self.active_laser_uid = laser_uid

    def _update_power(
        self, context: GcodeContext, gcode: List[str], power: float
    ) -> None:
        """
        Updates the target power. If power is set to 0 while the laser is
        active, it will be turned off. This method does NOT turn the laser on,
        but it WILL update the power level if the laser is already on.
        """
        # Avoid emitting redundant power commands
        if self.power is not None and math.isclose(power, self.power):
            return
        self.power = power

        if self.laser_active:
            if self.power > 0:
                # Find the currently active laser head to get its max power
                current_laser = self._get_current_laser_head(context)
                power_abs = power * current_laser.max_power
                power_val = self.dialect.format_laser_power(power_abs)
                gcode.append(self.dialect.laser_on.format(power=power_val))
            else:  # power <= 0
                self._laser_off(context, gcode)

    def _set_air_assist(
        self, context: GcodeContext, gcode: List[str], state: bool
    ) -> None:
        """Update air assist state with dialect commands"""
        if self.air_assist == state:
            return
        self.air_assist = state
        cmd = (
            self.dialect.air_assist_on
            if state
            else self.dialect.air_assist_off
        )
        if cmd:
            gcode.append(cmd)

    def _handle_move_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Rapid movement with laser safety"""
        self._laser_off(context, gcode)
        self._emit_modal_speed(gcode, self.travel_speed or 0)
        f_command = self.dialect.format_feedrate(self.travel_speed)
        gcode.append(
            self.dialect.travel_move.format(
                x=self._coord_format.format(x),
                y=self._coord_format.format(y),
                z=self._coord_format.format(z),
                f_command=f_command,
            )
        )

    def _handle_line_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Cutting movement with laser activation"""
        self._laser_on(context, gcode)
        self._emit_modal_speed(gcode, self.cut_speed or 0)
        f_command = self.dialect.format_feedrate(self.cut_speed)
        gcode.append(
            self.dialect.linear_move.format(
                x=self._coord_format.format(x),
                y=self._coord_format.format(y),
                z=self._coord_format.format(z),
                f_command=f_command,
            )
        )

    def _handle_arc_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        end: Tuple[float, float, float],
        center: Tuple[float, float],
        cw: bool,
    ) -> None:
        """Cutting arc with laser activation"""
        self._laser_on(context, gcode)
        self._emit_modal_speed(gcode, self.cut_speed or 0)
        x, y, z = end
        i, j = center
        template = self.dialect.arc_cw if cw else self.dialect.arc_ccw
        f_command = self.dialect.format_feedrate(self.cut_speed)
        gcode.append(
            template.format(
                x=self._coord_format.format(x),
                y=self._coord_format.format(y),
                z=self._coord_format.format(z),
                i=self._coord_format.format(i),
                j=self._coord_format.format(j),
                f_command=f_command,
            )
        )

    def _laser_on(self, context: GcodeContext, gcode: List[str]) -> None:
        """Activate laser if not already on"""
        if not self.laser_active and self.power:
            current_laser = self._get_current_laser_head(context)
            power_abs = self.power * current_laser.max_power
            power_val = self.dialect.format_laser_power(power_abs)
            gcode.append(self.dialect.laser_on.format(power=power_val))
            self.laser_active = True

    def _laser_off(self, context: GcodeContext, gcode: List[str]) -> None:
        """Deactivate laser if active"""
        if self.laser_active:
            gcode.append(self.dialect.laser_off)
            self.laser_active = False

    def _finalize(self, gcode: List[str]) -> None:
        """Ensures the G-code file ends with a newline."""
        if not gcode or gcode[-1]:
            gcode.append("")
