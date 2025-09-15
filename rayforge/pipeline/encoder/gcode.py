import logging
from typing import TYPE_CHECKING, Optional, List, Tuple
from ...core.ops import (
    Ops,
    Command,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
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
        self._coord_format: str = "{:.3f}"  # Default format

    @classmethod
    def for_machine(cls, machine: "Machine") -> "GcodeEncoder":
        """
        Factory method to create a GcodeEncoder instance configured for a
        specific machine's dialect.
        """
        dialect = get_dialect(machine.dialect_name)
        return cls(dialect)

    def encode(self, ops: Ops, machine: "Machine", doc: "Doc") -> str:
        """Main encoding workflow"""
        # Set the coordinate format based on the machine's precision setting
        self._coord_format = f"{{:.{machine.gcode_precision}f}}"

        context = GcodeContext(
            machine=machine, doc=doc, job=JobInfo(extents=ops.rect())
        )
        gcode: List[str] = []
        for cmd in ops:
            self._handle_command(gcode, cmd, context)
        self._finalize(gcode)
        return "\n".join(gcode)

    def _emit_scripts(
        self, gcode: List[str], context: GcodeContext, trigger: ScriptTrigger
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
        machine = context.machine
        doc = context.doc
        match cmd:
            case SetPowerCommand():
                self._update_power(gcode, cmd.power, machine)
            case SetCutSpeedCommand():
                # We limit to max travel speed, not max cut speed, to
                # allow framing operations to go faster. Cut limits should
                # should be kept by ensuring an Ops object is created
                # with limits in mind.
                self.cut_speed = min(cmd.speed, machine.max_travel_speed)
            case SetTravelSpeedCommand():
                self.travel_speed = min(cmd.speed, machine.max_travel_speed)
            case EnableAirAssistCommand():
                self._set_air_assist(gcode, True)
            case DisableAirAssistCommand():
                self._set_air_assist(gcode, False)
            case MoveToCommand():
                self._handle_move_to(gcode, *cmd.end)
            case LineToCommand():
                self._handle_line_to(gcode, *cmd.end)
            case ArcToCommand():
                self._handle_arc_to(
                    gcode, cmd.end, cmd.center_offset, cmd.clockwise
                )
            case JobStartCommand():
                self._emit_scripts(gcode, context, ScriptTrigger.JOB_START)
            case JobEndCommand():
                # This is the single point of truth for job cleanup.
                # First, perform guaranteed safety shutdowns. This emits the
                # first M5 and updates the internal state.
                self._laser_off(gcode)
                if self.air_assist:
                    self._set_air_assist(gcode, False)

                # Then, run the user script or the full default postscript.
                self._emit_scripts(gcode, context, ScriptTrigger.JOB_END)
            case LayerStartCommand(layer_uid=uid):
                descendant = doc.find_descendant_by_uid(uid)
                if isinstance(descendant, Layer):
                    context.layer = descendant
                elif descendant is not None:
                    logger.warning(
                        f"Expected Layer for UID {uid}, but "
                        f" found {type(descendant)}"
                    )
                self._emit_scripts(gcode, context, ScriptTrigger.LAYER_START)
            case LayerEndCommand():
                self._emit_scripts(gcode, context, ScriptTrigger.LAYER_END)
                context.layer = None
            case WorkpieceStartCommand(workpiece_uid=uid):
                descendant = doc.find_descendant_by_uid(uid)
                if isinstance(descendant, WorkPiece):
                    context.workpiece = descendant
                elif descendant is not None:
                    logger.warning(
                        f"Expected WorkPiece for UID {uid}, "
                        f" but found {type(descendant)}"
                    )
                self._emit_scripts(
                    gcode, context, ScriptTrigger.WORKPIECE_START
                )
            case WorkpieceEndCommand():
                self._emit_scripts(gcode, context, ScriptTrigger.WORKPIECE_END)
                context.workpiece = None

    def _emit_modal_speed(self, gcode: List[str], speed: float) -> None:
        """
        Emits a modal speed command if the dialect supports it and speed
        has changed.
        """
        if self.dialect.set_speed and speed != self.emitted_speed:
            gcode.append(self.dialect.set_speed.format(speed=speed))
            self.emitted_speed = speed

    def _update_power(
        self, gcode: List[str], power: float, machine: "Machine"
    ) -> None:
        """
        Updates the target power. If power is set to 0 while the laser is
        active, it will be turned off. This method does NOT turn the laser on.
        """
        self.power = min(power, machine.heads[0].max_power)
        if self.laser_active and self.power <= 0:
            self._laser_off(gcode)

    def _set_air_assist(self, gcode: List[str], state: bool) -> None:
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
        self, gcode: List[str], x: float, y: float, z: float
    ) -> None:
        """Rapid movement with laser safety"""
        self._laser_off(gcode)
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
        self, gcode: List[str], x: float, y: float, z: float
    ) -> None:
        """Cutting movement with laser activation"""
        self._laser_on(gcode)
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
        gcode: List[str],
        end: Tuple[float, float, float],
        center: Tuple[float, float],
        cw: bool,
    ) -> None:
        """Cutting arc with laser activation"""
        self._laser_on(gcode)
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

    def _laser_on(self, gcode: List[str]) -> None:
        """Activate laser if not already on"""
        if not self.laser_active and self.power:
            power_val = self.dialect.format_laser_power(self.power)
            gcode.append(self.dialect.laser_on.format(power=power_val))
            self.laser_active = True

    def _laser_off(self, gcode: List[str]) -> None:
        """Deactivate laser if active"""
        if self.laser_active:
            gcode.append(self.dialect.laser_off)
            self.laser_active = False

    def _finalize(self, gcode: List[str]) -> None:
        """Ensures the G-code file ends with a newline."""
        if not gcode or gcode[-1]:
            gcode.append("")
