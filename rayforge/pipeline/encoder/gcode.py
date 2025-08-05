from typing import TYPE_CHECKING
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
)
from ...machine.models.dialect import GcodeDialect, get_dialect
from .base import OpsEncoder
if TYPE_CHECKING:
    from ...machine.models.machine import Machine


class GcodeEncoder(OpsEncoder):
    """Converts Ops commands to G-code using instance state tracking"""

    def __init__(self, dialect: GcodeDialect):
        self.dialect = dialect
        self.power = None  # Current laser power (None = off)
        self.cut_speed = None  # Current cutting speed (mm/min)
        self.travel_speed = None  # Current travel speed (mm/min)
        self.air_assist = False  # Air assist state
        self.laser_active = False  # Laser on/off state

    @classmethod
    def for_machine(cls, machine: "Machine") -> "GcodeEncoder":
        """
        Factory method to create a GcodeEncoder instance configured for a
        specific machine's dialect.
        """
        dialect = get_dialect(machine.dialect_name)
        return cls(dialect)

    def encode(self, ops: Ops, machine: "Machine") -> str:
        """Main encoding workflow"""
        preamble = (
            machine.preamble
            if machine.use_custom_preamble
            else self.dialect.default_preamble
        )
        postscript = (
            machine.postscript
            if machine.use_custom_postscript
            else self.dialect.default_postscript
        )

        gcode = [] + preamble
        for cmd in ops:
            self._handle_command(gcode, cmd, machine)
        self._finalize(gcode, postscript)
        return "\n".join(gcode)

    def _handle_command(self, gcode: list, cmd: Command, machine: "Machine"):
        """Dispatch command to appropriate handler"""
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
                    gcode, *cmd.end, *cmd.center_offset, cmd.clockwise
                )

    def _update_power(self, gcode: list, power: float, machine: "Machine"):
        """
        Updates the target power. If power is set to 0 while the laser is
        active, it will be turned off. This method does NOT turn the laser on.
        """
        self.power = min(power, machine.heads[0].max_power)
        if self.laser_active and self.power <= 0:
            self._laser_off(gcode)

    def _set_air_assist(self, gcode: list, state: bool):
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

    def _handle_move_to(self, gcode: list, x: float, y: float):
        """Rapid movement with laser safety"""
        self._laser_off(gcode)
        if self.travel_speed:
            cmd = self.dialect.travel_move_with_speed.format(
                x=x, y=y, speed=self.travel_speed
            )
        else:
            cmd = self.dialect.travel_move.format(x=x, y=y)
        gcode.append(cmd)

    def _handle_line_to(self, gcode: list, x: float, y: float):
        """Cutting movement with laser activation"""
        self._laser_on(gcode)
        if self.cut_speed:
            cmd = self.dialect.linear_move_with_speed.format(
                x=x, y=y, speed=self.cut_speed
            )
        else:
            cmd = self.dialect.linear_move.format(x=x, y=y)
        gcode.append(cmd)

    def _handle_arc_to(
        self,
        gcode: list,
        x: float,
        y: float,
        i: float,
        j: float,
        clockwise: bool,
    ):
        """Cutting movement with laser activation"""
        self._laser_on(gcode)
        if clockwise:
            template = (
                self.dialect.arc_cw_with_speed
                if self.cut_speed
                else self.dialect.arc_cw
            )
        else:
            template = (
                self.dialect.arc_ccw_with_speed
                if self.cut_speed
                else self.dialect.arc_ccw
            )
        cmd = template.format(x=x, y=y, i=i, j=j, speed=self.cut_speed)
        gcode.append(cmd)

    def _laser_on(self, gcode: list):
        """Activate laser if not already on"""
        if not self.laser_active and self.power:
            power_val = self.dialect.format_laser_power(self.power)
            gcode.append(self.dialect.laser_on.format(power=power_val))
            self.laser_active = True

    def _laser_off(self, gcode: list):
        """Deactivate laser if active"""
        if self.laser_active:
            gcode.append(self.dialect.laser_off)
            self.laser_active = False

    def _finalize(self, gcode: list, postscript: list):
        """Cleanup at end of file"""
        self._laser_off(gcode)
        if self.air_assist and self.dialect.air_assist_off:
            gcode.append(self.dialect.air_assist_off)
        gcode.extend(postscript)
        gcode.append("")
