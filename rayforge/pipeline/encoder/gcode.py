import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandCategory, CommandType

from ...core.layer import Layer
from ...core.workpiece import WorkPiece
from ...machine.models.dialect import GcodeDialect
from ...machine.models.macro import MacroTrigger
from ...shared.util.template import TemplateFormatter
from .base import EncodedOutput, MachineCodeOpMap, OpsEncoder
from .context import GcodeContext, JobInfo

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)

_COORD_TOLERANCE = 1e-6


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
        self.emitted_power: Optional[float] = (
            None  # Last power sent to the controller
        )
        self._emitted_cut_feed: Optional[float] = None
        self.air_assist: bool = False  # Air assist state
        self.laser_active: bool = False  # Laser on/off state
        self.active_laser_uid: Optional[str] = None
        self.frequency: Optional[int] = None
        self.pulse_width: Optional[float] = None
        self.current_pos: Dict[Axis, float] = {
            Axis.X: 0.0,
            Axis.Y: 0.0,
            Axis.Z: 0.0,
        }
        self._coord_format: str = "{:.3f}"  # Default format
        self._feed_format: str = "{:.3f}"
        self._power_format: str = "{:.3f}"
        self._active_wcs: Optional[str] = None

    @classmethod
    def for_machine(cls, machine: "Machine") -> "GcodeEncoder":
        """
        Factory method to create a GcodeEncoder instance configured for a
        specific machine's dialect.
        Note: Precision is set in the `encode` method, which receives the
        machine instance.
        """
        assert machine.dialect is not None
        return cls(machine.dialect)

    def _get_current_laser_head(self, context: GcodeContext):
        if not self.active_laser_uid:
            default_head = context.machine.get_default_head()
            self.active_laser_uid = default_head.uid
            logger.debug(
                "No active laser set, defaulting to machine's first head: "
                f"{self.active_laser_uid}"
            )

        current_laser = context.machine.get_head_by_uid(self.active_laser_uid)
        if not current_laser:
            raise ValueError(
                f"Laser head with UID {self.active_laser_uid} not found"
            )
        return current_laser

    def _format_coord(self, value: float) -> str:
        """
        Format a coordinate value with the specified precision,
        stripping unnecessary trailing zeros.

        Args:
            value: The float value to format.

        Returns:
            The formatted string with trailing zeros removed.
        """
        if abs(value) < _COORD_TOLERANCE:
            value = 0.0
        formatted = self._coord_format.format(value)
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    def _format_feed(self, value: float) -> str:
        """
        Format a feedrate value with the specified precision,
        stripping unnecessary trailing zeros.

        Args:
            value: The float value to format.

        Returns:
            The formatted string with trailing zeros removed.
        """
        formatted = self._feed_format.format(value)
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    def _format_power(self, value: float) -> str:
        """
        Format a power value with the specified precision,
        stripping unnecessary trailing zeros.

        Args:
            value: The float value to format.

        Returns:
            The formatted string with trailing zeros removed.
        """
        formatted = self._power_format.format(value)
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted

    def _build_coord_commands(
        self,
        x: float,
        y: float,
        z: float,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> dict:
        """
        Build coordinate template variables, with conditional commands
        that omit unchanged axes when omit_unchanged_coords is enabled.

        Args:
            x: Target X coordinate.
            y: Target Y coordinate.
            z: Target Z coordinate.
            extra_axes: Optional dict of additional axis values (e.g. A, B, C).

        Returns:
            A dict with 'x', 'y', 'z' (always populated) and
            'x_cmd', 'y_cmd', 'z_cmd', 'extra_cmd' (conditional,
            include axis letter).
        """
        prev_x = self.current_pos.get(Axis.X, 0.0)
        prev_y = self.current_pos.get(Axis.Y, 0.0)
        prev_z = self.current_pos.get(Axis.Z, 0.0)

        x_cmd = f" X{self._format_coord(x)}"
        y_cmd = f" Y{self._format_coord(y)}"
        z_cmd = f" Z{self._format_coord(z)}"

        extra_cmd = ""
        if extra_axes:
            parts = []
            for axis, value in extra_axes.items():
                letter = axis.name
                prev_val = self.current_pos.get(axis, 0.0)
                formatted = f" {letter}{self._format_coord(value)}"
                if self.dialect.omit_unchanged_coords and math.isclose(
                    value, prev_val
                ):
                    formatted = ""
                parts.append(formatted)
            extra_cmd = "".join(parts)

        if self.dialect.omit_unchanged_coords:
            x_changed = not math.isclose(x, prev_x)
            y_changed = not math.isclose(y, prev_y)
            z_changed = not math.isclose(z, prev_z)
            none_changed = not (x_changed or y_changed or z_changed)

            x_cmd = x_cmd if x_changed or none_changed else ""
            y_cmd = y_cmd if y_changed else ""
            z_cmd = z_cmd if z_changed else ""

        return {
            "x": self._format_coord(x),
            "y": self._format_coord(y),
            "z": self._format_coord(z),
            "x_cmd": x_cmd,
            "y_cmd": y_cmd,
            "z_cmd": z_cmd,
            "extra_cmd": extra_cmd,
        }

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        """Main encoding workflow"""
        # Set coordinate and feedrate format based on the machine's precision
        self._coord_format = f"{{:.{machine.gcode_precision}f}}"
        self._feed_format = self._coord_format
        self._power_format = self._coord_format
        self.current_pos = {Axis.X: 0.0, Axis.Y: 0.0, Axis.Z: 0.0}
        self.active_laser_uid = None
        self.emitted_power = None
        self._emitted_cut_feed = None
        self.frequency = None
        self.pulse_width = None

        context = GcodeContext(
            machine=machine, doc=doc, job=JobInfo(extents=ops.rect())
        )
        gcode: List[str] = []
        op_map = MachineCodeOpMap()

        # Include a bi-directional map from ops to line number.
        # Since this is an n:n mapping, this needs to be stored as
        # two separate maps.
        for i in range(ops.len()):
            start_line = len(gcode)
            self._handle_command(gcode, ops, i, context)
            end_line = len(gcode)

            if end_line > start_line:
                line_indices = list(range(start_line, end_line))
                op_map.op_to_machine_code[i] = line_indices
                for line_num in line_indices:
                    op_map.machine_code_to_op[line_num] = i
            else:
                op_map.op_to_machine_code[i] = []

        self._finalize(gcode)
        return EncodedOutput(text="\n".join(gcode), op_map=op_map)

    def _emit_macros(
        self, context: GcodeContext, gcode: List[str], trigger: MacroTrigger
    ):
        """
        Finds the macro for a trigger and uses the TemplateFormatter to
        expand it.
        """
        macro_action = context.machine.hookmacros.get(trigger)

        if macro_action and macro_action.enabled:
            formatter = TemplateFormatter(context.machine, context)
            expanded_lines = formatter.expand_macro(macro_action)
            gcode.extend(expanded_lines)

    def _format_script_lines(
        self, lines: List[str], context: GcodeContext
    ) -> List[str]:
        """
        Formats a list of script lines by applying template substitution.

        Args:
            lines: The list of script lines to format.
            context: The GcodeContext containing job, machine, and doc info.

        Returns:
            A list of formatted script lines with placeholders replaced.
        """
        formatter = TemplateFormatter(context.machine, context)
        return [formatter.format_string(line) for line in lines]

    def _update_current_pos(self, ops: Ops, idx: int) -> None:
        end = ops.endpoint(idx)
        self.current_pos[Axis.X] = end[0]
        self.current_pos[Axis.Y] = end[1]
        self.current_pos[Axis.Z] = end[2]
        ea = ops.extra_axes(idx)
        if ea:
            for axis, value in ea.items():
                self.current_pos[axis] = value

    def _current_pos_xyz(self) -> Point3D:
        return (
            self.current_pos.get(Axis.X, 0.0),
            self.current_pos.get(Axis.Y, 0.0),
            self.current_pos.get(Axis.Z, 0.0),
        )

    def _handle_command(
        self,
        gcode: List[str],
        ops: Ops,
        idx: int,
        context: GcodeContext,
    ) -> None:
        """Dispatch command to appropriate handler"""
        ct = ops.command_type(idx)
        cat = ops.category(idx)

        if ct == CommandType.SET_POWER:
            self._update_power(context, gcode, ops.power(idx))
        elif ct == CommandType.SET_FEED_RATE:
            # Cut speed limits are the responsibility of the Ops
            # producer. The encoder trusts the value it receives.
            self.cut_speed = ops.rate(idx)
        elif ct == CommandType.SET_RAPID_RATE:
            self.travel_speed = min(
                ops.rate(idx), context.machine.max_travel_speed
            )
        elif ct == CommandType.SET_FREQUENCY:
            self.frequency = ops.frequency(idx)
        elif ct == CommandType.SET_PULSE_WIDTH:
            self.pulse_width = ops.pulse_width(idx)
        elif ct == CommandType.SET_AIR_ASSIST:
            self._handle_air_assist(context, gcode, ops.air_assist(idx))
        elif ct == CommandType.SET_COOLANT:
            self._handle_coolant(context, gcode, ops.coolant(idx))
        elif ct == CommandType.SET_HEAD:
            self._handle_set_laser(context, gcode, ops.head_uid(idx))
        elif cat == CommandCategory.MOVING:
            self._handle_moving(gcode, ops, idx, ct, context)
        elif ct == CommandType.DWELL:
            if self.dialect.dwell:
                duration = ops.dwell_duration(idx)
                gcode.append(
                    self.dialect.dwell.format(
                        seconds=duration / 1000.0,
                        milliseconds=duration,
                    )
                )
        elif ct == CommandType.JOB_START:
            # 1. Emit Preamble
            gcode.extend(
                self._format_script_lines(self.dialect.preamble, context)
            )

            # 2. Inject Active WCS command. This is done AFTER the
            # preamble to guarantee the correct coordinate system is
            # active for the job, treating the preamble as a black box
            # that may have changed state.
            if self.dialect.inject_wcs_after_preamble:
                gcode.append(context.machine.active_wcs)
                self._active_wcs = context.machine.active_wcs

        elif ct == CommandType.JOB_END:
            # This is the single point of truth for job cleanup.
            # First, perform guaranteed safety shutdowns. This emits the
            # first M5 and updates the internal state.
            self._laser_off(context, gcode)
            if self.air_assist:
                self._handle_air_assist(context, gcode, "Off")
            gcode.extend(
                self._format_script_lines(self.dialect.postscript, context)
            )
        elif ct == CommandType.LAYER_START:
            uid = ops.layer_uid(idx)
            descendant = context.doc.find_descendant_by_uid(uid)
            if isinstance(descendant, Layer):
                context.layer = descendant
                layer_wcs = descendant.get_effective_wcs(context.machine)
                if (
                    self.dialect.inject_wcs_after_preamble
                    and layer_wcs != self._active_wcs
                ):
                    gcode.append(layer_wcs)
                    self._active_wcs = layer_wcs
            elif descendant is not None:
                logger.warning(
                    f"Expected Layer for UID {uid}, but "
                    f" found {type(descendant)}"
                )
            self._emit_macros(context, gcode, MacroTrigger.LAYER_START)
        elif ct == CommandType.LAYER_END:
            self._emit_macros(context, gcode, MacroTrigger.LAYER_END)
            context.layer = None
        elif ct == CommandType.WORKPIECE_START:
            uid = ops.workpiece_uid(idx)
            descendant = context.doc.find_descendant_by_uid(uid)
            if isinstance(descendant, WorkPiece):
                context.workpiece = descendant
            elif descendant is not None:
                logger.warning(
                    f"Expected WorkPiece for UID {uid}, "
                    f" but found {type(descendant)}"
                )
            self._emit_macros(context, gcode, MacroTrigger.WORKPIECE_START)
        elif ct == CommandType.WORKPIECE_END:
            self._emit_macros(context, gcode, MacroTrigger.WORKPIECE_END)
            context.workpiece = None

    def _handle_moving(
        self,
        gcode: List[str],
        ops: Ops,
        idx: int,
        ct: CommandType,
        context: GcodeContext,
    ) -> None:
        ea = ops.extra_axes(idx)
        if ct == CommandType.MOVE_TO:
            end = ops.endpoint(idx)
            self._handle_move_to(
                context,
                gcode,
                *end,
                extra_axes=ea,
            )
            self._update_current_pos(ops, idx)
        elif ct == CommandType.LINE_TO:
            end = ops.endpoint(idx)
            self._handle_line_to(
                context,
                gcode,
                *end,
                extra_axes=ea,
            )
            self._update_current_pos(ops, idx)
        elif ct == CommandType.ARC_TO:
            end = ops.endpoint(idx)
            i, j, cw = ops.arc_params(idx)
            self._handle_arc_to(
                context,
                gcode,
                end,
                (i, j),
                cw,
                extra_axes=ea,
            )
            self._update_current_pos(ops, idx)
        elif ct == CommandType.BEZIER_TO:
            self._handle_bezier_to(context, gcode, ops, idx, ea)
            self._update_current_pos(ops, idx)
        elif ct == CommandType.SCAN_LINE:
            # Deconstruct into simpler commands that the encoder already
            # understands.
            sub_ops = ops.linearize(idx, self._current_pos_xyz())
            for j in range(sub_ops.len()):
                self._handle_command(gcode, sub_ops, j, context)
            # To avoid float precision errors, explicitly set the final pos
            self._update_current_pos(ops, idx)

    def _emit_modal_speed(self, gcode: List[str], speed: float) -> None:
        """
        Emits a modal speed command if the dialect supports it and speed
        has changed.
        """
        if self.dialect.set_speed and speed != self.emitted_speed:
            cmd_str = self.dialect.set_speed.format(speed=speed)
            if cmd_str:
                gcode.append(cmd_str)
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
        if cmd_str:
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

        if self.laser_active and not self.dialect.continuous_laser_mode:
            if self.power > 0:
                # Find the currently active laser head to get its max power
                current_laser = self._get_current_laser_head(context)
                power_abs = power * current_laser.max_power
                cmd_str = self.dialect.laser_on.format(power=power_abs)
                if cmd_str:
                    gcode.append(cmd_str)
            else:  # power <= 0
                self._laser_off(context, gcode)

    def _handle_air_assist(
        self, context: GcodeContext, gcode: List[str], mode: str
    ) -> None:
        """Update air assist state with dialect commands"""
        coolant_on = mode == "On"
        if self.air_assist == coolant_on:
            return
        self.air_assist = coolant_on
        cmd = (
            self.dialect.air_assist_on
            if coolant_on
            else self.dialect.air_assist_off
        )
        if cmd:
            gcode.append(cmd)

    def _handle_coolant(
        self, context: GcodeContext, gcode: List[str], mode: str
    ) -> None:
        """Handle coolant commands (not used on laser cutters)."""

    def _handle_move_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        x: float,
        y: float,
        z: float,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """Rapid movement with laser safety"""
        self._laser_off(context, gcode)

        coord_vars = self._build_coord_commands(x, y, z, extra_axes=extra_axes)
        template_vars = {
            "x": coord_vars["x"],
            "y": coord_vars["y"],
            "z": coord_vars["z"],
            "x_cmd": coord_vars["x_cmd"],
            "y_cmd": coord_vars["y_cmd"],
            "z_cmd": coord_vars["z_cmd"],
            "extra_cmd": coord_vars["extra_cmd"],
        }

        f_command = ""
        if self.dialect.can_g0_with_speed:
            self._emit_modal_speed(gcode, self.travel_speed or 0)
            if self.travel_speed is not None:
                f_command = f" F{self._format_feed(self.travel_speed)}"
                self._emitted_cut_feed = None
            template_vars["f_command"] = f_command

        s_command = ""
        if self.laser_active and self.dialect.continuous_laser_mode:
            s_command = " S0"
        template_vars["s_command"] = s_command

        gcode.append(self.dialect.travel_move.format(**template_vars))

    def _build_cut_move_vars(
        self,
        context: GcodeContext,
        gcode: List[str],
        x: float,
        y: float,
        z: float,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> dict:
        """Build template variables common to all cutting moves."""
        self._laser_on(context, gcode)
        self._emit_modal_speed(gcode, self.cut_speed or 0)

        if self.dialect.modal_feedrate:
            f_command = ""
            if self.cut_speed is not None and (
                self._emitted_cut_feed is None
                or not math.isclose(self.cut_speed, self._emitted_cut_feed)
            ):
                f_command = f" F{self._format_feed(self.cut_speed)}"
                self._emitted_cut_feed = self.cut_speed
        else:
            f_command = (
                f" F{self._format_feed(self.cut_speed)}"
                if self.cut_speed is not None
                else ""
            )

        s_command = ""
        power_abs = 0.0
        if self.power is not None and (
            self.power > 0 or self.dialect.continuous_laser_mode
        ):
            current_laser = self._get_current_laser_head(context)
            power_abs = self.power * current_laser.max_power
            s_command = f" S{self._format_power(power_abs)}"

        coord_vars = self._build_coord_commands(x, y, z, extra_axes=extra_axes)
        return {
            "x": coord_vars["x"],
            "y": coord_vars["y"],
            "z": coord_vars["z"],
            "x_cmd": coord_vars["x_cmd"],
            "y_cmd": coord_vars["y_cmd"],
            "z_cmd": coord_vars["z_cmd"],
            "extra_cmd": coord_vars["extra_cmd"],
            "f_command": f_command,
            "s_command": s_command,
            "power": power_abs,
        }

    def _handle_line_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        x: float,
        y: float,
        z: float,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """Cutting movement with laser activation"""
        template_vars = self._build_cut_move_vars(
            context, gcode, x, y, z, extra_axes=extra_axes
        )
        gcode.append(self.dialect.linear_move.format(**template_vars))

    def _handle_arc_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        end: Point3D,
        center: Tuple[float, float],
        cw: bool,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """Cutting arc with laser activation"""
        x, y, z = end
        i, j = center
        template = self.dialect.arc_cw if cw else self.dialect.arc_ccw
        template_vars = self._build_cut_move_vars(
            context, gcode, x, y, z, extra_axes=extra_axes
        )
        template_vars["i"] = self._format_coord(i)
        template_vars["j"] = self._format_coord(j)
        gcode.append(template.format(**template_vars))

    def _handle_bezier_to(
        self,
        context: GcodeContext,
        gcode: List[str],
        ops: Ops,
        idx: int,
        extra_axes: Optional[Dict[Axis, float]],
    ) -> None:
        if not self.dialect.bezier_cubic:
            start = self._current_pos_xyz()
            sub_ops = ops.linearize(idx, start)
            for j in range(sub_ops.len()):
                sub_ct = sub_ops.command_type(j)
                if sub_ct == CommandType.LINE_TO:
                    end = sub_ops.endpoint(j)
                    sub_ea = sub_ops.extra_axes(j)
                    self._handle_line_to(
                        context,
                        gcode,
                        *end,
                        extra_axes=sub_ea,
                    )
            return
        start = self._current_pos_xyz()
        end = ops.endpoint(idx)
        c1, c2 = ops.bezier_params(idx)
        ex, ey, ez = end
        template_vars = self._build_cut_move_vars(
            context,
            gcode,
            ex,
            ey,
            ez,
            extra_axes=extra_axes,
        )
        template_vars["i"] = self._format_coord(c1[0] - start[0])
        template_vars["j"] = self._format_coord(c1[1] - start[1])
        template_vars["p"] = self._format_coord(c2[0] - ex)
        template_vars["q"] = self._format_coord(c2[1] - ey)
        gcode.append(self.dialect.bezier_cubic.format(**template_vars))

    def _laser_on(self, context: GcodeContext, gcode: List[str]) -> None:
        """Activate laser if not already on"""
        if not self.laser_active:
            if self.power or self.dialect.continuous_laser_mode:
                current_laser = self._get_current_laser_head(context)
                power_abs = (
                    self.power * current_laser.max_power if self.power else 0.0
                )
                cmd_str = self.dialect.laser_on.format(power=power_abs)
                if cmd_str:
                    gcode.append(cmd_str)
                self.laser_active = True

    def _laser_off(self, context: GcodeContext, gcode: List[str]) -> None:
        """Deactivate laser if active"""
        if self.laser_active:
            if not self.dialect.continuous_laser_mode:
                cmd_str = self.dialect.laser_off
                if cmd_str:
                    gcode.append(cmd_str)
                self.laser_active = False

    def _finalize(self, gcode: List[str]) -> None:
        """Ensures the G-code file ends with a newline."""
        if not gcode or gcode[-1]:
            gcode.append("")
