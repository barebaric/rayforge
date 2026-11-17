"""
The Focus Test: lines engraved at different focus heights, or one long
line on tilted material.

Between the lines it runs machine commands: relative Z moves on
machines with a Z axis, or ``M0`` pauses so the head can be moved by
hand. A positive offset means more distance between the head and the
material.
"""

from enum import Enum
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry

from rayforge.core.step_registry import step_registry
from rayforge.core.varset import (
    IntVar,
    LabeledChoiceVar,
    LengthVar,
    VarSet,
)
from rayforge.machine.driver import get_driver_cls

from .common import (
    LABEL_GAP,
    LABEL_HEIGHT,
    label_block_height,
    label_vars,
    labels_cell,
    laser_vars,
)
from .generator import CalibrationGenerator, contour_step
from .layout import (
    CalibrationLayout,
    CellEntry,
    CommandEntry,
    VectorShape,
    line,
    row_positions,
    text,
)

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

#: The test never moves the head further than this from the height it
#: starts at, so a typo cannot drive it into the material.
MAX_FOCUS_OFFSET_MM = 10.0
#: Spacing and length of the distance ticks along the ramp, in mm.
RAMP_TICK_MM = 10.0
RAMP_TICK_LENGTH = 2.0


class FocusMode(Enum):
    """How the focus height changes between the lines."""

    Z_AXIS = "z_axis"
    MANUAL = "manual"
    RAMP = "ramp"

    @property
    def label(self) -> str:
        labels = {
            FocusMode.Z_AXIS: _("Z axis steps"),
            FocusMode.MANUAL: _("Manual (pause between lines)"),
            FocusMode.RAMP: _("Ramp (tilted material)"),
        }
        return labels[self]

    @property
    def help_text(self) -> str:
        texts = {
            FocusMode.Z_AXIS: _(
                "Focus the laser as usual first. The job moves the head "
                "to the first offset, engraves one line per offset and "
                "returns the head to the starting height. A positive "
                "offset means more distance between the head and the "
                "material."
            ),
            FocusMode.MANUAL: _(
                "Focus the laser as usual first. The labels are engraved "
                "at that height. Before each line the job pauses: set the "
                "head to the first offset, then move it by one step at "
                "every following pause, and press Resume. A positive "
                "offset means more distance between the head and the "
                "material."
            ),
            FocusMode.RAMP: _(
                "Prop up one end of a flat strip so it rises evenly under "
                "the line. The thinnest part of the line marks the focus; "
                "the ticks show the distance along the line, so the "
                "height there follows from the rise."
            ),
        }
        return texts[self]


def available_modes(context: "RayforgeContext") -> list[FocusMode]:
    """
    The modes the current machine supports, in order of preference: Z
    axis steps need a Z axis, and both Z and manual mode need a G-code
    driver and the Command step (Automation addon).
    """
    modes = [FocusMode.RAMP]
    machine = context.machine
    if machine is None or step_registry.get("CommandStep") is None:
        return modes
    if not get_driver_cls(machine.driver_name or "").uses_gcode:
        return modes
    modes.insert(0, FocusMode.MANUAL)
    if machine.has_z_axis:
        modes.insert(0, FocusMode.Z_AXIS)
    return modes


def focus_offsets(params: dict[str, Any]) -> list[float]:
    """The focus offset of every line."""
    count = int(params["count"])
    if count < 2:
        raise ValueError(_("A Focus Test needs at least two lines."))
    if params["step"] == 0:
        raise ValueError(_("The focus step must not be zero."))
    offsets = [
        params["start_offset"] + i * params["step"] for i in range(count)
    ]
    if max(abs(offsets[0]), abs(offsets[-1])) > MAX_FOCUS_OFFSET_MM:
        raise ValueError(
            _(
                "The focus offsets must stay within ±{limit:g} mm of "
                "the starting height."
            ).format(limit=MAX_FOCUS_OFFSET_MM)
        )
    return offsets


def format_offset(offset: float) -> str:
    """A focus offset as shown in labels, e.g. ``+0.5`` or ``-2.0``."""
    if abs(offset) < 1e-9:
        return "0.0"
    return f"{offset:+.1f}"


def _number(value: float) -> str:
    result = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if result in ("-0", "") else result


def _z_move(distance: float, reverse_z: bool) -> list[str]:
    """A relative Z move that leaves the controller in G90."""
    if reverse_z:
        distance = -distance
    return ["G91", f"G0 Z{_number(distance)}", "G90"]


def focus_commands(
    mode: FocusMode, params: dict[str, Any], reverse_z: bool
) -> dict[int, CommandEntry]:
    """
    The commands that run right before line *index* (an index equal to
    the line count means after the last line).
    """
    offsets = focus_offsets(params)
    if mode == FocusMode.Z_AXIS:
        commands: dict[int, CommandEntry] = {}
        previous = 0.0
        for index, offset in enumerate(offsets):
            name = _("Z to {offset} mm").format(offset=format_offset(offset))
            lines = _z_move(offset - previous, reverse_z)
            commands[index] = CommandEntry(name, lines)
            previous = offset
        commands[len(offsets)] = CommandEntry(
            _("Z back to the starting height"), _z_move(-previous, reverse_z)
        )
        return commands
    if mode == FocusMode.MANUAL:
        first = _("Pause: set the head to {offset} mm from focus").format(
            offset=format_offset(offsets[0])
        )
        later = _("Pause: move the head by {step} mm").format(
            step=format_offset(params["step"])
        )
        return {
            index: CommandEntry(first if index == 0 else later, ["M0"])
            for index in range(len(offsets))
        }
    return {}


def _is_ramp(values: dict[str, Any]) -> bool:
    return values.get("mode") == FocusMode.RAMP.value


def _is_not_ramp(values: dict[str, Any]) -> bool:
    return not _is_ramp(values)


class FocusTestGenerator(CalibrationGenerator):
    id = "focus_test"

    @property
    def title(self) -> str:
        return _("Focus Test")

    def varset(self, context: "RayforgeContext") -> VarSet:
        return self._varset(context, available_modes(context))

    def settings_varset(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> VarSet:
        """Also lists the test's own method when this machine lacks it."""
        modes = available_modes(context)
        mode = FocusMode(params["mode"])
        if mode not in modes:
            modes.append(mode)
        return self._varset(context, modes)

    def warning(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> str:
        mode = FocusMode(params["mode"])
        if mode in available_modes(context):
            return ""
        return _(
            "The method {method} is not available on this machine. Choose "
            "another method to rebuild the test."
        ).format(method=mode.label)

    def _varset(
        self, context: "RayforgeContext", modes: list[FocusMode]
    ) -> VarSet:
        limit = MAX_FOCUS_OFFSET_MM
        return VarSet(
            vars=[
                LabeledChoiceVar(
                    key="mode",
                    label=_("Method"),
                    choices=[(m.label, m.value) for m in modes],
                    default=modes[0].value,
                    allow_none=False,
                ),
                LengthVar(
                    key="start_offset",
                    label=_("First Offset"),
                    default=-2.0,
                    min_val=-limit,
                    max_val=limit,
                    digits=2,
                    visible_when=_is_not_ramp,
                ),
                LengthVar(
                    key="step",
                    label=_("Step"),
                    default=0.5,
                    min_val=-limit,
                    max_val=limit,
                    digits=2,
                    visible_when=_is_not_ramp,
                ),
                IntVar(
                    key="count",
                    label=_("Number of Lines"),
                    default=9,
                    min_val=2,
                    max_val=41,
                    visible_when=_is_not_ramp,
                ),
                LengthVar(
                    key="line_length",
                    label=_("Line Length"),
                    default=15.0,
                    min_val=1.0,
                    max_val=200.0,
                    visible_when=_is_not_ramp,
                ),
                LengthVar(
                    key="spacing",
                    label=_("Spacing"),
                    default=6.0,
                    min_val=0.5,
                    max_val=50.0,
                    visible_when=_is_not_ramp,
                ),
                LengthVar(
                    key="ramp_length",
                    label=_("Ramp Length"),
                    default=100.0,
                    min_val=10.0,
                    max_val=1000.0,
                    visible_when=_is_ramp,
                ),
                *laser_vars(context, 30.0, 1500),
                *label_vars(context),
            ]
        )

    def help_text(self, params: dict[str, Any]) -> str:
        return FocusMode(params["mode"]).help_text

    def layout(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> CalibrationLayout:
        mode = FocusMode(params["mode"])
        if mode not in available_modes(context):
            raise ValueError(
                _(
                    "This machine does not support the focus test mode {mode}."
                ).format(mode=mode.label)
            )
        if mode == FocusMode.RAMP:
            return self._ramp(params, context)
        return self._lines(mode, params, context)

    def _lines(
        self,
        mode: FocusMode,
        params: dict[str, Any],
        context: "RayforgeContext",
    ) -> CalibrationLayout:
        offsets = focus_offsets(params)
        xs = row_positions(len(offsets), 0.0, params["spacing"])
        line_y = label_block_height(params, lines=2)
        result = CalibrationLayout()
        if params["include_labels"]:
            labels = self._line_labels(params, xs, offsets, line_y)
            result.entries.append(labels_cell(context, params, labels))
        machine = context.machine
        reverse_z = bool(machine and machine.reverse_z_axis)
        commands = focus_commands(mode, params, reverse_z)
        for index, (x, offset) in enumerate(zip(xs, offsets)):
            if index in commands:
                result.entries.append(commands[index])
            name = _("Focus {offset} mm").format(offset=format_offset(offset))
            step = contour_step(
                context, params["power_percent"], params["speed"]
            )
            step.name = name
            shape = VectorShape(
                line(x, line_y, x, line_y + params["line_length"])
            )
            result.entries.append(CellEntry(name, shape, step))
        if len(offsets) in commands:
            result.entries.append(commands[len(offsets)])
        return result

    @staticmethod
    def _line_labels(
        params: dict[str, Any],
        xs: list[float],
        offsets: list[float],
        line_y: float,
    ) -> Geometry:
        """
        One label per line. Labels alternate between two rows, so each
        may be almost twice as wide as the line spacing.
        """
        labels = Geometry()
        top = line_y - LABEL_GAP * LABEL_HEIGHT
        for index, (x, offset) in enumerate(zip(xs, offsets)):
            row_top = top - (index % 2) * LABEL_HEIGHT * 1.5
            labels.extend(
                text(
                    format_offset(offset),
                    LABEL_HEIGHT,
                    x,
                    row_top,
                    params["spacing"] * 1.8,
                )
            )
        return labels

    def _ramp(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> CalibrationLayout:
        length = params["ramp_length"]
        line_y = label_block_height(params, lines=1) + RAMP_TICK_LENGTH
        result = CalibrationLayout()
        if params["include_labels"]:
            labels = Geometry()
            for i in range(int(length // RAMP_TICK_MM) + 1):
                x = i * RAMP_TICK_MM
                labels.extend(
                    line(x, line_y - RAMP_TICK_LENGTH, x, line_y - 0.5)
                )
                labels.extend(
                    text(
                        f"{x:g}",
                        LABEL_HEIGHT,
                        x,
                        line_y - RAMP_TICK_LENGTH - 0.5,
                    )
                )
            result.entries.append(labels_cell(context, params, labels))
        name = _("Focus ramp")
        step = contour_step(context, params["power_percent"], params["speed"])
        step.name = name
        shape = VectorShape(line(0.0, line_y, length, line_y))
        result.entries.append(CellEntry(name, shape, step))
        return result
