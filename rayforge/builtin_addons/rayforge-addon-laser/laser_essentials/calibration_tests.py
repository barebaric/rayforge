"""
Parameters and layout of the Interval Test and the Focus Test.

Both tests are generated as ordinary project content (a layer with
workpieces and steps, see ``commands/calibration_test_cmd.py``). This
module holds the parts that do not touch the document: the values each
cell or line is made with, their positions, and the machine commands
the Focus Test puts between its lines.
"""

from dataclasses import dataclass
from enum import Enum
from gettext import gettext as _

MM_PER_INCH = 25.4

# The Focus Test never moves the head further than this from the
# height it starts at, so a typo cannot drive it into the material.
MAX_FOCUS_OFFSET_MM = 10.0


class FocusMode(Enum):
    """How the focus height changes between the lines of a Focus Test."""

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
    def needs_commands(self) -> bool:
        """True when the mode puts machine commands between lines."""
        return self != FocusMode.RAMP


@dataclass
class IntervalTestParams:
    """Settings of an Interval Test."""

    min_interval: float = 0.05
    max_interval: float = 0.25
    count: int = 5
    cell_size: float = 10.0
    spacing: float = 5.0
    power_percent: float = 30.0
    speed: float = 3000.0
    include_labels: bool = True
    label_power_percent: float = 10.0
    label_speed: float = 1000.0
    label_height: float = 2.5


@dataclass
class FocusTestParams:
    """Settings of a Focus Test."""

    mode: FocusMode = FocusMode.MANUAL
    start_offset: float = -2.0
    step: float = 0.5
    count: int = 9
    line_length: float = 15.0
    spacing: float = 6.0
    ramp_length: float = 100.0
    power_percent: float = 30.0
    speed: float = 1500.0
    include_labels: bool = True
    label_power_percent: float = 10.0
    label_speed: float = 1000.0
    label_height: float = 2.5


def interval_values(params: IntervalTestParams) -> list[float]:
    """The line interval of every cell, evenly spread over the range."""
    if params.count < 2:
        raise ValueError(_("An Interval Test needs at least two cells."))
    low, high = params.min_interval, params.max_interval
    if low <= 0 or high <= low:
        raise ValueError(
            _(
                "The minimum line interval must be above zero and below "
                "the maximum."
            )
        )
    step = (high - low) / (params.count - 1)
    return [low + i * step for i in range(params.count)]


def lines_per_inch(interval_mm: float) -> float:
    """The line density (LPI) that matches a line interval in mm."""
    return MM_PER_INCH / interval_mm


def row_positions(count: int, size: float, spacing: float) -> list[float]:
    """The offsets of *count* items of *size* placed in a row."""
    return [i * (size + spacing) for i in range(count)]


def focus_offsets(params: FocusTestParams) -> list[float]:
    """The focus offset of every line of a Focus Test."""
    if params.count < 2:
        raise ValueError(_("A Focus Test needs at least two lines."))
    if params.step == 0:
        raise ValueError(_("The focus step must not be zero."))
    offsets = [
        params.start_offset + i * params.step for i in range(params.count)
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
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _z_move(distance: float, reverse_z: bool) -> list[str]:
    """A relative Z move that leaves the controller in G90."""
    if reverse_z:
        distance = -distance
    return ["G91", f"G0 Z{_number(distance)}", "G90"]


def focus_commands(
    params: FocusTestParams, reverse_z: bool
) -> list[tuple[int, list[str], str]]:
    """
    The machine commands a Focus Test runs between its lines.

    Returns ``(index, lines, name)`` tuples: *lines* run right before
    line *index* (an index equal to the line count means after the last
    line), and *name* describes the command in the workflow.

    A positive offset means more distance between the head and the
    material. In Z axis mode the head is moved relatively with
    ``G91``/``G0 Z``/``G90`` and returned to its starting height at the
    end. In manual mode the job pauses with ``M0`` before every line so
    the head can be moved by hand.
    """
    if not params.mode.needs_commands:
        return []
    offsets = focus_offsets(params)
    commands: list[tuple[int, list[str], str]] = []
    if params.mode == FocusMode.Z_AXIS:
        previous = 0.0
        for index, offset in enumerate(offsets):
            name = _("Z to {offset} mm").format(offset=format_offset(offset))
            commands.append(
                (index, _z_move(offset - previous, reverse_z), name)
            )
            previous = offset
        commands.append(
            (
                len(offsets),
                _z_move(-previous, reverse_z),
                _("Z back to the starting height"),
            )
        )
        return commands
    for index, offset in enumerate(offsets):
        if index == 0:
            name = _("Pause: set the head to {offset} mm from focus").format(
                offset=format_offset(offset)
            )
        else:
            name = _("Pause: move the head by {step} mm").format(
                step=format_offset(params.step)
            )
        commands.append((index, ["M0"], name))
    return commands
