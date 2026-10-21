"""
The Interval Test: a row of filled squares, each engraved with its own
line interval at the same power and speed.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry

from rayforge.core.varset import IntVar, LengthVar, VarSet

from .common import (
    LABEL_GAP,
    LABEL_HEIGHT,
    label_block_height,
    label_vars,
    labels_cell,
    laser_vars,
)
from .generator import CalibrationGenerator, engrave_step
from .layout import (
    CalibrationLayout,
    CellEntry,
    FilledRect,
    row_positions,
    text,
)

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

MM_PER_INCH = 25.4


def interval_values(params: dict[str, Any]) -> list[float]:
    """The line interval of every cell, evenly spread over the range."""
    count = int(params["count"])
    if count < 2:
        raise ValueError(_("An Interval Test needs at least two cells."))
    low, high = params["min_interval"], params["max_interval"]
    if low <= 0 or high <= low:
        raise ValueError(
            _(
                "The minimum line interval must be above zero and below "
                "the maximum."
            )
        )
    step = (high - low) / (count - 1)
    return [low + i * step for i in range(count)]


def lines_per_inch(interval_mm: float) -> float:
    """The line density (LPI) that matches a line interval in mm."""
    return MM_PER_INCH / interval_mm


class IntervalTestGenerator(CalibrationGenerator):
    id = "interval_test"

    @property
    def title(self) -> str:
        return _("Interval Test")

    def varset(self, context: "RayforgeContext") -> VarSet:
        return VarSet(
            vars=[
                IntVar(
                    key="count",
                    label=_("Number of Cells"),
                    default=5,
                    min_val=2,
                    max_val=20,
                ),
                LengthVar(
                    key="min_interval",
                    label=_("Smallest Interval"),
                    default=0.05,
                    min_val=0.01,
                    max_val=5.0,
                    digits=3,
                ),
                LengthVar(
                    key="max_interval",
                    label=_("Largest Interval"),
                    default=0.25,
                    min_val=0.01,
                    max_val=5.0,
                    digits=3,
                ),
                LengthVar(
                    key="cell_size",
                    label=_("Cell Size"),
                    default=10.0,
                    min_val=1.0,
                    max_val=100.0,
                ),
                LengthVar(
                    key="spacing",
                    label=_("Spacing"),
                    default=5.0,
                    min_val=0.0,
                    max_val=50.0,
                ),
                *laser_vars(context, 30.0, 3000),
                *label_vars(context),
            ]
        )

    def help_text(self, params: dict[str, Any]) -> str:
        return _(
            "Engraves a row of filled squares, each with its own line "
            "interval, at the same power and speed."
        )

    def layout(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> CalibrationLayout:
        intervals = interval_values(params)
        size = params["cell_size"]
        xs = row_positions(len(intervals), size, params["spacing"])
        cell_y = label_block_height(params, lines=2)
        result = CalibrationLayout()
        if params["include_labels"]:
            labels = self._labels(params, xs, intervals, cell_y)
            result.entries.append(labels_cell(context, params, labels))
        for x, interval in zip(xs, intervals):
            name = _("Interval {interval:.2f} mm").format(interval=interval)
            step = engrave_step(
                context, params["power_percent"], params["speed"]
            )
            step.name = name
            step.line_interval_mm = interval
            step.depth_mode = "CONSTANT_POWER"
            shape = FilledRect(x, cell_y, size, size)
            result.entries.append(CellEntry(name, shape, step))
        return result

    @staticmethod
    def _labels(
        params: dict[str, Any],
        xs: list[float],
        intervals: list[float],
        cell_y: float,
    ) -> Geometry:
        size = params["cell_size"]
        top = cell_y - LABEL_GAP * LABEL_HEIGHT
        fit = (size + params["spacing"]) * 0.9
        labels = Geometry()
        for x, interval in zip(xs, intervals):
            center = x + size / 2
            labels.extend(
                text(f"{interval:.2f}", LABEL_HEIGHT, center, top, fit)
            )
            lpi = _("{lpi:.0f} LPI").format(lpi=lines_per_inch(interval))
            labels.extend(
                text(lpi, LABEL_HEIGHT, center, top - LABEL_HEIGHT * 1.5, fit)
            )
        return labels
