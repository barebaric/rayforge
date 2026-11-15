"""Settings and building blocks shared by the Interval and Focus Tests."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry

from rayforge.core.varset import BoolVar, SliderFloatVar, SpeedVar, Var

from .generator import contour_step
from .layout import CellEntry, VectorShape

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

#: Gap between a cell or line and its label, in label heights.
LABEL_GAP = 0.6
LABEL_HEIGHT = 2.5


def labels_active(values: dict[str, Any]) -> bool:
    return bool(values.get("include_labels"))


def _speed_limit(context: "RayforgeContext") -> int | None:
    machine = context.machine
    return int(machine.max_cut_speed) if machine is not None else None


def _capped(speed: int, limit: int | None) -> int:
    return speed if limit is None else min(speed, limit)


def laser_vars(
    context: "RayforgeContext", power_percent: float, speed: int
) -> list[Var]:
    """Power and speed of the test, capped at the machine's speed."""
    limit = _speed_limit(context)
    return [
        SliderFloatVar(
            key="power_percent",
            label=_("Power"),
            default=power_percent,
            min_val=0.0,
            max_val=100.0,
            show_value=True,
            format_suffix="%",
        ),
        SpeedVar(
            key="speed",
            label=_("Speed"),
            default=_capped(speed, limit),
            min_val=1,
            max_val=limit,
        ),
    ]


def label_vars(context: "RayforgeContext") -> list[Var]:
    limit = _speed_limit(context)
    return [
        BoolVar(
            key="include_labels",
            label=_("Include Labels"),
            default=True,
        ),
        SliderFloatVar(
            key="label_power_percent",
            label=_("Label Engrave Power (%)"),
            default=10.0,
            min_val=0.0,
            max_val=100.0,
            show_value=True,
            format_suffix="%",
            sensitive_when=labels_active,
        ),
        SpeedVar(
            key="label_speed",
            label=_("Label Engrave Speed"),
            default=_capped(1000, limit),
            min_val=1,
            max_val=limit,
            sensitive_when=labels_active,
        ),
    ]


def label_block_height(params: dict[str, Any], lines: int) -> float:
    """Height reserved below the cells for *lines* lines of labels."""
    if not params["include_labels"]:
        return 0.0
    return LABEL_HEIGHT * (LABEL_GAP + lines + (lines - 1) * 0.5)


def labels_cell(
    context: "RayforgeContext", params: dict[str, Any], geometry: Geometry
) -> CellEntry:
    """All labels of a test as one workpiece with a low-power step."""
    step = contour_step(
        context, params["label_power_percent"], params["label_speed"]
    )
    step.name = _("Labels")
    return CellEntry(_("Labels"), VectorShape(geometry), step)
