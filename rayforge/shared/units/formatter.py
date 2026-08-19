import math
from typing import TYPE_CHECKING, Optional

from ...context import get_context
from .definitions import (
    get_base_unit_for_quantity,
    get_unit,
)

if TYPE_CHECKING:
    from .definitions import Unit


def format_value(value_in_base: float, quantity: str) -> str:
    """
    Formats a value from its base unit into a user-friendly string
    with the user's preferred display unit.
    """
    config = get_context().config
    base_unit = get_base_unit_for_quantity(quantity)
    pref_unit_name = config.unit_preferences.get(
        quantity, base_unit.name if base_unit else ""
    )
    display_unit = get_unit(pref_unit_name)

    if not display_unit:
        # Fallback if something is misconfigured
        return f"{value_in_base:.0f}"

    display_value = display_unit.from_base(value_in_base)
    return f"{display_value:.{display_unit.precision}f} {display_unit.label}"


def get_preferred_unit(quantity: str) -> Optional["Unit"]:
    """
    Returns the user's preferred display unit for a quantity, falling back
    to the quantity's base unit when no preference is set.
    """
    config = get_context().config
    base_unit = get_base_unit_for_quantity(quantity)
    pref_unit_name = config.unit_preferences.get(
        quantity, base_unit.name if base_unit else ""
    )
    return get_unit(pref_unit_name) or base_unit


def get_preferred_unit_factor(quantity: str) -> float:
    """
    Returns the number of base units (mm) in one preferred display unit.
    """
    unit = get_preferred_unit(quantity)
    if unit is None:
        return 1.0
    return unit.to_base(1.0)


def get_display_unit_settings(quantity: str) -> tuple[str, float, int]:
    """
    Returns ``(unit_name, factor, precision)`` describing how to display
    values of *quantity* in the user's preferred display unit.

    ``factor`` converts a base-unit value to the display unit
    (``display = base / factor``); it is the number of base units in one
    display unit. ``precision`` is the number of decimal places the unit
    suggests for display. Falls back to the quantity's base unit when no
    preference is set or the preferred unit is unknown.
    """
    base_unit = get_base_unit_for_quantity(quantity)
    unit = get_preferred_unit(quantity) or base_unit
    if unit is None:
        return "", 1.0, 0
    factor = unit.to_base(1.0)
    if not math.isfinite(factor) or factor <= 0.0:
        factor = 1.0
    return unit.name, factor, unit.precision


def get_default_grid_step_mm() -> float:
    """
    Returns a sensible fixed grid spacing (in mm) for the user's preferred
    length unit: one unit, except for mm which keeps the classic 10mm step.
    """
    factor = get_preferred_unit_factor("length")
    return 10.0 if factor <= 1.0 else factor
