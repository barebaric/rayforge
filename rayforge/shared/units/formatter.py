from .definitions import (
    get_base_unit_for_quantity,
    get_unit,
)
from ...context import get_context


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
