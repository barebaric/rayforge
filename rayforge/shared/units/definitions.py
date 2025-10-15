from dataclasses import dataclass
from typing import Dict, List, Optional
from .engine import engine


@dataclass(frozen=True)
class Unit:
    """
    A declarative definition for a unit of measurement.
    Conversion logic is handled by the global ConversionEngine.
    """

    name: str  # Programmatic, normalized identifier (e.g., "mm/min")
    label: str  # User-facing, translatable string (e.g., "mm/min")
    quantity: str  # Physical quantity measured (e.g., "speed", "length")
    description: Optional[str] = None  # Translatable tooltip
    precision: int = 2  # Suggested decimal places for display

    def to_base(self, value: float) -> float:
        """Converts a value from this unit to the application's base unit."""
        base_unit = get_base_unit_for_quantity(self.quantity)
        if not base_unit or base_unit.name == self.name:
            return value
        converted_value, _ = engine.convert(value, self.name, base_unit.name)
        return converted_value

    def from_base(self, value: float) -> float:
        """Converts a value from the application's base unit to this unit."""
        base_unit = get_base_unit_for_quantity(self.quantity)
        if not base_unit or base_unit.name == self.name:
            return value
        converted_value, _ = engine.convert(value, base_unit.name, self.name)
        return converted_value


_UNIT_REGISTRY: Dict[str, Unit] = {}
_BASE_UNITS: Dict[str, str] = {}


def register_unit(unit: Unit):
    """Adds a unit to the central registry."""
    if unit.name in _UNIT_REGISTRY:
        raise ValueError(
            f"Unit with name '{unit.name}' is already registered."
        )
    _UNIT_REGISTRY[unit.name] = unit


def set_base_unit(quantity: str, unit_name: str):
    """Sets the application-wide base unit for a given quantity."""
    if quantity in _BASE_UNITS:
        raise ValueError(
            f"Base unit for quantity '{quantity}' is already set."
        )
    if unit_name not in _UNIT_REGISTRY:
        raise ValueError(
            f"Cannot set unregistered unit '{unit_name}' as base."
        )
    _BASE_UNITS[quantity] = unit_name


def get_units_for_quantity(quantity: str) -> List[Unit]:
    """Returns all registered units for a specific physical quantity."""
    units = [u for u in _UNIT_REGISTRY.values() if u.quantity == quantity]
    # Sort by label for consistent UI presentation
    return sorted(units, key=lambda u: u.label)


def get_unit(name: str) -> Optional[Unit]:
    """Retrieves a specific unit by its programmatic name."""
    return _UNIT_REGISTRY.get(name)


def get_base_unit_for_quantity(quantity: str) -> Optional[Unit]:
    """Retrieves the designated base unit for a quantity."""
    base_unit_name = _BASE_UNITS.get(quantity)
    return get_unit(base_unit_name) if base_unit_name else None


# --- Define and Register Speed Units ---
# Application base unit for speed is mm/min.

register_unit(
    Unit(name="mm/min", label=_("mm/min"), quantity="speed", precision=0)
)
register_unit(
    Unit(name="mm/s", label=_("mm/s"), quantity="speed", precision=1)
)
register_unit(
    Unit(name="in/min", label=_("in/min"), quantity="speed", precision=1)
)
register_unit(
    Unit(name="in/s", label=_("in/s"), quantity="speed", precision=2)
)

set_base_unit("speed", "mm/min")

# --- Define and Register Length Units ---
# Application base unit for length is mm.

register_unit(Unit(name="mm", label=_("mm"), quantity="length", precision=1))
register_unit(Unit(name="cm", label=_("cm"), quantity="length", precision=2))
register_unit(Unit(name="m", label=_("m"), quantity="length", precision=3))
register_unit(Unit(name="in", label=_("in"), quantity="length", precision=3))
register_unit(Unit(name="ft", label=_("ft"), quantity="length", precision=3))

set_base_unit("length", "mm")

# --- Define and Register Acceleration Units ---
# Application base unit for acceleration is mm/s².

register_unit(
    Unit(name="mm/s²", label=_("mm/s²"), quantity="acceleration", precision=0)
)
register_unit(
    Unit(name="cm/s²", label=_("cm/s²"), quantity="acceleration", precision=1)
)
register_unit(
    Unit(name="m/s²", label=_("m/s²"), quantity="acceleration", precision=2)
)
register_unit(
    Unit(name="in/s²", label=_("in/s²"), quantity="acceleration", precision=2)
)
register_unit(
    Unit(name="ft/s²", label=_("ft/s²"), quantity="acceleration", precision=3)
)

set_base_unit("acceleration", "mm/s²")
