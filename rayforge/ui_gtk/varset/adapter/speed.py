from typing import Any, Optional, Tuple

from ....context import get_context
from ....core.varset import SpeedVar, Var
from ...shared.unit_spin_row import SpeedSpinRow
from .base import RowAdapter, escape_title, register_adapter

_DEFAULT_MAX_SPEED = 3000


def _resolve_max_speed(var: SpeedVar) -> int:
    if var.max_val is not None:
        return var.max_val
    machine = get_context().machine if get_context() else None
    if machine is None:
        return _DEFAULT_MAX_SPEED
    if var.role == "travel":
        return machine.max_travel_speed
    return machine.max_cut_speed


@register_adapter(SpeedVar)
class SpeedRowAdapter(RowAdapter):
    """
    Adapts a SpeedSpinRow for speed values with unit conversion.

    Values are always read/written in base units (mm/min).
    """

    def __init__(self, row: SpeedSpinRow) -> None:
        super().__init__()
        self._row = row
        row.value_changed.connect(lambda r: self.changed.send(self))

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[SpeedSpinRow, "SpeedRowAdapter"]:
        assert isinstance(var, SpeedVar)
        max_speed = _resolve_max_speed(var)
        initial_val = getattr(var, target_property)
        min_val = var.min_val or 0

        row = SpeedSpinRow(
            escape_title(var.label),
            lower=min_val,
            upper=max_speed,
            max_value_in_base=max_speed,
            value_in_base=(int(initial_val) if initial_val is not None else 0),
        )
        return row, cls(row)

    def get_value(self) -> Optional[Any]:
        return int(self._row.get_value_in_base_units())

    def set_value(self, value: Any) -> None:
        self._row.set_value_in_base_units(value)

    def update_from_var(self, var: Var):
        assert isinstance(var, SpeedVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.min_val is not None or var.max_val is not None:
            self._row.set_bounds_in_base(var.min_val, var.max_val)
