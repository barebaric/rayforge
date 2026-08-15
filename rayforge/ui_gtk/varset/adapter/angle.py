from typing import Any

from ....core.varset import AngleVar, Var
from ...shared.pref_rows.angle_spin_row import AngleSpinRow
from .base import RowAdapter, escape_title, register_adapter


@register_adapter(AngleVar)
class AngleRowAdapter(RowAdapter):
    """Adapts an :class:`AngleSpinRow` for angle values in degrees."""

    def __init__(self, row: AngleSpinRow) -> None:
        super().__init__()
        self._row = row
        row.value_changed.connect(lambda r: self.changed.send(self))

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[AngleSpinRow, "AngleRowAdapter"]:
        assert isinstance(var, AngleVar)
        min_val = var.min_val if var.min_val is not None else -360.0
        max_val = var.max_val if var.max_val is not None else 360.0
        initial_val = getattr(var, target_property)

        row = AngleSpinRow(
            escape_title(var.label),
            var.description or None,
            lower=min_val,
            upper=max_val,
            value=(float(initial_val) if initial_val is not None else 0.0),
        )
        return row, cls(row)

    def get_value(self) -> Any | None:
        return self._row.get_value()

    def set_value(self, value: Any) -> None:
        self._row.set_value(float(value))

    def update_from_var(self, var: Var):
        assert isinstance(var, AngleVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        min_val = var.min_val if var.min_val is not None else -360.0
        max_val = var.max_val if var.max_val is not None else 360.0
        self._row.set_range(min_val, max_val)
