from typing import Any

from ....core.varset import LengthVar, Var
from ...shared.pref_rows.length_spin_row import LengthSpinRow
from .base import RowAdapter, escape_title, register_adapter


@register_adapter(LengthVar)
class LengthRowAdapter(RowAdapter):
    """
    Adapts a LengthSpinRow for length values with unit conversion.

    Values are always read/written in base units (mm).
    """

    def __init__(self, row: LengthSpinRow) -> None:
        super().__init__()
        self._row = row
        row.value_changed.connect(lambda r: self.changed.send(self))

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[LengthSpinRow, "LengthRowAdapter"]:
        assert isinstance(var, LengthVar)
        initial_val = getattr(var, target_property)
        min_val = var.min_val if var.min_val is not None else -2147483647
        max_val = var.max_val if var.max_val is not None else 2147483647

        row = LengthSpinRow(
            escape_title(var.label),
            None,
            lower=min_val,
            upper=max_val,
            value_in_base=(
                float(initial_val) if initial_val is not None else 0.0
            ),
        )
        if var.description:
            row.set_subtitle(var.description)
        return row, cls(row)

    def get_value(self) -> Any | None:
        return self._row.get_value_in_base_units()

    def set_value(self, value: Any) -> None:
        self._row.set_value_in_base_units(value)

    def update_from_var(self, var: Var):
        assert isinstance(var, LengthVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.min_val is not None or var.max_val is not None:
            self._row.set_range(
                var.min_val if var.min_val is not None else -2147483647,
                var.max_val if var.max_val is not None else 2147483647,
            )
