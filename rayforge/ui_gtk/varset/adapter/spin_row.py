from typing import Any

from ....core.varset import FloatVar, IntVar, Var
from ...shared.pref_rows.base import SpinRow
from .base import RowAdapter, escape_title, register_adapter


@register_adapter(IntVar, FloatVar)
class SpinRowAdapter(RowAdapter):
    def __init__(self, row: SpinRow, is_int: bool) -> None:
        super().__init__()
        self._row = row
        self._is_int = is_int
        row.value_changed.connect(lambda r: self.changed.send(self))

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[SpinRow, "SpinRowAdapter"]:
        min_val = (
            var.min_val
            if isinstance(var, (IntVar, FloatVar)) and var.min_val is not None
            else -2147483647
        )
        max_val = (
            var.max_val
            if isinstance(var, (IntVar, FloatVar)) and var.max_val is not None
            else 2147483647
        )
        initial_val = getattr(var, target_property)
        is_int = var.var_type is int
        if is_int:
            digits = 0
        else:
            digits = (
                var.digits
                if isinstance(var, FloatVar) and var.digits is not None
                else 3
            )

        row = SpinRow(
            escape_title(var.label),
            var.description or None,
            lower=min_val,
            upper=max_val,
            digits=digits,
            value=(
                (int(initial_val) if is_int else float(initial_val))
                if initial_val is not None
                else (0 if is_int else 0.0)
            ),
        )
        return row, cls(row, is_int)

    def get_value(self) -> Any | None:
        if self._is_int:
            return self._row.get_int_value()
        return self._row.get_value()

    def set_value(self, value: Any) -> None:
        self._row.set_value(float(value))

    def update_from_var(self, var: Var):
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)
        min_val = (
            var.min_val
            if isinstance(var, (IntVar, FloatVar)) and var.min_val is not None
            else -2147483647
        )
        max_val = (
            var.max_val
            if isinstance(var, (IntVar, FloatVar)) and var.max_val is not None
            else 2147483647
        )
        self._row.set_range(min_val, max_val)
