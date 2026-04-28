from typing import Any, Optional, Tuple

from gi.repository import Adw, Gtk

from ....core.varset import Var
from ...shared.adwfix import get_spinrow_int
from .base import RowAdapter, escape_title


class SpinRowAdapter(RowAdapter):
    def __init__(self, row: Adw.SpinRow, is_int: bool) -> None:
        self._row = row
        self._is_int = is_int

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SpinRowAdapter"]:
        min_val = getattr(var, "min_val", None)
        max_val = getattr(var, "max_val", None)
        lower = min_val if min_val is not None else -2147483647
        upper = max_val if max_val is not None else 2147483647
        initial_val = getattr(var, target_property)
        is_int = var.var_type is int

        adj = Gtk.Adjustment(
            value=(int(initial_val) if is_int else float(initial_val))
            if initial_val is not None
            else (0 if is_int else 0.0),
            lower=lower,
            upper=upper,
            step_increment=1 if is_int else 0.1,
        )
        row = Adw.SpinRow(
            adjustment=adj,
            digits=3 if not is_int else 0,
            title=escape_title(var.label),
        )
        if var.description:
            row.set_subtitle(var.description)
        return row, cls(row, is_int)

    def get_value(self) -> Optional[Any]:
        if self._is_int:
            return get_spinrow_int(self._row)
        return self._row.get_value()

    def set_value(self, value: Any) -> None:
        self._row.set_value(float(value))
