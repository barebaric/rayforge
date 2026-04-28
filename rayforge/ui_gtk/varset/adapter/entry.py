from typing import Any, Optional, Tuple

from gi.repository import Adw

from ....core.varset import Var
from .base import RowAdapter, escape_title


class EntryAdapter(RowAdapter):
    def __init__(self, row: Adw.EntryRow) -> None:
        self._row = row

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "EntryAdapter"]:
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        return row, cls(row)

    def get_value(self) -> Optional[Any]:
        return self._row.get_text()

    def set_value(self, value: Any) -> None:
        self._row.set_text(str(value))


class HostnameAdapter(EntryAdapter):
    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "HostnameAdapter"]:
        row = Adw.EntryRow(title=escape_title(var.label))
        if var.description:
            row.set_tooltip_text(var.description)
        row.set_show_apply_button(True)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            row.set_text(str(initial_val))
        return row, cls(row)
