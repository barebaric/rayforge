from typing import Any, Optional, Tuple

from gi.repository import Adw

from ....core.varset import BoolVar, Var
from .base import RowAdapter, escape_title, register_adapter


@register_adapter(BoolVar)
class SwitchAdapter(RowAdapter):
    def __init__(self, row: Adw.SwitchRow) -> None:
        super().__init__()
        self._row = row
        self._row.connect(
            "notify::active", lambda r, p: self.changed.send(self)
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SwitchAdapter"]:
        row = Adw.SwitchRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        row.set_active(bool(initial_val) if initial_val is not None else False)
        return row, cls(row)

    def get_value(self) -> Optional[Any]:
        return self._row.get_active()

    def set_value(self, value: Any) -> None:
        self._row.set_active(bool(value))

    def update_from_var(self, var: Var):
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)
