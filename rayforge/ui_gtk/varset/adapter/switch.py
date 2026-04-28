from typing import Any, Optional, Tuple

from gi.repository import Adw, Gtk

from ....core.varset import Var
from .base import RowAdapter, escape_title


class SwitchAdapter(RowAdapter):
    def __init__(self, switch: Gtk.Switch) -> None:
        self._switch = switch

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SwitchAdapter"]:
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        initial_val = getattr(var, target_property)
        switch.set_active(
            bool(initial_val) if initial_val is not None else False
        )
        row.add_suffix(switch)
        row.set_activatable_widget(switch)
        return row, cls(switch)

    def get_value(self) -> Optional[Any]:
        return self._switch.get_active()

    def set_value(self, value: Any) -> None:
        self._switch.set_active(bool(value))
