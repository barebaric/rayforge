from typing import Any, Optional, Tuple

from gi.repository import Adw

from ....core.varset import HostnameVar, Var
from ....machine.transport.validators import is_valid_hostname_or_ip
from .base import RowAdapter, escape_title, register_adapter


class EntryAdapter(RowAdapter):
    has_natural_commit = True

    def __init__(self, row: Adw.EntryRow) -> None:
        super().__init__()
        self._row = row
        self._row.connect("apply", lambda r: self.changed.send(self))

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

    def update_from_var(self, var: Var):
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_tooltip_text(var.description)


@register_adapter(HostnameVar)
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

    def __init__(self, row: Adw.EntryRow) -> None:
        super().__init__(row)

        def on_validate(entry_row):
            if is_valid_hostname_or_ip(entry_row.get_text()):
                entry_row.remove_css_class("error")
            else:
                entry_row.add_css_class("error")

        self._row.connect("changed", on_validate)
        on_validate(self._row)
