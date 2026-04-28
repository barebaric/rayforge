from typing import Any, Optional, Tuple

from gi.repository import Adw, Gtk

from ....core.varset import TextAreaVar, Var
from .base import RowAdapter, escape_title, register_adapter


@register_adapter(TextAreaVar)
class TextAreaAdapter(RowAdapter):
    def __init__(self, row: Adw.ExpanderRow, text_view: Gtk.TextView) -> None:
        super().__init__()
        self._row = row
        self._text_view = text_view

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "TextAreaAdapter"]:
        row = Adw.ExpanderRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        text_view = Gtk.TextView(
            monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR
        )
        scroller = Gtk.ScrolledWindow(
            child=text_view,
            min_content_height=100,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
        )
        row.add_row(scroller)
        initial_val = getattr(var, target_property)
        if initial_val is not None:
            text_view.get_buffer().set_text(str(initial_val))
        row.core_widget = text_view  # type: ignore
        return row, cls(row, text_view)

    def get_value(self) -> Optional[Any]:
        buf = self._text_view.get_buffer()
        start, end = buf.get_start_iter(), buf.get_end_iter()
        return buf.get_text(start, end, True)

    def set_value(self, value: Any) -> None:
        self._text_view.get_buffer().set_text(str(value))

    def update_from_var(self, var: Var):
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(var.description)
