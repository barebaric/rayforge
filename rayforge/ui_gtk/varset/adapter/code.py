from typing import Any

from gi.repository import Adw, Gtk

from ....core.varset import CodeVar, Var
from .base import RowAdapter, register_adapter


@register_adapter(CodeVar)
class CodeAreaAdapter(RowAdapter):
    """Flat, always-expanded code editor row.

    Unlike :class:`~rayforge.ui_gtk.varset.adapter.textarea.TextAreaAdapter`
    this renders no expander header: just a tall monospace text area
    (220px) inside a plain row, so the setting is immediately visible
    and editable without unfolding.
    """

    def __init__(self, row: Adw.ActionRow, text_view: Gtk.TextView) -> None:
        super().__init__()
        self._row = row
        self._text_view = text_view

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "CodeAreaAdapter"]:
        row = Adw.ActionRow()
        row.set_activatable(False)

        text_view = Gtk.TextView(
            monospace=True,
            wrap_mode=Gtk.WrapMode.NONE,
            top_margin=6,
            bottom_margin=6,
            left_margin=8,
            right_margin=8,
        )
        scroller = Gtk.ScrolledWindow(
            child=text_view,
            min_content_height=220,
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        # Inset the editor from the row edges: the boxed row draws its
        # own rounded outline, and a full-bleed text view would paint
        # its differently-rounded background over that outline, making
        # the border line disappear at the corners.
        inset = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        inset.append(scroller)
        row.set_child(inset)

        initial_val = getattr(var, target_property)
        if initial_val is not None:
            text_view.get_buffer().set_text(str(initial_val))
        row.core_widget = text_view  # type: ignore
        return row, cls(row, text_view)

    def get_value(self) -> Any | None:
        buf = self._text_view.get_buffer()
        start, end = buf.get_start_iter(), buf.get_end_iter()
        return buf.get_text(start, end, True)

    def set_value(self, value: Any) -> None:
        self._text_view.get_buffer().set_text(str(value))

    def update_from_var(self, var: Var):
        pass
