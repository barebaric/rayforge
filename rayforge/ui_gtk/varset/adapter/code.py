from typing import Any, cast

from gi.repository import Adw, Gtk

from ....core.varset import CodeVar, Var
from ...shared.code_editor import CodeEditor
from ...shared.gtk import apply_css
from .base import RowAdapter, register_adapter

# A real 1px border on the row itself, so it sits exactly at the
# card's outer edge and follows the row's rounded corners (12px, from
# the boxed-list first/last-child rules). The editor's own styling
# (transparent text node, rounded view surface) lives in CodeEditor.
_CODE_AREA_CSS = """
.code-area-row {
    border: 1px solid @borders;
}
"""


@register_adapter(CodeVar)
class CodeAreaAdapter(RowAdapter):
    """Flat, always-expanded code editor row.

    Unlike :class:`~rayforge.ui_gtk.varset.adapter.textarea.TextAreaAdapter`
    this renders no expander header: just a tall monospace editor with
    variable and macro popovers inside a plain row, so the setting is
    immediately visible and editable without unfolding.
    """

    def __init__(self, row: Adw.ActionRow, editor: CodeEditor) -> None:
        super().__init__()
        self._row = row
        self._editor = editor

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "CodeAreaAdapter"]:
        code_var = cast(CodeVar, var)
        apply_css(_CODE_AREA_CSS)

        macros = None
        if code_var.macros_provider is not None:
            macros = code_var.macros_provider()
        editor = CodeEditor(
            text=str(getattr(code_var, target_property) or ""),
            variable_context_level=code_var.variable_context_level,
            macros=macros,
            wrap_mode=Gtk.WrapMode.NONE,
            min_content_height=220,
        )
        row = Adw.ActionRow()
        row.set_activatable(False)
        row.add_css_class("code-area-row")
        row.set_child(editor)

        row.core_widget = editor.text_view  # type: ignore
        adapter = cls(row, editor)
        # Typing fires the adapter so the varset machinery picks the
        # edit up; the widget debounces these into one model update.
        editor.buffer.connect(
            "changed", lambda _b: adapter.changed.send(adapter)
        )
        return row, adapter

    def get_value(self) -> Any | None:
        return self._editor.get_text()

    def set_value(self, value: Any) -> None:
        self._editor.set_text(str(value))

    def update_from_var(self, var: Var):
        pass
