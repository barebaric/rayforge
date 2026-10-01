"""Shared monospace code editor with insertion toolbars.

Used by the macro editor dialog and by ``CodeVar``-backed rows: a
monospace text view with a small toolbar offering two popovers —
one listing the template placeholders (path variables) available at
a given context level, one listing the other macros that can be
included via ``@include(Name)``.
"""

from gettext import gettext as _

from gi.repository import Adw, GLib, Gtk

from ...machine.models.macro import Macro
from ...pipeline.encoder.context import GcodeContext
from ..icons import get_icon
from .gtk import apply_css

# The text surface is a "view" box below the toolbar: darker theme
# surface, clearly distinct from the toolbar's card background. Its
# top corners stay square so it meets the toolbar flush; only the
# bottom corners are rounded, following the card's curvature. The text
# view and its text node stay transparent: an opaque text node would
# paint a square-cornered patch over the surface's rounded corners.
apply_css(
    """
.code-editor-surface {
    border-radius: 0 0 10px 10px;
}
.code-area-editor,
.code-area-editor text {
    background: transparent;
}
"""
)


class CodeEditor(Gtk.Box):
    """A monospace code text view with variable and macro popovers.

    Args:
        text: Initial content.
        variable_context_level: Context level for the placeholder
            documentation (``job``, ``layer``, or ``workpiece``).
        macros: Macros offered for inclusion; the macro being edited
            is excluded via *exclude_macro_uid*.
        exclude_macro_uid: UID to filter out of the macros popover.
        wrap_mode: Text view wrap mode.
        min_content_height: Height of the scrolled text area.
    """

    def __init__(
        self,
        text: str = "",
        *,
        variable_context_level: str = "job",
        macros: list[Macro] | None = None,
        exclude_macro_uid: str | None = None,
        wrap_mode: Gtk.WrapMode = Gtk.WrapMode.WORD_CHAR,
        min_content_height: int = 250,
        **kwargs,
    ):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, **kwargs
        )
        self._exclude_macro_uid = exclude_macro_uid

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_top(3)
        toolbar.set_margin_bottom(3)
        toolbar.set_margin_start(6)
        toolbar.set_margin_end(6)

        variables_button = Gtk.MenuButton(
            child=get_icon("variable-symbolic"),
            tooltip_text=_("Insert Variable"),
        )
        variables_button.add_css_class("flat")
        toolbar.append(variables_button)
        self._build_variables_popover(variables_button, variable_context_level)

        macros_button = Gtk.MenuButton(
            child=get_icon("code-symbolic"),
            tooltip_text=_("Include Macro"),
        )
        macros_button.add_css_class("flat")
        toolbar.append(macros_button)
        self._build_macros_popover(macros_button, macros or [])

        self.text_view = Gtk.TextView(
            wrap_mode=wrap_mode,
            monospace=True,
            pixels_above_lines=2,
            pixels_below_lines=2,
            left_margin=6,
            right_margin=6,
        )
        self.text_view.add_css_class("code-area-editor")
        self._scrolled_window = Gtk.ScrolledWindow(
            child=self.text_view,
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=min_content_height,
        )
        self._scrolled_window.add_css_class("view")
        self._scrolled_window.add_css_class("code-editor-surface")

        self._toolbar_view = Adw.ToolbarView()
        self._toolbar_view.add_top_bar(toolbar)
        self._toolbar_view.set_content(self._scrolled_window)
        self._toolbar_view.set_vexpand(True)
        self._toolbar_view.set_hexpand(True)
        self.append(self._toolbar_view)

        self.set_text(text)

    @property
    def buffer(self) -> Gtk.TextBuffer:
        return self.text_view.get_buffer()

    def get_text(self) -> str:
        start, end = self.buffer.get_bounds()
        return self.buffer.get_text(start, end, True)

    def set_text(self, text: str) -> None:
        self.buffer.set_text(text, -1)

    def _insert_text_at_cursor(self, text: str) -> None:
        """Insert text at the current cursor position."""
        buffer = self.text_view.get_buffer()
        insert_mark = buffer.get_insert()
        iterator = buffer.get_iter_at_mark(insert_mark)
        buffer.insert(iterator, text, -1)

    def _build_variables_popover(
        self, parent_button: Gtk.MenuButton, level: str
    ) -> None:
        """Create the popover listing the template placeholders."""
        self.variables_popover = Gtk.Popover()
        parent_button.set_popover(self.variables_popover)
        self.variables_popover.connect("closed", self._on_popover_closed)

        # A popover sizes to its content, whose natural width is
        # narrow; force a readable minimum instead.
        clamp = Adw.Clamp(maximum_size=500)
        clamp.set_size_request(420, -1)
        self.variables_popover.set_child(clamp)

        popover_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        clamp.set_child(popover_box)

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER, min_content_height=250
        )
        popover_box.append(scrolled_window)

        list_box = Gtk.ListBox()
        list_box.add_css_class("boxed-list")
        scrolled_window.set_child(list_box)

        variables = GcodeContext.get_docs(level)
        for var, desc in variables:
            row = Adw.ActionRow(subtitle=desc, activatable=True)
            escaped_var = GLib.markup_escape_text(f"{{{var}}}")
            row.set_title(
                f'<span font_family="monospace">{escaped_var}</span>'
            )
            row.set_use_markup(True)
            row.connect("activated", self._on_variable_activated, var)
            list_box.append(row)

    def _build_macros_popover(
        self, parent_button: Gtk.MenuButton, macros: list[Macro]
    ) -> None:
        """Create the popover for including other macros."""
        self.macros_popover = Gtk.Popover()
        parent_button.set_popover(self.macros_popover)
        self.macros_popover.connect("closed", self._on_popover_closed)

        clamp = Adw.Clamp(maximum_size=500)
        clamp.set_size_request(420, -1)
        self.macros_popover.set_child(clamp)

        popover_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        clamp.set_child(popover_box)

        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER, min_content_height=150
        )
        popover_box.append(scrolled_window)

        list_box = Gtk.ListBox()
        list_box.add_css_class("boxed-list")
        scrolled_window.set_child(list_box)

        macros_to_include = [
            m for m in macros if m.uid != self._exclude_macro_uid
        ]

        if macros_to_include:
            for macro in sorted(macros_to_include, key=lambda s: s.name):
                row = Adw.ActionRow(title=macro.name, activatable=True)
                row.connect("activated", self._on_macro_activated, macro.name)
                list_box.append(row)
        else:
            placeholder = Gtk.Label(label=_("No other macros to include."))
            placeholder.add_css_class("dim-label")
            placeholder.set_margin_top(12)
            placeholder.set_margin_bottom(12)
            list_box.append(
                Gtk.ListBoxRow(child=placeholder, selectable=False)
            )

    def _on_popover_closed(self, popover: Gtk.Popover) -> None:
        """Ensure the text view regains focus when a popover is closed."""
        self.text_view.grab_focus()

    def _on_variable_activated(self, row: Adw.ActionRow, variable: str):
        """Insert the clicked variable placeholder at the cursor."""
        self._insert_text_at_cursor(f"{{{variable}}}")
        self.variables_popover.popdown()

    def _on_macro_activated(self, row: Adw.ActionRow, macro_name: str):
        """Insert the clicked macro include at the cursor."""
        self._insert_text_at_cursor(f"@include({macro_name})")
        self.macros_popover.popdown()
