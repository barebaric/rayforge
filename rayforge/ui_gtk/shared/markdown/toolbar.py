"""Toolbar that inserts the supported Markdown syntax for the user."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...icons import get_icon
from ..gtk import apply_css
from .editor import MarkdownEditor

apply_css("""
.markdown-toolbar button {
    padding-left: 12px;
    padding-right: 12px;
}
""")


class MarkdownToolbar(Gtk.Box):
    """Buttons that insert the supported Markdown syntax for the user."""

    def __init__(self, editor: MarkdownEditor, **kwargs):
        super().__init__(spacing=6, **kwargs)
        self.add_css_class("markdown-toolbar")
        self._editor = editor
        self.help_toggled = Signal()
        actions = Gtk.Box()
        actions.add_css_class("linked")
        self.append(actions)
        self.buttons: dict[str, Gtk.Button] = {}
        for name, icon_name, tooltip, callback in self._action_specs():
            button = Gtk.Button(tooltip_text=tooltip)
            button.set_child(get_icon(icon_name))
            button.connect("clicked", callback)
            actions.append(button)
            self.buttons[name] = button

        spacer = Gtk.Box(hexpand=True)
        self.append(spacer)

        self.help_button = Gtk.ToggleButton(
            tooltip_text=_("Show formatting help")
        )
        self.help_button.set_child(get_icon("question-mark-symbolic"))
        self.help_button.connect("toggled", self._on_help_toggled)
        self.append(self.help_button)

    def _action_specs(self):
        return (
            (
                "bold",
                "format-text-bold-symbolic",
                _("Bold"),
                lambda *_a: self._editor.wrap_selection(
                    "**", "**", _("bold text")
                ),
            ),
            (
                "italic",
                "format-text-italic-symbolic",
                _("Italic"),
                lambda *_a: self._editor.wrap_selection("*", "*", _("italic")),
            ),
            (
                "heading",
                "format-text-heading-symbolic",
                _("Heading"),
                lambda *_a: self._editor.prefix_lines("## ", _("Heading")),
            ),
            (
                "bullet-list",
                "view-list-bullet-symbolic",
                _("Bulleted list"),
                lambda *_a: self._editor.prefix_lines("- ", _("List item")),
            ),
            (
                "numbered-list",
                "view-list-ordered-symbolic",
                _("Numbered list"),
                lambda *_a: self._editor.prefix_lines(". ", _("First step")),
            ),
            (
                "link",
                "link-symbolic",
                _("Link"),
                lambda *_a: self._editor.wrap_selection(
                    "[", "](https://example.com)", _("link text")
                ),
            ),
            (
                "inline-code",
                "code-symbolic",
                _("Inline code"),
                lambda *_a: self._editor.wrap_selection(
                    "`", "`", _("command")
                ),
            ),
            (
                "code-block",
                "code-block-symbolic",
                _("Code block"),
                lambda *_a: self._editor.insert_block(
                    "```\n{content}\n```",
                    "G0 X0 Y0",
                ),
            ),
            (
                "quote",
                "format-quote-symbolic",
                _("Quote"),
                lambda *_a: self._editor.prefix_lines("> ", _("Remark")),
            ),
            (
                "details",
                "details-symbolic",
                _("Expandable section"),
                lambda *_a: self._editor.insert_block(
                    ":::details "
                    + _("Section title")
                    + "\n{content}\n:::enddetails",
                    _("Hidden details"),
                ),
            ),
        )

    def _on_help_toggled(self, button):
        self.help_toggled.send(self, active=button.get_active())

    def set_help_active(self, active: bool) -> None:
        if self.help_button.get_active() != active:
            self.help_button.set_active(active)
