"""Reusable native GTK widgets for Rayforge's limited Markdown subset."""

from gettext import gettext as _
from html import escape
from typing import ClassVar
from urllib.parse import urlsplit

from gi.repository import Adw, GLib, GObject, Gtk, Pango

from ...shared.markdown import (
    BlockQuote,
    CodeBlock,
    Details,
    Emphasis,
    Heading,
    InlineCode,
    LineBreak,
    Link,
    ListBlock,
    MarkdownParser,
    Paragraph,
    Strong,
    Text,
)
from ..icons import get_icon
from .even_split_paned import EvenSplitPaned
from .expander import Expander
from .gtk import apply_css

apply_css("""
.markdown-heading-1 { font-size: 1.8em; font-weight: bold; }
.markdown-heading-2 { font-size: 1.5em; font-weight: bold; }
.markdown-heading-3 { font-size: 1.25em; font-weight: bold; }
.markdown-heading-4, .markdown-heading-5, .markdown-heading-6 {
    font-weight: bold;
}
.markdown-inline-code {
    font-family: monospace;
    background-color: alpha(@headerbar_bg_color, 0.7);
    border-radius: 4px;
    padding: 3px 5px;
}
.markdown-code, .markdown-code-in-details {
    font-family: monospace;
    border: 1px solid @borders;
    border-radius: 6px;
    padding: 8px;
}
.markdown-code {
    background-color: @view_bg_color;
}
.markdown-code-in-details {
    background-color: @headerbar_bg_color;
}
.markdown-block-quote {
    border-left: 3px solid @borders;
    padding-left: 12px;
}
""")

_SUPPORTED_HINT = _(
    "Supported: headings, paragraphs, bold, italic, lists, links, "
    "inline code, fenced code blocks, and expandable details sections."
)

_FENCE_MARKER = "```"


def build_help_markdown() -> str:
    """Describe the supported syntax using the supported syntax itself.

    Only the prose is translated; every example stays a literal so a
    translation can never break the syntax it demonstrates.
    """
    topics = (
        (
            _("Bold and italic"),
            _("Wrap words in two stars for bold, or one star for italic."),
            "**bold text**\n*italic text*",
        ),
        (
            _("Headings"),
            _("Start a line with number signs. More signs, smaller heading."),
            "# Main title\n## Section\n### Subsection",
        ),
        (
            _("Lists"),
            _(
                "Start lines with a dash for bullets, or with a number and a "
                "dot for a numbered list."
            ),
            "- First item\n- Second item\n\n1. First step\n2. Second step",
        ),
        (
            _("Links"),
            _(
                "Put the link text in square brackets and the address in "
                "round brackets."
            ),
            "[Rayforge website](https://rayforge.org)",
        ),
        (
            _("Inline code"),
            _(
                "Wrap short commands in single backticks to keep them "
                "readable."
            ),
            "Send `G0 X0 Y0` to move to the origin.",
        ),
        (
            _("Quotes"),
            _("Start a line with a greater-than sign to call out a remark."),
            "> Always wear laser safety glasses.",
        ),
        (
            _("Expandable sections"),
            _("Hide long details behind a title that readers can unfold."),
            (
                ":::details Advanced tuning\n"
                "Only needed for thick material.\n"
                ":::enddetails"
            ),
        ),
    )
    parts = [
        f"# {_('Formatting help')}",
        _("Type the text shown below to get the matching result."),
    ]
    for name, description, example in topics:
        parts.append(f"## {name}")
        parts.append(description)
        parts.append(f"{_FENCE_MARKER}\n{example}\n{_FENCE_MARKER}")
    parts.append(f"## {_('Code blocks')}")
    parts.append(
        _(
            "Put a line with three backticks before and after a block of "
            "commands to show it unchanged."
        )
    )
    return "\n\n".join(parts)


class MarkdownView(Gtk.Box):
    """Render limited Markdown using native GTK widgets."""

    def __init__(self, text: str = "", **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=8, **kwargs
        )
        self._parser = MarkdownParser()
        self._text = ""
        self._details_expanded: dict[tuple[str, int], bool] = {}
        self._details_seen: dict[str, int] = {}
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self._text = text
        self._details_seen = {}
        while child := self.get_first_child():
            self.remove(child)
        document = self._parser.parse(text)
        for block in document.blocks:
            self.append(self._render_block(block))
        self._forget_removed_details()

    def _forget_removed_details(self) -> None:
        """Drop remembered states of sections the new text no longer has.

        Without this the expanded/collapsed state of every section a user
        ever opened would be retained, even after renaming or deleting it.
        """
        self._details_expanded = {
            key: expanded
            for key, expanded in self._details_expanded.items()
            if key[1] < self._details_seen.get(key[0], 0)
        }

    def get_text(self) -> str:
        return self._text

    def _render_block(self, block, in_details: bool = False):
        if isinstance(block, Heading):
            widget = self._render_inlines(block.children)
            widget.add_css_class(f"markdown-heading-{block.level}")
            return widget
        if isinstance(block, Paragraph):
            return self._render_inlines(block.children)
        if isinstance(block, ListBlock):
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            for index, item in enumerate(block.items, 1):
                prefix = f"{index}. " if block.ordered else "• "
                row = Gtk.Box(spacing=6)
                label = Gtk.Label(
                    label=prefix, xalign=0, valign=Gtk.Align.START
                )
                label.add_css_class("dim-label")
                row.append(label)
                content = Gtk.Box(
                    orientation=Gtk.Orientation.VERTICAL,
                    spacing=4,
                    hexpand=True,
                )
                for child in item.blocks:
                    content.append(self._render_block(child, in_details))
                row.append(content)
                row.set_hexpand(True)
                box.append(row)
            return box
        if isinstance(block, BlockQuote):
            box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            box.add_css_class("markdown-block-quote")
            for child in block.blocks:
                box.append(self._render_block(child, in_details))
            return box
        if isinstance(block, CodeBlock):
            label = Gtk.Label(
                label=block.value,
                xalign=0,
                selectable=True,
                hexpand=True,
            )
            label.set_wrap(True)
            label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
            label.add_css_class(
                "markdown-code-in-details" if in_details else "markdown-code"
            )
            return label
        if isinstance(block, Details):
            expander = Expander()
            expander.set_title(block.title)
            content = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                spacing=6,
                margin_start=12,
                margin_end=12,
                margin_bottom=12,
            )
            for child in block.blocks:
                content.append(self._render_block(child, in_details=True))
            expander.set_child(content)
            key = self._details_key(block.title)
            expander.set_expanded(self._details_expanded.get(key, False))
            expander.revealer.connect(
                "notify::reveal-child", self._on_details_toggled, key
            )
            return expander
        raise TypeError(f"Unsupported Markdown block: {type(block)!r}")

    def _details_key(self, title: str) -> tuple[str, int]:
        """Identify a details section by title and duplicate occurrence."""
        index = self._details_seen.get(title, 0)
        self._details_seen[title] = index + 1
        return title, index

    def _on_details_toggled(self, revealer, _param, key):
        self._details_expanded[key] = revealer.get_reveal_child()

    def _render_inlines(self, nodes):
        label = Gtk.Label(xalign=0, hexpand=True)
        label.set_wrap(True)
        label.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        label.set_markup(self._inline_markup(nodes))
        return label

    @classmethod
    def _inline_markup(cls, nodes) -> str:
        parts = []
        for node in nodes:
            if isinstance(node, Text):
                parts.append(escape(node.value))
            elif isinstance(node, LineBreak):
                parts.append("\n")
            elif isinstance(node, InlineCode):
                background = (
                    "#42464c"
                    if Adw.StyleManager.get_default().get_dark()
                    else "#e8edf2"
                )
                parts.append(
                    f'<span background="{background}">'
                    f"<tt>{escape(node.value)}</tt></span>"
                )
            elif isinstance(node, Link):
                label = cls._inline_markup(node.label)
                if cls._is_safe_link(node.destination):
                    destination = escape(node.destination, quote=True)
                    parts.append(f'<a href="{destination}">{label}</a>')
                else:
                    parts.append(label)
            elif isinstance(node, Emphasis):
                parts.append(f"<i>{cls._inline_markup(node.children)}</i>")
            elif isinstance(node, Strong):
                parts.append(f"<b>{cls._inline_markup(node.children)}</b>")
            else:
                raise TypeError(f"Unsupported Markdown inline: {type(node)!r}")
        return "".join(parts)

    @staticmethod
    def _is_safe_link(destination: str) -> bool:
        try:
            parsed_url = urlsplit(destination)
        except ValueError:
            return False
        is_web_link = parsed_url.scheme.lower() in ("http", "https") and bool(
            parsed_url.netloc
        )
        is_email_link = parsed_url.scheme.lower() == "mailto" and bool(
            parsed_url.path
        )
        return is_web_link or is_email_link


class MarkdownEditor(Gtk.Box):
    """A raw Markdown multiline editor."""

    def __init__(self, text: str = "", **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, **kwargs
        )
        self.text_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            monospace=True,
            top_margin=10,
            bottom_margin=10,
            left_margin=10,
            right_margin=10,
        )
        self.text_view.set_vexpand(True)
        self.text_view.set_hexpand(True)
        self.scrolled_window = Gtk.ScrolledWindow(child=self.text_view)
        self.scrolled_window.set_vexpand(True)
        self.scrolled_window.set_hexpand(True)
        self.append(self.scrolled_window)
        self.set_text(text)

    @property
    def buffer(self) -> Gtk.TextBuffer:
        return self.text_view.get_buffer()

    def get_text(self) -> str:
        start, end = self.buffer.get_bounds()
        return self.buffer.get_text(start, end, True)

    def set_text(self, text: str) -> None:
        self.buffer.set_text(text)

    def wrap_selection(
        self, prefix: str, suffix: str, placeholder: str
    ) -> None:
        """Surround the selection with markers, or insert an example."""
        start, end = self._selection()
        selected = self.buffer.get_text(start, end, True) or placeholder
        self._replace(start, end, f"{prefix}{selected}{suffix}")

    def prefix_lines(self, prefix: str, placeholder: str) -> None:
        """Prefix every selected line, numbering it when needed."""
        start, end = self._selection()
        start.set_line_offset(0)
        if not end.ends_line():
            end.forward_to_line_end()
        selected = self.buffer.get_text(start, end, True) or placeholder
        lines = selected.split("\n")
        numbered = prefix.endswith(". ")
        rendered = [
            f"{index}. {line}" if numbered else f"{prefix}{line}"
            for index, line in enumerate(lines, 1)
        ]
        self._replace(start, end, "\n".join(rendered))

    def insert_block(self, template: str, placeholder: str) -> None:
        """Insert a multi-line snippet around the selection."""
        start, end = self._selection()
        selected = self.buffer.get_text(start, end, True) or placeholder
        block = template.format(content=selected)
        start.set_line_offset(0)
        if not end.ends_line():
            end.forward_to_line_end()
        if start.get_line() > 0:
            block = f"\n{block}"
        self._replace(start, end, f"{block}\n")

    def _selection(self) -> tuple[Gtk.TextIter, Gtk.TextIter]:
        bounds = self.buffer.get_selection_bounds()
        if bounds:
            return bounds
        cursor = self.buffer.get_iter_at_mark(self.buffer.get_insert())
        return cursor, cursor.copy()

    def _replace(
        self, start: Gtk.TextIter, end: Gtk.TextIter, text: str
    ) -> None:
        self.buffer.begin_user_action()
        self.buffer.delete(start, end)
        self.buffer.insert(start, text)
        self.buffer.end_user_action()
        self.text_view.grab_focus()


class MarkdownToolbar(Gtk.Box):
    """Buttons that insert the supported Markdown syntax for the user."""

    __gsignals__: ClassVar = {
        "help-toggled": (GObject.SignalFlags.RUN_LAST, None, (bool,)),
    }

    def __init__(self, editor: "MarkdownEditor", **kwargs):
        super().__init__(spacing=6, **kwargs)
        self._editor = editor
        actions = Gtk.Box()
        actions.add_css_class("linked")
        self.append(actions)
        self.buttons: dict[str, Gtk.Button] = {}
        for name, label, tooltip, callback in self._action_specs():
            button = Gtk.Button(tooltip_text=tooltip)
            button.set_child(self._make_label(label))
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
                "<b>B</b>",
                _("Bold"),
                lambda *_a: self._editor.wrap_selection(
                    "**", "**", _("bold text")
                ),
            ),
            (
                "italic",
                "<i>I</i>",
                _("Italic"),
                lambda *_a: self._editor.wrap_selection("*", "*", _("italic")),
            ),
            (
                "heading",
                "H",
                _("Heading"),
                lambda *_a: self._editor.prefix_lines("## ", _("Heading")),
            ),
            (
                "bullet-list",
                "\u2022",
                _("Bulleted list"),
                lambda *_a: self._editor.prefix_lines("- ", _("List item")),
            ),
            (
                "numbered-list",
                "1.",
                _("Numbered list"),
                lambda *_a: self._editor.prefix_lines(". ", _("First step")),
            ),
            (
                "link",
                "\U0001f517",
                _("Link"),
                lambda *_a: self._editor.wrap_selection(
                    "[", "](https://example.com)", _("link text")
                ),
            ),
            (
                "inline-code",
                "<tt>&lt;/&gt;</tt>",
                _("Inline code"),
                lambda *_a: self._editor.wrap_selection(
                    "`", "`", _("command")
                ),
            ),
            (
                "code-block",
                "<tt>{ }</tt>",
                _("Code block"),
                lambda *_a: self._editor.insert_block(
                    f"{_FENCE_MARKER}\n{{content}}\n{_FENCE_MARKER}",
                    "G0 X0 Y0",
                ),
            ),
            (
                "quote",
                "\u201c",
                _("Quote"),
                lambda *_a: self._editor.prefix_lines("> ", _("Remark")),
            ),
            (
                "details",
                "\u25be",
                _("Expandable section"),
                lambda *_a: self._editor.insert_block(
                    ":::details "
                    + _("Section title")
                    + "\n{content}\n:::enddetails",
                    _("Hidden details"),
                ),
            ),
        )

    @staticmethod
    def _make_label(markup: str) -> Gtk.Label:
        label = Gtk.Label()
        label.set_markup(markup)
        label.set_width_chars(2)
        return label

    def _on_help_toggled(self, button):
        self.emit("help-toggled", button.get_active())

    def set_help_active(self, active: bool) -> None:
        if self.help_button.get_active() != active:
            self.help_button.set_active(active)


class MarkdownPreviewEditor(Gtk.Box):
    """Compose a raw editor and preview with debounced updates."""

    SIDE_BY_SIDE = "side-by-side"
    STACKED = "stacked"
    _NARROW_WIDTH = 700

    def __init__(self, text: str = "", **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=8, **kwargs
        )
        self.editor = MarkdownEditor(text)
        self.preview = MarkdownView(text)
        self.help_view = MarkdownView(build_help_markdown())
        self.toolbar = MarkdownToolbar(self.editor)
        self.toolbar.connect("help-toggled", self._on_help_toggled)
        self.layout_dropdown = Gtk.DropDown()
        self.layout_dropdown.set_model(
            Gtk.StringList.new([_("Side by side"), _("Stacked")])
        )
        self.layout_dropdown.set_selected(1)
        self.layout_dropdown.connect("notify::selected", self._layout_changed)
        self._updating_layout = False
        self._paned = EvenSplitPaned(orientation=Gtk.Orientation.VERTICAL)
        self.preview_scroller = Gtk.ScrolledWindow(child=self.preview)
        self.preview_scroller.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.preview_scroller.set_vexpand(True)
        self.preview_scroller.set_hexpand(True)
        self.help_scroller = Gtk.ScrolledWindow(child=self.help_view)
        self.help_scroller.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.help_scroller.set_vexpand(True)
        self.help_scroller.set_hexpand(True)
        self.preview_stack = Gtk.Stack()
        self.preview_stack.add_named(self.preview_scroller, "preview")
        self.preview_stack.add_named(self.help_scroller, "help")
        self.editor_pane, _editor_title = self._make_pane(
            _("Raw Markdown"), self.editor
        )
        self.preview_pane, self.preview_title = self._make_pane(
            _("Preview"), self.preview_stack
        )
        for pane in (self.editor_pane, self.preview_pane):
            pane.set_hexpand(True)
            pane.set_vexpand(True)
        self._paned.set_start_child(self.editor_pane)
        self._paned.set_end_child(self.preview_pane)
        self.action_bar = Gtk.Box(spacing=12)
        self.toolbar.set_hexpand(True)
        self.action_bar.append(self.toolbar)
        self.action_bar.append(self.layout_dropdown)
        self.append(self.action_bar)
        self.append(self._paned)
        self._paned.set_vexpand(True)
        self._user_selected_layout = False
        self._timeout_id = 0
        self.editor.buffer.connect("changed", self._on_source_changed)
        self._paned.connect("notify::max-position", self._on_paned_resized)
        self._on_width_changed()

    def _on_paned_resized(self, *_args):
        """Allocation changes may make the responsive layout stale."""
        self._on_width_changed()

    @staticmethod
    def _make_pane(title: str, child: Gtk.Widget) -> tuple[Gtk.Box, Gtk.Label]:
        pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        label = Gtk.Label(label=title, xalign=0)
        label.add_css_class("heading")
        pane.append(label)
        pane.append(child)
        return pane, label

    def _on_help_toggled(self, _toolbar, active: bool):
        self.preview_stack.set_visible_child_name(
            "help" if active else "preview"
        )
        self.preview_title.set_label(
            _("Formatting help") if active else _("Preview")
        )

    def show_help(self, active: bool) -> None:
        """Swap the preview pane for the formatting help."""
        self.toolbar.set_help_active(active)

    def _layout_changed(self, dropdown, _pspec):
        if self._updating_layout:
            return
        self._user_selected_layout = True
        self.set_layout(
            self.STACKED if dropdown.get_selected() == 1 else self.SIDE_BY_SIDE
        )

    def _on_width_changed(self, *_args):
        if self._user_selected_layout:
            return
        self.set_layout(
            self.STACKED
            if not self.get_width() or self.get_width() < self._NARROW_WIDTH
            else self.SIDE_BY_SIDE,
        )

    def set_layout(self, layout: str, update_dropdown: bool = True) -> None:
        if layout not in (self.SIDE_BY_SIDE, self.STACKED):
            raise ValueError(f"Unknown Markdown layout: {layout}")
        orientation = (
            Gtk.Orientation.VERTICAL
            if layout == self.STACKED
            else Gtk.Orientation.HORIZONTAL
        )
        if self._paned.get_orientation() != orientation:
            self._paned.set_orientation(orientation)
            self._paned.reset_split()
        if update_dropdown:
            self._updating_layout = True
            try:
                self.layout_dropdown.set_selected(
                    1 if layout == self.STACKED else 0
                )
            finally:
                self._updating_layout = False

    def get_layout(self) -> str:
        return (
            self.STACKED
            if self._paned.get_orientation() == Gtk.Orientation.VERTICAL
            else self.SIDE_BY_SIDE
        )

    def _on_source_changed(self, *_args):
        self.show_help(False)
        if self._timeout_id:
            GLib.source_remove(self._timeout_id)
        self._timeout_id = GLib.timeout_add(250, self._refresh_preview)

    def _refresh_preview(self):
        self.preview.set_text(self.editor.get_text())
        self._timeout_id = 0
        return GLib.SOURCE_REMOVE

    def refresh_preview(self) -> None:
        if self._timeout_id:
            GLib.source_remove(self._timeout_id)
            self._timeout_id = 0
        self._refresh_preview()


class MarkdownEditorDialog(Adw.Window):
    """Dialog for editing raw Markdown and its live preview."""

    __gsignals__: ClassVar = {
        "saved": (GObject.SignalFlags.RUN_LAST, None, (str,)),
        "cancelled": (GObject.SignalFlags.RUN_LAST, None, ()),
    }

    def __init__(
        self,
        title: str,
        description: str | None = None,
        initial_text: str = "",
        **kwargs,
    ):
        super().__init__(modal=True, **kwargs)
        self.set_title(title)
        self.set_default_size(1150, 750)
        self._result: str | None = None
        self._action_closed = False
        self.preview_editor = MarkdownPreviewEditor(initial_text)

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        header_bar.set_show_end_title_buttons(False)
        toolbar_view.add_top_bar(header_bar)

        self.cancel_button = Gtk.Button(label=_("Cancel"))
        self.cancel_button.connect("clicked", self._on_cancel)
        header_bar.pack_start(self.cancel_button)

        self.save_button = Gtk.Button(label=_("Save"))
        self.save_button.add_css_class("suggested-action")
        self.save_button.connect("clicked", self._on_save)
        header_bar.pack_end(self.save_button)

        root = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=18,
            margin_bottom=18,
            margin_start=18,
            margin_end=18,
        )
        if description:
            root.append(Gtk.Label(label=description, xalign=0, wrap=True))
        root.append(self.preview_editor)
        hint = Gtk.Label(label=_SUPPORTED_HINT, xalign=0, wrap=True)
        hint.add_css_class("dim-label")
        root.append(hint)
        toolbar_view.set_content(root)

    def get_text(self) -> str:
        return self.preview_editor.editor.get_text()

    def get_result(self) -> str | None:
        return self._result

    def _on_save(self, *_args):
        self._result = self.get_text()
        self._action_closed = True
        self.emit("saved", self._result)
        self.close()

    def _on_cancel(self, *_args):
        self._result = None
        self._action_closed = True
        self.emit("cancelled")
        self.close()

    def do_close_request(self):
        if not self._action_closed:
            self._action_closed = True
            self.emit("cancelled")
        return False


__all__ = [
    "MarkdownEditor",
    "MarkdownEditorDialog",
    "MarkdownPreviewEditor",
    "MarkdownToolbar",
    "MarkdownView",
    "build_help_markdown",
]
