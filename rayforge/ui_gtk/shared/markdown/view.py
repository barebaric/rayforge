"""Render limited Markdown using native GTK widgets."""

from html import escape
from urllib.parse import urlsplit

from gi.repository import Adw, Gtk, Pango

from ....shared.markdown import (
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
from ..expander import Expander
from ..gtk import apply_css

apply_css("""
.markdown-heading-1 { font-size: 1.8em; font-weight: bold; }
.markdown-heading-2 { font-size: 1.5em; font-weight: bold; }
.markdown-heading-3 { font-size: 1.25em; font-weight: bold; }
.markdown-heading-4, .markdown-heading-5, .markdown-heading-6 {
    font-weight: bold;
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
        self._style_handler_ids: list[int] = []
        style_manager = Adw.StyleManager.get_default()
        for notify in ("notify::dark", "notify::high-contrast"):
            self._style_handler_ids.append(
                style_manager.connect(notify, self._on_style_changed)
            )
        self.connect("destroy", self._on_destroy)
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

    def _on_style_changed(self, *_args):
        """Inline code bakes its background color into Pango markup, so
        switching between light and dark needs a full re-render.
        """
        self.set_text(self._text)

    def _on_destroy(self, *_args):
        style_manager = Adw.StyleManager.get_default()
        for handler_id in self._style_handler_ids:
            style_manager.disconnect(handler_id)
        self._style_handler_ids = []

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
                margin_top=12,
                margin_bottom=12,
                margin_start=12,
                margin_end=12,
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
