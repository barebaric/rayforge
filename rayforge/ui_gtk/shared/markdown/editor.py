"""Editor widget for raw Markdown source text."""

from gi.repository import Gtk


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
        self._replace(
            start,
            end,
            f"{prefix}{selected}{suffix}",
            (len(prefix), len(selected)),
        )

    def prefix_lines(self, prefix: str, placeholder: str) -> None:
        """Prefix every selected line, numbering it when needed."""
        start, end = self._selection()
        start.set_line_offset(0)
        if not end.ends_line():
            end.forward_to_line_end()
        selected = self.buffer.get_text(start, end, True) or placeholder
        lines = selected.split("\n")
        numbered = prefix.endswith(". ")
        rendered = "\n".join(
            f"{index}. {line}" if numbered else f"{prefix}{line}"
            for index, line in enumerate(lines, 1)
        )
        self._replace(start, end, rendered, (0, len(rendered)))

    def insert_block(self, template: str, placeholder: str) -> None:
        """Insert a multi-line snippet around the selection."""
        start, end = self._selection()
        selected = self.buffer.get_text(start, end, True) or placeholder
        block = template.format(content=selected)
        content_at = template.index("{content}")
        start.set_line_offset(0)
        if not end.ends_line():
            end.forward_to_line_end()
        if start.get_line() > 0:
            block = f"\n{block}"
            content_at += 1
        self._replace(start, end, f"{block}\n", (content_at, len(selected)))

    def _selection(self) -> tuple[Gtk.TextIter, Gtk.TextIter]:
        bounds = self.buffer.get_selection_bounds()
        if bounds:
            return bounds
        cursor = self.buffer.get_iter_at_mark(self.buffer.get_insert())
        return cursor, cursor.copy()

    def _replace(
        self,
        start: Gtk.TextIter,
        end: Gtk.TextIter,
        text: str,
        select: tuple[int, int] | None = None,
    ) -> None:
        """Replace a range, optionally re-selecting part of the new text.

        The *select* offset and length are relative to the inserted text,
        so a toolbar action keeps the affected text highlighted instead of
        dropping the cursor after it.
        """
        self.buffer.begin_user_action()
        offset = start.get_offset()
        self.buffer.delete(start, end)
        self.buffer.insert(self.buffer.get_iter_at_offset(offset), text)
        if select is not None:
            select_start, length = select
            begin = self.buffer.get_iter_at_offset(offset + select_start)
            stop = self.buffer.get_iter_at_offset(
                offset + select_start + length
            )
            self.buffer.select_range(begin, stop)
        self.buffer.end_user_action()
        self.text_view.grab_focus()
