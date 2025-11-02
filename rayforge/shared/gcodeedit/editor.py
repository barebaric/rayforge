from __future__ import annotations
from typing import TYPE_CHECKING
from gi.repository import Gtk, GLib, Gdk
from blinker import Signal
from .highlighter import GcodeHighlighter

if TYPE_CHECKING:
    pass


class GcodeEditor(Gtk.Box):
    """
    A self-contained widget for displaying and editing G-code, featuring
    syntax highlighting and a search bar (Ctrl+F).
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.set_can_focus(True)
        self.line_activated = Signal()

        self.text_view = Gtk.TextView(
            monospace=True,
            wrap_mode=Gtk.WrapMode.NONE,
            pixels_above_lines=2,
            pixels_below_lines=2,
            left_margin=6,
            right_margin=6,
            can_focus=True,
        )

        self.scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
            hexpand=True,
        )
        self.scrolled_window.set_child(self.text_view)

        self.search_entry = Gtk.SearchEntry()
        self.search_bar = Gtk.SearchBar(child=self.search_entry)
        self.search_bar.set_key_capture_widget(self)
        self.search_entry.connect("search-changed", self._on_search_changed)

        buffer = self.text_view.get_buffer()
        tag_table = buffer.get_tag_table()
        self.search_tag = Gtk.TextTag(name="search")
        tag_table.add(self.search_tag)
        self.highlight_tag = Gtk.TextTag(name="highlight")
        tag_table.add(self.highlight_tag)
        self.current_highlight_line = -1

        style_context = self.get_style_context()
        found, color = style_context.lookup_color("theme_selected_bg_color")
        if found and color:
            self.search_tag.set_property("background-rgba", color)
            highlight_color = color.copy()
            highlight_color.alpha = 0.3
            self.highlight_tag.set_property("background-rgba", highlight_color)
        else:
            self.search_tag.set_property("background", "#4A90D9")
            fallback_rgba = Gdk.RGBA()
            fallback_rgba.parse("#fce94f")
            fallback_rgba.alpha = 0.3
            self.highlight_tag.set_property("background-rgba", fallback_rgba)

        self.append(self.search_bar)
        self.append(self.scrolled_window)

        self.highlighter = GcodeHighlighter(self.text_view)

        self.connect("map", self._on_map)
        self.connect("unmap", self._on_unmap)

        # Connect to cursor movement to detect line activation
        buffer.connect("mark-set", self._on_cursor_move)

    def _on_cursor_move(self, buffer, location, mark):
        """Fires the line-activated signal when the cursor moves."""
        if mark.get_name() == "insert":
            line_number = location.get_line()
            self.line_activated.send(self, line_number=line_number)

    def _on_search_changed(self, search_entry: Gtk.SearchEntry):
        """Callback to highlight search results in the text buffer."""
        buffer = self.text_view.get_buffer()
        text = search_entry.get_text()

        buffer.remove_tag(
            self.search_tag, buffer.get_start_iter(), buffer.get_end_iter()
        )

        if not text:
            return

        current_iter = buffer.get_start_iter()
        while True:
            try:
                result = current_iter.forward_search(
                    text, Gtk.TextSearchFlags.CASE_INSENSITIVE, None
                )

                if result is None:
                    break

                start, end = result
                buffer.apply_tag(self.search_tag, start, end)
                current_iter = end
            except GLib.Error:
                break

    def _on_map(self, widget: Gtk.Widget):
        """Starts the live highlighter when the widget is shown."""
        self.highlighter.start()
        buffer = self.text_view.get_buffer()
        self.highlighter.highlight(
            buffer.get_start_iter(), buffer.get_end_iter()
        )

    def _on_unmap(self, widget: Gtk.Widget):
        """Stops the live highlighter when the widget is hidden."""
        self.highlighter.stop()

    def get_text(self) -> str:
        """Returns the full text content of the editor."""
        buffer = self.text_view.get_buffer()
        start, end = buffer.get_start_iter(), buffer.get_end_iter()
        return buffer.get_text(start, end, include_hidden_chars=True)

    def set_text(self, text: str):
        """
        Sets the text content of the editor and triggers a full highlight.
        """
        buffer = self.text_view.get_buffer()
        buffer.set_text(text, -1)
        self.highlighter.highlight(
            buffer.get_start_iter(), buffer.get_end_iter()
        )
        self.search_bar.set_search_mode(False)

    def highlight_line(self, line_number: int, use_align: bool = True):
        buffer = self.text_view.get_buffer()

        # Remove the old highlight if it exists
        if self.current_highlight_line != -1:
            found, start_iter = buffer.get_iter_at_line(
                self.current_highlight_line
            )
            if found:
                end_iter = start_iter.copy()
                if not end_iter.ends_line():
                    end_iter.forward_to_line_end()
                buffer.remove_tag(self.highlight_tag, start_iter, end_iter)

        # Add the new highlight if valid
        if line_number != -1:
            found, start_iter = buffer.get_iter_at_line(line_number)
            if found:
                end_iter = start_iter.copy()
                if not end_iter.ends_line():
                    end_iter.forward_to_line_end()
                buffer.apply_tag(self.highlight_tag, start_iter, end_iter)
                # Scroll to the highlighted line with optional alignment
                yalign = 0.5 if use_align else 0.0
                self.text_view.scroll_to_iter(
                    start_iter, 0.0, use_align, 0.5, yalign
                )

        self.current_highlight_line = line_number

    def clear_highlight(self):
        self.highlight_line(-1)

    @property
    def text(self) -> str:
        """The text content of the editor."""
        return self.get_text()

    @text.setter
    def text(self, value: str):
        self.set_text(value)

    def insert_text_at_cursor(self, text: str):
        """Inserts the given text at the current cursor position."""
        buffer = self.text_view.get_buffer()
        buffer.insert_at_cursor(text, -1)
