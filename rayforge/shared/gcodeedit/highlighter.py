from __future__ import annotations
import logging
from gi.repository import Gtk

logger = logging.getLogger(__name__)


class GcodeHighlighter:
    """
    Applies syntax highlighting to a Gtk.TextBuffer containing G-code.

    It uses simple, fast tokenization rather than a full regex parser,
    and connects to the buffer's "changed" signal to provide live updates
    as the user types. Colors are derived from the current GTK theme.
    """

    def __init__(self, text_view: Gtk.TextView):
        self.text_view = text_view
        self.buffer = self.text_view.get_buffer()
        self._changed_handler_id: int | None = None

        tag_table = self.buffer.get_tag_table()
        style_context = self.text_view.get_style_context()

        # Define and add tags for each token type using theme colors
        self._create_tag(
            tag_table,
            "comment",
            foreground=self._lookup_theme_color(
                style_context, "dim_label_color", "#888A85"
            ),
        )
        self._create_tag(
            tag_table,
            "gcode",
            foreground=self._lookup_theme_color(
                style_context, "accent_color", "#729FCF"
            ),
        )
        self._create_tag(
            tag_table,
            "mcode",
            foreground=self._lookup_theme_color(
                style_context, "warning_color", "#F57900"
            ),
        )
        self._create_tag(
            tag_table,
            "coord",
            foreground=self._lookup_theme_color(
                style_context, "success_color", "#8AE234"
            ),
        )
        # Use info_color which is typically blue/cyan in Adwaita
        self._create_tag(
            tag_table,
            "param",
            foreground=self._lookup_theme_color(
                style_context, "info_color", "#72D6D6"
            ),
        )

    def _lookup_theme_color(
        self, context: Gtk.StyleContext, name: str, fallback: str
    ) -> str:
        """Looks up a named color from the theme, returning a fallback."""
        found, color = context.lookup_color(name)
        if found and color:
            return color.to_string()
        return fallback

    def _create_tag(
        self, tag_table: Gtk.TextTagTable, name: str, **properties
    ):
        """Helper to create and add a Gtk.TextTag."""
        tag = Gtk.TextTag.new(name)
        for prop, value in properties.items():
            tag.set_property(prop, value)
        tag_table.add(tag)

    def start(self):
        """Connects to buffer signals to enable live highlighting."""
        if self._changed_handler_id is None:
            self._changed_handler_id = self.buffer.connect(
                "changed", self._on_buffer_changed
            )

    def stop(self):
        """Disconnects from buffer signals to disable live highlighting."""
        if self._changed_handler_id is not None:
            self.buffer.disconnect(self._changed_handler_id)
            self._changed_handler_id = None

    def highlight(
        self, start_iter: Gtk.TextIter, end_iter: Gtk.TextIter
    ) -> None:
        """
        Highlights the specified range in the buffer.

        Args:
            start_iter: The starting iterator of the range to highlight.
            end_iter: The ending iterator of the range to highlight.
        """
        if self._changed_handler_id is None:
            return

        # Prevent signals from firing during the update
        self.buffer.handler_block(self._changed_handler_id)

        # Remove all existing tags from the range
        self.buffer.remove_all_tags(start_iter, end_iter)

        current_iter = start_iter.copy()
        while current_iter.compare(end_iter) < 0:
            line_end_iter = current_iter.copy()
            if not line_end_iter.ends_line():
                line_end_iter.forward_to_line_end()

            line_text = self.buffer.get_text(current_iter, line_end_iter, True)
            self._highlight_line(current_iter, line_text)

            if not current_iter.forward_line():
                break

        # Re-enable signals
        self.buffer.handler_unblock(self._changed_handler_id)

    def _highlight_line(self, line_start_iter: Gtk.TextIter, line_text: str):
        """Applies tags to a single line of text."""
        # 1. Handle comments first, as they take precedence
        comment_char = ";"
        if "(" in line_text:
            comment_char = "("

        comment_start_idx = line_text.find(comment_char)

        if comment_start_idx != -1:
            # Get the part of the line before the comment
            code_text = line_text[:comment_start_idx]

            # Tag the comment
            comment_start_iter = line_start_iter.copy()
            comment_start_iter.forward_chars(comment_start_idx)
            comment_end_iter = comment_start_iter.copy()
            comment_end_iter.forward_chars(len(line_text) - comment_start_idx)
            self.buffer.apply_tag_by_name(
                "comment", comment_start_iter, comment_end_iter
            )
        else:
            code_text = line_text

        # 2. Tokenize and highlight the rest of the line
        offset = 0
        for word in code_text.split():
            word_start_idx = code_text.find(word, offset)
            if word_start_idx == -1:
                continue

            word_len = len(word)
            tag_name = None

            if not word:
                offset = word_start_idx + 1
                continue

            first_char = word[0].upper()
            if first_char == "G":
                tag_name = "gcode"
            elif first_char == "M":
                tag_name = "mcode"
            elif first_char in "XYZIJK":
                tag_name = "coord"
            elif first_char in "FSPTH":
                tag_name = "param"

            if tag_name:
                word_start_iter = line_start_iter.copy()
                word_start_iter.forward_chars(word_start_idx)
                word_end_iter = word_start_iter.copy()
                word_end_iter.forward_chars(word_len)
                self.buffer.apply_tag_by_name(
                    tag_name, word_start_iter, word_end_iter
                )

            offset = word_start_idx + word_len

    def _on_buffer_changed(self, buffer: Gtk.TextBuffer):
        """
        Called when the buffer content changes. To keep things simple and
        performant enough, we just re-highlight the entire buffer.
        """
        start_iter = buffer.get_start_iter()
        end_iter = buffer.get_end_iter()
        self.highlight(start_iter, end_iter)
