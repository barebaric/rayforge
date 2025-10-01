from __future__ import annotations
from typing import TYPE_CHECKING
from gi.repository import Gtk
from .editor import GcodeEditor

if TYPE_CHECKING:
    pass


class GcodePreviewer(Gtk.Box):
    """
    A specialized, read-only widget for displaying G-code, intended for use
    as a preview panel.

    It uses a GcodeEditor internally but configures it for a better viewing
    experience (non-editable, word wrapping).
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.editor = GcodeEditor()

        # Configure the internal editor for previewing
        self.editor.text_view.set_editable(False)
        # A visible cursor is necessary to show focus, even in read-only mode
        self.editor.text_view.set_cursor_visible(True)
        self.editor.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)

        self.append(self.editor)

    def set_gcode(self, gcode: str):
        """
        Sets the G-code content to be displayed in the previewer.

        Args:
            gcode: The G-code to display, as a single string.
        """
        self.editor.set_text(gcode)

    def clear(self):
        """Clears the content of the previewer."""
        self.editor.set_text("")
