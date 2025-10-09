from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from gi.repository import Gtk
from blinker import Signal
from .editor import GcodeEditor

if TYPE_CHECKING:
    from ...pipeline.encoder.gcode import GcodeOpMap


class GcodePreviewer(Gtk.Box):
    """
    A specialized, read-only widget for displaying G-code, intended for use
    as a preview panel.

    It uses a GcodeEditor internally but configures it for a better viewing
    experience (non-editable, word wrapping).
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.op_activated = Signal()
        self.editor = GcodeEditor()
        self.op_map: Optional[GcodeOpMap] = None

        # Configure the internal editor for previewing
        self.editor.text_view.set_editable(False)
        # A visible cursor is necessary to show focus, even in read-only mode
        self.editor.text_view.set_cursor_visible(True)
        self.editor.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.editor.line_activated.connect(self._on_line_activated)

        self.append(self.editor)

    def _on_line_activated(self, sender, *, line_number: int):
        if self.op_map and line_number in self.op_map.gcode_to_op:
            op_index = self.op_map.gcode_to_op[line_number]
            self.op_activated.send(self, op_index=op_index)

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
        self.op_map = None
        self.clear_highlight()

    def set_op_map(self, op_map: GcodeOpMap):
        self.op_map = op_map

    def highlight_line(self, line_number: int):
        """Highlights a specific line number in the editor."""
        self.editor.highlight_line(line_number)

    def highlight_op(self, op_index: int):
        if not self.op_map or op_index not in self.op_map.op_to_gcode:
            self.clear_highlight()
            return

        line_numbers = self.op_map.op_to_gcode[op_index]
        if line_numbers:
            # Highlight the first line associated with this op
            self.editor.highlight_line(line_numbers[0])
        else:
            # Op produced no g-code, so clear any existing highlight
            self.clear_highlight()

    def clear_highlight(self):
        self.editor.clear_highlight()
