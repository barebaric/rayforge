from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from gettext import gettext as _
from gi.repository import Gtk, Pango
from blinker import Signal
from ...shared.util.size import format_byte_size
from ...ui_gtk.shared.gtk import apply_css
from .editor import GcodeEditor

if TYPE_CHECKING:
    from ...pipeline.encoder.gcode import MachineCodeOpMap

css = """
.gcode-viewer {
    border-radius: 3px;
}
.gcode-status-label {
    background-color: alpha(@window_bg_color, 0.8);
    border-radius: 3px;
    padding: 2px 6px;
}
"""


class GcodeViewer(Gtk.Box):
    """
    A specialized, read-only widget for displaying G-code, intended for use
    as a preview panel.

    It uses a GcodeEditor internally but configures it for a better viewing
    experience (non-editable, word wrapping).
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        apply_css(css)
        self.add_css_class("gcode-viewer")

        self.set_margin_top(9)

        self.op_activated = Signal()
        self.line_activated = Signal()
        self.editor = GcodeEditor()
        self.op_map: Optional[MachineCodeOpMap] = None
        self._line_count: int = 0
        self._size_bytes: int = 0

        self.editor.text_view.set_editable(False)
        self.editor.text_view.set_cursor_visible(True)
        self.editor.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.editor.line_activated.connect(self._on_line_activated)

        self._create_overlay()

    def _create_overlay(self):
        self._overlay = Gtk.Overlay()
        self._overlay.set_child(self.editor)

        self.status_label = Gtk.Label()
        self.status_label.add_css_class("gcode-status-label")
        self.status_label.set_halign(Gtk.Align.END)
        self.status_label.set_valign(Gtk.Align.END)
        self.status_label.set_margin_bottom(3)
        self.status_label.set_margin_end(3)
        self.status_label.set_ellipsize(Pango.EllipsizeMode.END)
        self._overlay.add_overlay(self.status_label)

        self.append(self._overlay)
        self._update_status_bar()

    def _update_status_bar(self):
        if self._line_count == 0:
            self.status_label.set_text("")
            return

        size_str = format_byte_size(self._size_bytes)
        self.status_label.set_text(
            _("{line_count:,} lines  ·  {size}").format(
                line_count=self._line_count, size=size_str
            )
        )

    def _on_line_activated(self, sender, *, line_number: int):
        self.line_activated.send(self, line_number=line_number)
        if self.op_map and line_number in self.op_map.machine_code_to_op:
            op_index = self.op_map.machine_code_to_op[line_number]
            self.op_activated.send(self, op_index=op_index)

    def set_gcode(self, gcode: str):
        """
        Sets the G-code content to be displayed in the previewer.

        Args:
            gcode: The G-code to display, as a single string.
        """
        if gcode:
            self._line_count = gcode.count("\n") + 1
            self._size_bytes = len(gcode.encode("utf-8"))
        else:
            self._line_count = 0
            self._size_bytes = 0

        if self._line_count > 20000:
            lines = gcode.split("\n", 20000)
            truncated = "\n".join(lines[:20000])
            truncated += "\n\n" + _(
                "— Truncated (showing first 20,000 of {line_count:,} lines) —"
            ).format(line_count=self._line_count)
            self.editor.set_text(truncated)
        else:
            self.editor.set_text(gcode)
        self._update_status_bar()

    def clear(self):
        """Clears the content of the previewer."""
        self._line_count = 0
        self._size_bytes = 0
        self.editor.set_text("")
        self.op_map = None
        self.clear_highlight()
        self._update_status_bar()

    def set_op_map(self, op_map: MachineCodeOpMap):
        self.op_map = op_map

    def highlight_line(self, line_number: int, use_align: bool = True):
        """Highlights a specific line number in the editor."""
        self.editor.highlight_line(line_number, use_align)

    def highlight_op(self, op_index: int):
        if not self.op_map or op_index not in self.op_map.op_to_machine_code:
            self.clear_highlight()
            return

        line_numbers = self.op_map.op_to_machine_code[op_index]
        if line_numbers:
            # Highlight the first line associated with this op
            self.editor.highlight_line(line_numbers[0])
        else:
            # Op produced no g-code, so clear any existing highlight
            self.clear_highlight()

    def clear_highlight(self):
        self.editor.clear_highlight()
