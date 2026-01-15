from typing import Optional
from gi.repository import Gtk, Adw, GLib
from blinker import Signal
from ...logging_setup import (
    ui_log_event_received,
    get_memory_handler,
    get_ui_formatter,
    UILogFilter,
)
from ...machine.models.machine import Machine


css = """
.terminal {
    font-family: Monospace;
    font-size: 10pt;
}
"""


class MachineLogDialog(Adw.Dialog):  # TODO: with Adw 1.6, use BottomSheet
    notification_requested = Signal()

    def __init__(self, parent, machine: Optional[Machine], **kwargs):
        super().__init__(**kwargs)
        self.set_presentation_mode(Adw.DialogPresentationMode.BOTTOM_SHEET)
        self.set_title(_("Machine Log"))

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_child(box)

        self.terminal = Gtk.TextView()
        self.terminal.set_editable(False)
        self.terminal.set_cursor_visible(False)
        self.terminal.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.terminal.set_margin_top(12)
        self.terminal.set_margin_bottom(12)
        self.terminal.set_margin_start(12)
        self.terminal.set_margin_end(12)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(css)
        self.terminal.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_min_content_height(600)
        self.scrolled_window.set_child(self.terminal)
        box.append(self.scrolled_window)

        self._populate_history()

        # Connect to the new global UI log signal
        ui_log_event_received.connect(self.on_ui_log_received)
        self.connect("closed", self.on_closed)

        parent_width = parent.get_allocated_width()
        self.set_size_request(max(100, parent_width - 80), -1)
        self.set_follows_content_size(True)

    def on_closed(self, *args):
        # Disconnect from the signal when the dialog is closed to prevent leaks
        ui_log_event_received.disconnect(self.on_ui_log_received)

    def _populate_history(self):
        memory_handler = get_memory_handler()
        ui_formatter = get_ui_formatter()
        if not memory_handler or not ui_formatter:
            return

        ui_filter = UILogFilter()
        log_records = [
            record
            for record in memory_handler.buffer
            if ui_filter.filter(record)
        ]

        text_buffer = self.terminal.get_buffer()
        # Use the one true formatter to create the lines
        formatted_lines = [
            ui_formatter.format(record) + "\n" for record in log_records
        ]
        text_buffer.set_text("".join(formatted_lines), -1)
        # Always scroll to the bottom on initial population
        GLib.idle_add(self._scroll_to_bottom)

    def _is_at_bottom(self) -> bool:
        """Check if the scrolled window is at the bottom."""
        vadjustment = self.scrolled_window.get_vadjustment()
        # The maximum value for the adjustment is upper - page_size
        max_value = vadjustment.get_upper() - vadjustment.get_page_size()
        # Use a small tolerance to account for floating point inaccuracies
        return vadjustment.get_value() >= max_value - 1.0

    def append_to_terminal(self, data: str):
        # Check if we should scroll after appending text.
        # This is true if the user is already at the bottom.
        should_autoscroll = self._is_at_bottom()

        # The 'data' is already a fully pre-formatted string from the handler.
        # We just need to add the newline.
        formatted_message = f"{data}\n"
        text_buffer = self.terminal.get_buffer()
        text_buffer.insert(text_buffer.get_end_iter(), formatted_message, -1)

        if should_autoscroll:
            GLib.idle_add(self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        text_buffer = self.terminal.get_buffer()
        end_iter = text_buffer.get_end_iter()
        mark = text_buffer.create_mark("end_mark", end_iter, False)
        self.terminal.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        text_buffer.delete_mark(mark)
        return False

    def on_ui_log_received(self, sender, message: Optional[str] = None):
        if not message:
            return
        GLib.idle_add(self.append_to_terminal, message)
