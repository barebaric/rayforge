from typing import Optional
from gi.repository import Gtk, GLib
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


class MachineControlPanel(Gtk.Box):
    notification_requested = Signal()

    def __init__(self, machine: Optional[Machine], **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self.terminal = Gtk.TextView()
        self.terminal.set_editable(False)
        self.terminal.set_cursor_visible(False)
        self.terminal.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(css)
        self.terminal.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_vexpand(True)
        self.scrolled_window.set_child(self.terminal)
        self.append(self.scrolled_window)

        self._populate_history()

        ui_log_event_received.connect(self.on_ui_log_received)

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
        formatted_lines = [
            ui_formatter.format(record) + "\n" for record in log_records
        ]
        text_buffer.set_text("".join(formatted_lines), -1)
        GLib.idle_add(self._scroll_to_bottom)

    def _is_at_bottom(self) -> bool:
        vadjustment = self.scrolled_window.get_vadjustment()
        max_value = vadjustment.get_upper() - vadjustment.get_page_size()
        return vadjustment.get_value() >= max_value - 1.0

    def append_to_terminal(self, data: str):
        should_autoscroll = self._is_at_bottom()
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
