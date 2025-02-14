import asyncio
from datetime import datetime
from locale import getdefaultlocale
from gi.repository import Gtk, Adw, GLib, Pango
from ..driver.driver import driver_mgr, Status


css = """
.terminal {
    font-family: Monospace;
    font-size: 10pt;
}
"""


class MachineView(Adw.Dialog):
    def __init__(self):
        super().__init__()
        self.set_presentation_mode(Adw.DialogPresentationMode.BOTTOM_SHEET)

        # Main container
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.set_child(box)

        # WebSocket terminal-like display
        self.terminal = Gtk.TextView()
        self.terminal.set_editable(False)  # Make it read-only
        self.terminal.set_cursor_visible(False)  # Hide the cursor
        self.terminal.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)  # Wrap text
        self.terminal.set_margin_top(12)
        self.terminal.set_margin_bottom(12)
        self.terminal.set_margin_start(12)
        self.terminal.set_margin_end(12)

        # Apply a monospace font using CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(css)
        self.terminal.get_style_context().add_provider(
            css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Wrap the TextView in a ScrolledWindow
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_min_content_height(400)
        scrolled_window.set_child(self.terminal)
        box.append(scrolled_window)

        # Listen to driver
        driver = driver_mgr.driver
        driver.log_received_safe.connect(self.on_log_received)
        driver.command_status_changed_safe.connect(
            self.on_command_status_changed
        )
        driver.connection_status_changed_safe.connect(
            self.on_connection_status_changed
        )

        # The dialog does not support expansion. Adw 1.6 will support
        # BottomSheet, at which time this widget should probably use that
        # instead. But for now, it is not available in Ubuntu 24.04,
        # so we need to define a fixed size.
        self.set_size_request(900, -1)  # Allow the dialog to expand
        self.set_follows_content_size(True)

    def append_to_terminal(self, data):
        # Get the current timestamp in the user's locale
        timestamp = datetime.now().strftime("%x %X")  # Locale-specific date and time
        formatted_message = f"[{timestamp}] {data}\n"

        # Get the TextBuffer and insert the new message
        text_buffer = self.terminal.get_buffer()
        text_buffer.insert(text_buffer.get_end_iter(), formatted_message)

        # Scroll to the end of the buffer
        self.terminal.scroll_to_iter(text_buffer.get_end_iter(), 0, False, 0, 0)

    def on_log_received(self, sender, message=None):
        """
        Update terminal display.
        """
        driver_name = sender.__class__.__name__
        self.append_to_terminal(f"{driver_name}: {message}")

    def on_command_status_changed(self, sender, status: Status, message: str|None=None):
        self.append_to_terminal(
            f"Command status changed to {status} with message: {message}"
        )

    def on_connection_status_changed(self, sender, status: Status, message: str|None=None):
        self.append_to_terminal(
            f"Status changed to {status} with message: {message}"
        )
