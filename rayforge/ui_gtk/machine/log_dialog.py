import shutil
from pathlib import Path
from typing import Optional
from gi.repository import Gtk, Adw, GLib, Gio
from blinker import Signal
from ...context import get_context
from ..icons import get_icon
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

        self._temp_archive_path: Optional[Path] = None

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

        self.save_log_button = Gtk.Button()
        button_content = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        button_content.append(get_icon("save-symbolic"))
        button_content.append(Gtk.Label(label=_("Save Debug Log")))
        self.save_log_button.set_child(button_content)
        self.save_log_button.add_css_class("suggested-action")
        self.save_log_button.set_margin_top(6)
        self.save_log_button.set_margin_bottom(6)
        self.save_log_button.set_halign(Gtk.Align.CENTER)
        self.save_log_button.connect("clicked", self._on_save_log_clicked)
        box.append(self.save_log_button)

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
        # The message is already formatted by the UILogHandler's formatter
        GLib.idle_add(self.append_to_terminal, message)

    def _on_save_log_clicked(self, button: Gtk.Button):
        self.save_log_button.set_sensitive(False)

        archive_path = get_context().debug_dump_manager.create_dump_archive()

        if not archive_path:
            self.notification_requested.send(
                self, message=_("Failed to create debug archive.")
            )
            self.save_log_button.set_sensitive(True)
            return

        self._temp_archive_path = archive_path

        parent_window = self.get_root()

        if not isinstance(parent_window, Gtk.Window):
            self.notification_requested.send(
                self,
                message=_("Could not find parent window to attach dialog."),
            )
            self.save_log_button.set_sensitive(True)
            return

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Save Debug Log"))
        dialog.set_initial_name(self._temp_archive_path.name)
        dialog.save(parent_window, None, self._on_save_dialog_response)

    def _on_save_dialog_response(self, dialog, result):
        try:
            destination_file = dialog.save_finish(result)
            if destination_file and self._temp_archive_path:
                destination_path = Path(destination_file.get_path())
                shutil.move(self._temp_archive_path, destination_path)
                self.notification_requested.send(
                    self,
                    message=_("Debug log saved to {path}").format(
                        path=destination_path.name
                    ),
                )
        except GLib.Error as e:
            if not e.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                self.notification_requested.send(
                    self,
                    message=_("Error saving file: {msg}").format(
                        msg=e.message
                    ),
                )
        except Exception as e:
            self.notification_requested.send(
                self,
                message=_("An unexpected error occurred: {error}").format(
                    error=e
                ),
            )
        finally:
            if self._temp_archive_path and self._temp_archive_path.exists():
                self._temp_archive_path.unlink()
            self._temp_archive_path = None
            self.save_log_button.set_sensitive(True)
