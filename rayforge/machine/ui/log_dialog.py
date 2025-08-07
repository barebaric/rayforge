import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from gi.repository import Gtk, Adw, GLib, Gio  # type: ignore
from blinker import Signal
from ..driver.driver import TransportStatus
from ..models.machine import Machine
from ...debug import debug_log_manager, LogEntry, LogType


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

        self.save_log_button = Gtk.Button.new_with_label(_("Save Debug Log"))
        self.save_log_button.set_icon_name("document-save-symbolic")
        self.save_log_button.add_css_class("suggested-action")
        self.save_log_button.set_margin_top(6)
        self.save_log_button.set_margin_bottom(6)
        self.save_log_button.set_halign(Gtk.Align.CENTER)
        self.save_log_button.connect("clicked", self._on_save_log_clicked)
        box.append(self.save_log_button)

        self._populate_history()

        if machine:
            driver = machine.driver
            driver.log_received.connect(self.on_log_received)
            driver.command_status_changed.connect(
                self.on_command_status_changed
            )
            driver.connection_status_changed.connect(
                self.on_connection_status_changed
            )

        parent_width = parent.get_allocated_width()
        self.set_size_request(max(100, parent_width - 24), -1)
        self.set_follows_content_size(True)

    def _populate_history(self):
        log_snapshot = debug_log_manager._get_log_snapshot()
        text_buffer = self.terminal.get_buffer()
        formatted_lines = [
            self._format_log_entry_for_terminal(entry)
            for entry in log_snapshot
        ]
        text_buffer.set_text("".join(formatted_lines))
        # Always scroll to the bottom on initial population
        GLib.idle_add(self._scroll_to_bottom)

    def _format_log_entry_for_terminal(self, entry: LogEntry) -> str:
        local_timestamp = entry.timestamp.astimezone().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        data_str = ""
        if isinstance(entry.data, bytes):
            try:
                data_str = entry.data.decode("utf-8").strip()
            except UnicodeDecodeError:
                data_str = f"[Binary data: {len(entry.data)} bytes]"
        elif isinstance(entry.data, str):
            data_str = entry.data.strip()
        else:
            data_str = str(entry.data)

        if entry.log_type in [LogType.TX, LogType.RX]:
            return ""

        return (
            f"[{local_timestamp}] {entry.source} "
            f" ({entry.log_type.name}): {data_str}\n"
        )

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

        timestamp = datetime.now().strftime("%x %X")
        formatted_message = f"[{timestamp}] {data}\n"
        text_buffer = self.terminal.get_buffer()
        text_buffer.insert(text_buffer.get_end_iter(), formatted_message)

        if should_autoscroll:
            GLib.idle_add(self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        text_buffer = self.terminal.get_buffer()
        end_iter = text_buffer.get_end_iter()
        mark = text_buffer.create_mark("end_mark", end_iter, False)
        self.terminal.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        text_buffer.delete_mark(mark)
        return False

    def on_log_received(self, sender, message: Optional[str] = None):
        if not message:
            return
        driver_name = sender.__class__.__name__
        self.append_to_terminal(f"{driver_name}: {message}")

    def on_command_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        msg = _("Command status changed to {status}").format(
            status=status.name
        )
        if message:
            msg += f" with message: {message}"
        self.append_to_terminal(msg)

    def on_connection_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        msg = _("Connection status changed to {status}").format(
            status=status.name
        )
        if message:
            msg += f" with message: {message}"
        self.append_to_terminal(msg)

    def _on_save_log_clicked(self, button: Gtk.Button):
        self.save_log_button.set_sensitive(False)

        archive_path = debug_log_manager.create_dump_archive()

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
