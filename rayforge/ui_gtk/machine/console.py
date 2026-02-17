import logging
from typing import Optional
from gi.repository import Gtk, GLib
from blinker import Signal
from ...logging_setup import (
    get_memory_handler,
    get_ui_formatter,
    UILogFilter,
)
from ...machine.models.machine import Machine
from ...machine.driver.dummy import NoDeviceDriver
from ..icons import get_icon


logger = logging.getLogger(__name__)

css = """
.terminal {
    font-family: Monospace;
    font-size: 10pt;
}
.console-input {
    font-family: Monospace;
    font-size: 10pt;
    background-color: transparent;
    border: none;
    padding: 4px;
}
.console-input-scrolled {
    background-color: alpha(@window_fg_color, 0.05);
    border-radius: 4px;
}
"""


class Console(Gtk.Box):
    command_submitted = Signal()

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        self._show_verbose = False
        self._command_history: list[str] = []
        self._history_index = -1
        self._history_max = 1000
        self._machine: Optional[Machine] = None
        self._max_input_lines = 5
        self._single_line_height = 24
        self._auto_scroll = True

        self._setup_ui()
        self._setup_tags()
        self._populate_history()

    def _setup_ui(self):
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
        self.scrolled_window.set_hexpand(True)
        self.scrolled_window.set_child(self.terminal)

        scroll_controller = Gtk.EventControllerScroll(
            flags=Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll_controller.connect("scroll", self._on_user_scroll)
        self.scrolled_window.add_controller(scroll_controller)

        self.overlay = Gtk.Overlay()
        self.overlay.set_child(self.scrolled_window)

        self.verbose_toggle = Gtk.ToggleButton()
        self.verbose_toggle.set_active(False)
        self.verbose_toggle.set_tooltip_text(
            _("Show verbose output (status polls)")
        )
        self.verbose_toggle.connect("toggled", self._on_verbose_toggled)
        self.verbose_toggle.set_margin_top(6)
        self.verbose_toggle.set_margin_end(6)
        self.verbose_toggle.set_halign(Gtk.Align.END)
        self.verbose_toggle.set_valign(Gtk.Align.START)
        verbose_icon = get_icon("utilities-terminal-symbolic")
        self.verbose_toggle.set_child(verbose_icon)
        self.overlay.add_overlay(self.verbose_toggle)

        self.append(self.overlay)

        entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        entry_box.set_spacing(6)
        entry_box.set_margin_top(6)

        self.input_scrolled = Gtk.ScrolledWindow()
        self.input_scrolled.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self.input_scrolled.set_min_content_height(self._single_line_height)
        self.input_scrolled.set_max_content_height(
            self._single_line_height * self._max_input_lines
        )
        self.input_scrolled.set_propagate_natural_height(True)
        self.input_scrolled.add_css_class("console-input-scrolled")

        self.console_input = Gtk.TextView()
        self.console_input.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.console_input.add_css_class("console-input")
        self.console_input.set_hexpand(True)

        css_provider_input = Gtk.CssProvider()
        css_provider_input.load_from_string(css)
        self.console_input.get_style_context().add_provider(
            css_provider_input, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.input_scrolled.get_style_context().add_provider(
            css_provider_input, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.input_buffer = self.console_input.get_buffer()
        self.input_buffer.connect("changed", self._on_input_changed)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_input_key_pressed)
        self.console_input.add_controller(key_controller)

        self.input_scrolled.set_child(self.console_input)
        entry_box.append(self.input_scrolled)

        self.append(entry_box)

    def _setup_tags(self):
        tag_table = self.terminal.get_buffer().get_tag_table()
        style_context = self.terminal.get_style_context()

        self._create_tag(
            tag_table,
            "timestamp",
            foreground=self._lookup_theme_color(
                style_context, "dim_label_color", "#888A85"
            ),
        )
        self._create_tag(
            tag_table,
            "user_command",
            foreground=self._lookup_theme_color(
                style_context, "accent_color", "#729FCF"
            ),
            weight=700,
        )
        self._create_tag(
            tag_table,
            "error",
            foreground=self._lookup_theme_color(
                style_context, "error_color", "#EF2929"
            ),
        )
        self._create_tag(
            tag_table,
            "warning",
            foreground=self._lookup_theme_color(
                style_context, "warning_color", "#F57900"
            ),
        )
        self._create_tag(
            tag_table,
            "status_dim",
            foreground=self._lookup_theme_color(
                style_context, "dim_label_color", "#888A85"
            ),
        )
        self._create_tag(
            tag_table,
            "status_state",
            foreground=self._lookup_theme_color(
                style_context, "success_color", "#8AE234"
            ),
        )
        self._create_tag(
            tag_table,
            "status_coord",
            foreground=self._lookup_theme_color(
                style_context, "accent_color", "#729FCF"
            ),
        )

    def _lookup_theme_color(
        self, context: Gtk.StyleContext, name: str, fallback: str
    ) -> str:
        found, color = context.lookup_color(name)
        if found and color:
            return color.to_string()
        return fallback

    def _create_tag(
        self, tag_table: Gtk.TextTagTable, name: str, **properties
    ):
        tag = Gtk.TextTag.new(name)
        for prop, value in properties.items():
            tag.set_property(prop, value)
        tag_table.add(tag)

    def set_machine(self, machine: Optional[Machine]):
        self._machine = machine
        self._update_sensitivity()

    def _populate_history(self):
        memory_handler = get_memory_handler()
        ui_formatter = get_ui_formatter()
        if not memory_handler or not ui_formatter:
            return

        ui_filter = UILogFilter()
        machine_id = self._machine.id if self._machine else None
        log_records = [
            record
            for record in memory_handler.buffer
            if ui_filter.filter(record)
            and (
                self._show_verbose
                or record.__dict__.get("log_category")
                not in UILogFilter.VERBOSE_CATEGORIES
            )
            and (
                machine_id is None
                or record.__dict__.get("machine_id") is None
                or record.__dict__.get("machine_id") == machine_id
            )
        ]

        text_buffer = self.terminal.get_buffer()
        text_buffer.set_text("", -1)
        for record in log_records:
            formatted = ui_formatter.format(record)
            category = record.__dict__.get("log_category")
            self._append_highlighted(formatted, category)
        GLib.idle_add(self._scroll_to_bottom_and_track)

    def _is_at_bottom(self) -> bool:
        vadjustment = self.scrolled_window.get_vadjustment()
        page_size = vadjustment.get_page_size()
        if page_size == 0:
            return True
        max_value = vadjustment.get_upper() - page_size
        threshold = min(page_size * 0.1, 50.0)
        return vadjustment.get_value() >= max_value - threshold

    def _on_user_scroll(self, controller, dx, dy):
        if dy < 0:
            self._auto_scroll = False
        elif self._is_at_bottom():
            self._auto_scroll = True
        return False

    def _highlight_status_poll(
        self, text_buffer, message: str, base_offset: int
    ):
        if not (message.startswith("<") and message.endswith(">")):
            text_buffer.insert(text_buffer.get_end_iter(), f"{message}\n", -1)
            return

        content = message[1:-1]
        parts = content.split("|")
        full_text = "<" + "|".join(parts) + ">\n"

        text_buffer.insert(text_buffer.get_end_iter(), full_text, -1)

        offset = base_offset
        text_buffer.apply_tag_by_name(
            "status_dim",
            text_buffer.get_iter_at_offset(offset),
            text_buffer.get_iter_at_offset(offset + 1),
        )
        offset += 1

        for i, part in enumerate(parts):
            if ":" in part:
                key, value = part.split(":", 1)
                text_buffer.apply_tag_by_name(
                    "status_coord",
                    text_buffer.get_iter_at_offset(offset),
                    text_buffer.get_iter_at_offset(offset + len(key)),
                )
                offset += len(key) + 1 + len(value)
            else:
                text_buffer.apply_tag_by_name(
                    "status_state",
                    text_buffer.get_iter_at_offset(offset),
                    text_buffer.get_iter_at_offset(offset + len(part)),
                )
                offset += len(part)

            if i < len(parts) - 1:
                text_buffer.apply_tag_by_name(
                    "status_dim",
                    text_buffer.get_iter_at_offset(offset),
                    text_buffer.get_iter_at_offset(offset + 1),
                )
                offset += 1

        text_buffer.apply_tag_by_name(
            "status_dim",
            text_buffer.get_iter_at_offset(offset),
            text_buffer.get_iter_at_offset(offset + 1),
        )

    def _append_highlighted(self, message: str, category: Optional[str]):
        text_buffer = self.terminal.get_buffer()
        end_iter = text_buffer.get_end_iter()

        if category == "USER_COMMAND":
            start_offset = end_iter.get_offset()
            text_buffer.insert(end_iter, f"{message}\n", -1)
            start_iter = text_buffer.get_iter_at_offset(start_offset)
            end_iter = text_buffer.get_end_iter()
            text_buffer.apply_tag_by_name("user_command", start_iter, end_iter)
        elif category == "STATUS_POLL":
            if message and len(message) > 19 and message[19] == " ":
                timestamp = message[:19]
                rest = message[20:]
                start_offset = end_iter.get_offset()
                text_buffer.insert(end_iter, f"{timestamp} ", -1)
                tag_start = text_buffer.get_iter_at_offset(start_offset)
                tag_end = text_buffer.get_iter_at_offset(start_offset + 20)
                text_buffer.apply_tag_by_name("timestamp", tag_start, tag_end)
                self._highlight_status_poll(
                    text_buffer, rest, start_offset + 20
                )
            else:
                start_offset = end_iter.get_offset()
                self._highlight_status_poll(text_buffer, message, start_offset)
        elif category == "ERROR":
            if message.startswith("ERROR "):
                start_offset = end_iter.get_offset()
                text_buffer.insert(end_iter, "ERROR ", -1)
                tag_start = text_buffer.get_iter_at_offset(start_offset)
                tag_end = text_buffer.get_iter_at_offset(start_offset + 5)
                text_buffer.apply_tag_by_name("error", tag_start, tag_end)
                end_iter = text_buffer.get_end_iter()
                text_buffer.insert(end_iter, f"{message[6:]}\n", -1)
            else:
                text_buffer.insert(end_iter, f"{message}\n", -1)
        elif category == "WARNING":
            if message.startswith("WARN "):
                start_offset = end_iter.get_offset()
                text_buffer.insert(end_iter, "WARN ", -1)
                tag_start = text_buffer.get_iter_at_offset(start_offset)
                tag_end = text_buffer.get_iter_at_offset(start_offset + 4)
                text_buffer.apply_tag_by_name("warning", tag_start, tag_end)
                end_iter = text_buffer.get_end_iter()
                text_buffer.insert(end_iter, f"{message[5:]}\n", -1)
            else:
                text_buffer.insert(end_iter, f"{message}\n", -1)
        elif message and len(message) > 19 and message[19] == " ":
            timestamp = message[:19]
            rest = message[20:]
            start_offset = end_iter.get_offset()
            text_buffer.insert(end_iter, f"{timestamp} ", -1)
            tag_start = text_buffer.get_iter_at_offset(start_offset)
            tag_end = text_buffer.get_iter_at_offset(start_offset + 20)
            text_buffer.apply_tag_by_name("timestamp", tag_start, tag_end)
            end_iter = text_buffer.get_end_iter()
            text_buffer.insert(end_iter, f"{rest}\n", -1)
        else:
            text_buffer.insert(end_iter, f"{message}\n", -1)

    def append_to_terminal(self, message: str, category: Optional[str] = None):
        self._append_highlighted(message, category)
        if self._auto_scroll:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        text_buffer = self.terminal.get_buffer()
        end_iter = text_buffer.get_end_iter()
        mark = text_buffer.create_mark("end_mark", end_iter, False)
        self.terminal.scroll_to_mark(mark, 0.0, False, 0.0, 0.0)
        text_buffer.delete_mark(mark)

    def _scroll_to_bottom_and_track(self):
        self._auto_scroll = True
        self._scroll_to_bottom()

    def on_log_received(
        self,
        sender,
        message: Optional[str] = None,
        category: Optional[str] = None,
        machine_id: Optional[str] = None,
    ):
        if not message:
            return
        if self._machine and machine_id and machine_id != self._machine.id:
            return
        if (
            category in UILogFilter.VERBOSE_CATEGORIES
            and not self._show_verbose
        ):
            return
        GLib.idle_add(self.append_to_terminal, message, category)

    def _on_verbose_toggled(self, button):
        self._show_verbose = button.get_active()
        self._populate_history()

    def _get_input_text(self) -> str:
        start = self.input_buffer.get_start_iter()
        end = self.input_buffer.get_end_iter()
        return self.input_buffer.get_text(start, end, False)

    def _set_input_text(self, text: str):
        self.input_buffer.set_text(text, -1)

    def _on_input_changed(self, buffer):
        line_count = buffer.get_line_count()
        height = min(
            line_count * self._single_line_height,
            self._single_line_height * self._max_input_lines,
        )
        self.input_scrolled.set_min_content_height(
            max(height, self._single_line_height)
        )

    def _on_input_key_pressed(
        self, controller, keyval, keycode, state
    ) -> bool:
        from gi.repository import Gdk

        if keyval == Gdk.KEY_Return or keyval == Gdk.KEY_KP_Enter:
            if state & Gdk.ModifierType.SHIFT_MASK:
                return False
            self._send_commands()
            return True
        elif keyval == Gdk.KEY_Up:
            self._navigate_history(-1)
            return True
        elif keyval == Gdk.KEY_Down:
            self._navigate_history(1)
            return True
        return False

    def _send_commands(self):
        machine = self._machine
        if not machine:
            return

        text = self._get_input_text().strip()
        if not text:
            return

        is_dummy = isinstance(machine.driver, NoDeviceDriver)
        is_connected = machine.is_connected()

        if not is_connected and not is_dummy:
            logger.error(
                "Machine not connected",
                extra={"log_category": "ERROR", "machine_id": machine.id},
            )
            return

        commands = [line.strip() for line in text.split("\n") if line.strip()]
        self._set_input_text("")

        for command in commands:
            self._add_to_history(command)
            self.command_submitted.send(self, command=command, machine=machine)

    def _add_to_history(self, command: str):
        if self._command_history and self._command_history[-1] == command:
            return
        self._command_history.append(command)
        if len(self._command_history) > self._history_max:
            self._command_history.pop(0)
        self._history_index = len(self._command_history)

    def _navigate_history(self, direction: int):
        if not self._command_history:
            return

        new_index = self._history_index + direction

        if new_index < 0:
            new_index = 0
        elif new_index >= len(self._command_history):
            new_index = len(self._command_history)
            self._set_input_text("")
            self._history_index = new_index
            return

        self._history_index = new_index
        command = self._command_history[new_index]
        self._set_input_text(command)

    def _update_sensitivity(self):
        if not self._machine:
            sensitive = False
        else:
            is_dummy = isinstance(self._machine.driver, NoDeviceDriver)
            is_connected = self._machine.is_connected()
            sensitive = is_connected or is_dummy

        self.console_input.set_sensitive(sensitive)

    def on_machine_state_changed(self, machine, state):
        self._update_sensitivity()
