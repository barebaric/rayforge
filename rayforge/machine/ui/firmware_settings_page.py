import logging
from typing import List
from gi.repository import Gtk, Adw, GLib, Gdk  # type: ignore
from blinker import Signal
from ...config import config
from ..driver import get_driver_cls
from ...shared.varset.varsetwidget import VarSetWidget, VarSet

logger = logging.getLogger(__name__)


class FirmwareSettingsPage(Adw.PreferencesPage):
    """
    A preferences page for reading and writing firmware settings.
    """

    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Firmware"),
            icon_name="drive-harddisk-symbolic",
            **kwargs,
        )
        logger.debug("__init__")
        self.machine = machine
        self._current_error = None
        self._is_updating_from_model = False
        self._varset_widgets = []
        self._error_timeout_id = 0
        self._warning_rows = []
        self._dismissed_warnings = set()
        self._is_busy = False

        self.show_toast = Signal()

        # Create a single main group for all static content
        self.main_group = Adw.PreferencesGroup()
        self.add(self.main_group)
        self._main_group_title = _("Device Settings")
        self._main_group_desc = _(
            "Read or apply settings directly to the device firmware."
        )

        # Create header controls once and store them
        self.spinner = Gtk.Spinner()
        self.read_button = Gtk.Button.new_from_icon_name(
            "view-refresh-symbolic"
        )
        self.read_button.set_tooltip_text(_("Read from Device"))
        self.read_button.connect("clicked", self._on_read_clicked)
        self.header_box = Gtk.Box(spacing=6)
        self.header_box.append(self.spinner)
        self.header_box.append(self.read_button)
        self.main_group.set_header_suffix(self.header_box)

        # Banners
        self.unsupported_banner = Adw.Banner(
            title=_(
                "The current driver does not support reading device settings."
            )
        )
        self.main_group.add(self.unsupported_banner)

        self.not_connected_banner = Adw.Banner(
            title=_("This machine is not active."),
            button_label=_("Activate Machine"),
        )
        self.not_connected_banner.connect(
            "button-clicked", self._on_activate_clicked
        )
        self.main_group.add(self.not_connected_banner)

        # Error row with copy and close buttons
        self.error_row = Adw.ActionRow(use_markup=True, activatable=False)
        self.error_row.set_icon_name("dialog-error-symbolic")
        self.error_row.add_css_class("error")

        copy_button = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        copy_button.set_tooltip_text(_("Copy Error Details"))
        copy_button.add_css_class("flat")
        copy_button.set_valign(Gtk.Align.CENTER)
        copy_button.connect("clicked", self._on_copy_error_clicked)
        self.error_row.add_suffix(copy_button)

        error_close_button = Gtk.Button.new_from_icon_name(
            "window-close-symbolic"
        )
        error_close_button.set_tooltip_text(_("Dismiss Error"))
        error_close_button.add_css_class("flat")
        error_close_button.set_valign(Gtk.Align.CENTER)
        error_close_button.connect("clicked", self._on_error_dismissed)
        self.error_row.add_suffix(error_close_button)
        self.main_group.add(self.error_row)

        # Dismissible warning rows
        items = [
            _(
                "Editing these values can be dangerous and may render your"
                " machine inoperable!"
            ),
            _(
                "The device may restart or temporarily disconnect after a"
                " setting is changed."
            ),
        ]
        for item in items:
            warning_row = Adw.ActionRow(title=item, activatable=False)
            warning_row.set_icon_name("dialog-warning-symbolic")
            warning_row.add_css_class("warning")

            close_button = Gtk.Button.new_from_icon_name(
                "window-close-symbolic"
            )
            close_button.set_tooltip_text(_("Dismiss Warning"))
            close_button.add_css_class("flat")
            close_button.set_valign(Gtk.Align.CENTER)
            close_button.connect(
                "clicked", self._on_warning_dismissed, warning_row
            )
            warning_row.add_suffix(close_button)
            self.main_group.add(warning_row)
            self._warning_rows.append(warning_row)

        # A group and row to prompt the user to load settings
        self.prompt_group = Adw.PreferencesGroup()
        prompt_row = Adw.ActionRow(
            title=_(
                "Click the refresh button to load settings from the device."
            ),
            icon_name="info-symbolic",
            activatable=False,
        )
        self.prompt_group.add(prompt_row)
        self.add(self.prompt_group)

        # Signal Connections & Initial State
        self.machine.changed.connect(self._on_machine_config_changed)
        config.changed.connect(self._on_machine_config_changed)
        self.machine.settings_updated.connect(
            self._on_settings_op_success
        )
        self.machine.setting_applied.connect(self._on_setting_applied)
        self.machine.settings_error.connect(self._on_settings_op_error)
        self.connect("destroy", self.on_destroy)

        self._update_ui_state()
        logger.debug("__init__ finished.")

    def on_destroy(self, _widget):
        logger.debug("on_destroy: Disconnecting signals.")
        self.machine.changed.disconnect(self._on_machine_config_changed)
        config.changed.disconnect(self._on_machine_config_changed)
        self.machine.settings_updated.disconnect(
            self._on_settings_op_success
        )
        self.machine.setting_applied.disconnect(self._on_setting_applied)
        self.machine.settings_error.disconnect(self._on_settings_op_error)
        if self._error_timeout_id > 0:
            GLib.source_remove(self._error_timeout_id)

    def _on_machine_config_changed(self, sender, **kwargs):
        logger.debug("_on_machine_config_changed: Rebuilding UI.")
        for widget in self._varset_widgets:
            self.remove(widget)
        self._varset_widgets.clear()
        self._update_ui_state()

    def _rebuild_settings_widgets(self, var_sets: List[VarSet]):
        for widget in self._varset_widgets:
            self.remove(widget)
        self._varset_widgets.clear()

        if not var_sets:
            return

        for var_set in var_sets:
            widget = VarSetWidget(explicit_apply=True)
            widget.set_title(GLib.markup_escape_text(var_set.title or ""))
            if var_set.description:
                widget.set_description(
                    GLib.markup_escape_text(var_set.description)
                )
            widget.populate(var_set)
            widget.data_changed.connect(self._on_setting_apply)
            self.add(widget)
            self._varset_widgets.append(widget)
        logger.debug(f"Created {len(self._varset_widgets)} widgets.")

    def _update_ui_state(self):
        logger.debug(f"_update_ui_state: Starting (is_busy={self._is_busy}).")
        driver_cls = (
            get_driver_cls(self.machine.driver)
            if self.machine.driver
            else None
        )
        is_supported = driver_cls is not None and driver_cls.supports_settings

        active_machine = config.machine
        is_active = active_machine and self.machine.id == active_machine.id

        can_interact = is_supported and is_active
        has_settings_to_show = can_interact and len(self._varset_widgets) > 0

        # Control banners
        self.unsupported_banner.set_revealed(not is_supported)
        self.not_connected_banner.set_revealed(is_supported and not is_active)

        # Control the state of the single main group
        self.main_group.set_title(
            self._main_group_title if can_interact else ""
        )
        self.main_group.set_description(
            self._main_group_desc if can_interact else ""
        )
        self.header_box.set_visible(can_interact)

        for row in self._warning_rows:
            row.set_visible(
                can_interact and row not in self._dismissed_warnings
            )

        has_error = self._current_error is not None
        self.error_row.set_visible(has_error)
        if has_error:
            self.error_row.set_title(_("Operation failed"))
            self.error_row.set_subtitle(self._current_error or "")

        # The main group is visible if any of its contents are.
        is_any_warning_visible = any(
            row.get_visible() for row in self._warning_rows
        )
        self.main_group.set_visible(
            self.unsupported_banner.get_revealed()
            or self.not_connected_banner.get_revealed()
            or self.error_row.get_visible()
            or is_any_warning_visible
            or has_settings_to_show
        )

        # Control visibility of the dynamic settings widgets
        for widget in self._varset_widgets:
            widget.set_visible(has_settings_to_show)

        # Control visibility of prompt group
        show_prompt = (
            can_interact
            and not has_settings_to_show
            and not self._is_busy
            and not has_error
        )
        self.prompt_group.set_visible(show_prompt)

        if self._is_busy:
            self.spinner.start()
            self.read_button.set_sensitive(False)
        else:
            self.spinner.stop()
            self.read_button.set_sensitive(is_active)
        logger.debug("_update_ui_state: Finished.")

    def _on_settings_op_success(self, sender, var_sets: List[VarSet]):
        logger.debug("Success signal received with var_sets (from read).")

        scrolled_window = self.get_ancestor(Gtk.ScrolledWindow)
        adj = scrolled_window.get_vadjustment() if scrolled_window else None
        scroll_value = (
            adj.get_value() if adj and self._varset_widgets else None
        )

        self._is_busy = False
        self._clear_error_state()

        self._rebuild_settings_widgets(var_sets)
        self._update_ui_state()

        if adj and scroll_value is not None:

            def restore_scroll():
                max_scroll = adj.get_upper() - adj.get_page_size()
                if scroll_value <= max_scroll:
                    adj.set_value(scroll_value)
                return GLib.SOURCE_REMOVE

            GLib.idle_add(restore_scroll)

    def _on_setting_applied(self, sender):
        """Handles the successful application of a single setting."""
        logger.debug("Success signal received for applying setting.")
        self._is_busy = False
        self._clear_error_state()
        self.show_toast.send(self, message=_("Setting applied successfully."))
        self._update_ui_state()

    def _on_settings_op_error(self, sender, error):
        logger.debug(f"Error signal received: {error}")
        self._is_busy = False
        # Clear any stale settings from the UI by passing an empty list
        self._rebuild_settings_widgets([])
        self._show_error(str(error))
        self._update_ui_state()

    def _on_setting_apply(self, sender: VarSetWidget, key: str):
        if self._is_busy:
            return
        if self._is_updating_from_model:
            return

        new_value = sender.get_values().get(key)
        if new_value is None:
            return

        logger.debug(f"_on_setting_apply: Applying key '{key}'.")
        self._is_busy = True
        self._clear_error_state()
        self._update_ui_state()
        self.machine.apply_setting(key, new_value)

    def _on_read_clicked(self, _button=None):
        if self._is_busy:
            return

        logger.debug("_on_read_clicked: Read button clicked.")
        self._is_busy = True
        self._clear_error_state()
        self._update_ui_state()
        self.machine.refresh_settings()

    def _on_warning_dismissed(self, _button, row_to_dismiss):
        self._dismissed_warnings.add(row_to_dismiss)
        row_to_dismiss.set_visible(False)
        self._update_ui_state()

    def _on_error_dismissed(self, _button):
        """Hides the error row and cancels the auto-hide timer."""
        self._clear_error_state()
        self._update_ui_state()

    def _on_copy_error_clicked(self, _button):
        """Copies the current error message to the clipboard."""
        if self._current_error:
            clipboard = self.get_clipboard()
            provider = Gdk.ContentProvider.new_for_value(self._current_error)
            clipboard.set_content(provider)

    def _on_activate_clicked(self, _banner):
        """Handler for the 'Activate Machine' button."""
        logger.debug(f"Activating machine: {self.machine.name}")
        config.set_machine(self.machine)
        self.show_toast.send(self, message=_("Machine activated."))

    def _on_error_timeout(self):
        self._clear_error_state()
        self._update_ui_state()
        return GLib.SOURCE_REMOVE

    def _clear_error_state(self):
        if self._error_timeout_id > 0:
            GLib.source_remove(self._error_timeout_id)
            self._error_timeout_id = 0
        self._current_error = None

    def _show_error(self, error_message: str):
        # Do not show a red error row for the "not active" state, as it's
        # handled by the `not_connected_banner`.
        if _("Machine is not active") in error_message:
            logger.debug("Ignoring 'not active' error for error row display.")
            return

        logger.debug(f"_show_error: Displaying '{error_message}'.")
        self._clear_error_state()
        self._current_error = error_message
        self._update_ui_state()
        self._error_timeout_id = GLib.timeout_add_seconds(
            8, self._on_error_timeout
        )
