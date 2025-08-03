import logging
from gi.repository import Gtk, Adw, GLib  # type: ignore
from .firmwaresettingswidget import FirmwareSettingsWidget
from ..driver.driver import driver_mgr
from ..config import task_mgr
from ..tasker.task import Task, CancelledError

logger = logging.getLogger(__name__)


class FirmwareSettingsPage(Adw.PreferencesPage):
    """
    A preferences page dedicated to reading and writing low-level firmware
    settings, controlling the entire layout and interaction logic.
    """

    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Firmware"),
            icon_name="drive-harddisk-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._current_error = None

        # The PreferencesGroup is the main container for this page.
        self.settings_group = Adw.PreferencesGroup(
            title=_("Firmware Settings")
        )
        self.settings_group.set_description(
            _("Read and write settings from the device firmware.")
        )
        self.add(self.settings_group)

        self.unsupported_banner = Adw.Banner(
            title=_(
                "The current driver does not support reading device settings."
            )
        )
        self.error_banner = Adw.Banner(revealed=False, use_markup=True)
        self.error_banner.add_css_class("error")

        # A master Gtk.Box manages the main content's layout.
        self.main_layout_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )

        self.action_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            margin_top=12,
            margin_bottom=12,
        )
        self.spinner = Gtk.Spinner()
        self.read_button = Gtk.Button(label=_("Read from Device"))

        self.warning_box_1 = self._create_warning_box(
            message=_(
                "Editing these values can be dangerous "
                "and may render your machine inoperable!"
            )
        )
        self.warning_box_2 = self._create_warning_box(
            message=_(
                "The device may restart or temporarily disconnect "
                "after a setting is changed."
            )
        )
        self.warning_box_2.set_margin_bottom(0)

        # The list widget is initially None.
        self.firmware_settings_widget = None

        self.settings_group.add(self.unsupported_banner)
        self.settings_group.add(self.error_banner)
        self.settings_group.add(self.main_layout_box)

        self.read_button.connect("clicked", self._on_read_clicked)
        self.action_box.append(self.spinner)
        self.action_box.append(Gtk.Box(hexpand=True))
        self.action_box.append(self.read_button)

        # Build the initial list widget and place it correctly in the layout
        self._rebuild_firmware_settings_widget()

        # Connect remaining signals
        driver_mgr.changed.connect(self.on_driver_changed)
        self.machine.firmware_settings_updated.connect(
            self._on_settings_updated
        )
        self.connect("destroy", self.on_destroy)
        self._update_ui_state()

    def _create_warning_box(self, message):
        """Helper to create a consistently styled warning box."""
        label = Gtk.Label(label=message, wrap=True, xalign=0, hexpand=True)
        icon = Gtk.Image(icon_name="dialog-warning-symbolic")
        box = Gtk.Box(
            spacing=12,
            margin_top=6,
            margin_bottom=6,
            margin_start=12,
            margin_end=12,
        )
        box.add_css_class("warning")
        box.append(icon)
        box.append(label)
        return box

    def on_destroy(self, _widget):
        """Disconnect signal handlers."""
        driver_mgr.changed.disconnect(self.on_driver_changed)
        self.machine.firmware_settings_updated.disconnect(
            self._on_settings_updated
        )
        task_mgr.cancel_task("device-settings-read")
        task_mgr.cancel_task("device-settings-write")

    def on_driver_changed(self, sender, driver):
        """Handles driver changes by rebuilding the settings widget."""
        self._rebuild_firmware_settings_widget()

    def _rebuild_firmware_settings_widget(self):
        """
        Rebuilds the central list widget and its surrounding static elements,
        preserving scroll position.
        """
        scroll_position = 0
        if self.firmware_settings_widget:
            adj = (
                self.firmware_settings_widget.scrolled_window.get_vadjustment()
            )
            scroll_position = adj.get_value()
            self.main_layout_box.remove(self.firmware_settings_widget)
            self.main_layout_box.remove(self.action_box)
            self.main_layout_box.remove(self.warning_box_1)
            self.main_layout_box.remove(self.warning_box_2)

        self.firmware_settings_widget = FirmwareSettingsWidget(
            machine=self.machine
        )
        self.firmware_settings_widget.set_vexpand(True)

        # Add widgets to the main layout box in the correct final order
        self.main_layout_box.append(self.firmware_settings_widget)
        self.main_layout_box.append(self.action_box)
        self.main_layout_box.append(self.warning_box_1)
        self.main_layout_box.append(self.warning_box_2)

        self.firmware_settings_widget.connect(
            "setting-apply", self._on_setting_apply
        )

        def restore_scroll():
            if not self.firmware_settings_widget:
                return GLib.SOURCE_REMOVE
            adj = (
                self.firmware_settings_widget.scrolled_window.get_vadjustment()
            )
            upper = adj.get_upper() - adj.get_page_size()
            if upper > 0:
                adj.set_value(min(scroll_position, upper))
            return GLib.SOURCE_REMOVE

        GLib.idle_add(restore_scroll)
        self._update_ui_state()

    def _update_ui_state(self, is_busy=False):
        driver = driver_mgr.driver
        is_supported = driver is not None and driver.supports_settings

        self.unsupported_banner.set_revealed(not is_supported)
        self.main_layout_box.set_visible(is_supported)

        if self._current_error:
            error = GLib.markup_escape_text(self._current_error)
            self.error_banner.set_title(
                f"<b>{_('Could not communicate:')}</b> {error}"
            )
            self.error_banner.set_revealed(True)
        else:
            self.error_banner.set_revealed(False)

        if is_busy:
            self.spinner.start()
            self.read_button.set_sensitive(False)
        else:
            self.spinner.stop()
            self.read_button.set_sensitive(is_supported)

    def _on_settings_updated(self, sender, **kwargs):
        """Callback for when the machine model's settings are updated."""
        self._current_error = None
        if self.firmware_settings_widget:
            self.firmware_settings_widget.populate_list()
        self._update_ui_state()

    def _on_setting_apply(self, widget, entry, key):
        """Handles the 'setting-apply' signal from the child widget."""
        new_value = entry.get_text()
        task_mgr.add_coroutine(
            self._write_and_refresh,
            key,
            new_value,
            key="device-settings-write",
            when_done=self._on_task_done,
        )

    def _on_read_clicked(self, _button=None):
        """Triggered when the 'Read from Device' button is clicked."""
        task_mgr.add_coroutine(
            self._read_from_device,
            key="device-settings-read",
            when_done=self._on_task_done,
        )

    def _on_task_done(self, task: Task):
        """Generic callback for when a task finishes."""
        self._update_ui_state(is_busy=False)

    async def _write_and_refresh(self, ctx, key, value):
        """Coroutine to write a setting and then refresh the list."""
        self._current_error = None
        self._update_ui_state(is_busy=True)
        try:
            await self.machine.write_setting_to_device(key, value)
        except (Exception, CancelledError) as e:
            logger.error(f"Failed to write setting: {e}")
            self._current_error = str(e)

    async def _read_from_device(self, ctx):
        """Coroutine to read settings from the device."""
        self._current_error = None
        self._update_ui_state(is_busy=True)
        try:
            await self.machine.read_settings_from_device()
        except (Exception, CancelledError) as e:
            logger.error(f"Failed to read settings: {e}")
            self._current_error = str(e)
