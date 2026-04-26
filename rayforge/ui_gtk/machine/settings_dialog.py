import logging
import webbrowser
from pathlib import Path
from typing import Optional
from gettext import gettext as _

from gi.repository import Adw, Gdk, GLib, Gtk

from ...camera.models import Camera
from ...context import get_context
from ... import const
from ...machine.driver import (
    get_driver_cls,
    DriverMaturity,
    DRIVER_MATURITY_LABELS,
)
from ...machine.models.machine import Machine
from ..camera.camera_preferences_page import CameraPreferencesPage
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.gtk import apply_css
from .advanced_preferences_page import AdvancedPreferencesPage
from .device_settings_page import DeviceSettingsPage
from .general_preferences_page import GeneralPreferencesPage
from .gcode_settings_page import GcodeSettingsPage
from .hardware_page import HardwarePage
from .hooks_macros_page import HooksMacrosPage
from .laser_preferences_page import LaserPreferencesPage
from .maintenance_page import MaintenancePage
from .rotary_module_page import RotaryModulePage
from .nogo_zones_page import NogoZonesPage

logger = logging.getLogger(__name__)

apply_css("""
.maturity-warning {
    background-color: alpha(@warning_color, 0.15);
    padding: 10px 28px;
}
.maturity-link {
    text-decoration: underline;
}
""")


class MachineSettingsDialog(PatchedDialogWindow):
    def __init__(
        self,
        *,
        machine: Machine,
        transient_for=None,
        initial_page: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(skip_usage_tracking=True, **kwargs)
        if transient_for:
            self.set_transient_for(transient_for)
        self.machine = machine
        self._row_to_page_name = {}
        self._initial_page = initial_page
        self._gcode_row: Optional[Gtk.ListBoxRow] = None
        self._gcode_stack_page: Optional[Gtk.StackPage] = None
        if machine.name:
            self.set_title(
                _("{machine_name} - Machine Settings").format(
                    machine_name=machine.name
                )
            )
        else:
            self.set_title(_("Machine Settings"))
        self.set_default_size(800, 800)

        # --- Layout ---
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        # Main layout container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(main_box)

        # Header bar
        header_bar = Adw.HeaderBar()
        export_button = Gtk.Button(child=get_icon("share-symbolic"))
        export_button.set_tooltip_text(_("Export Machine Profile"))
        export_button.add_css_class("flat")
        export_button.connect("clicked", self._on_export_clicked)
        header_bar.pack_end(export_button)
        main_box.append(header_bar)

        # Maturity warning banner
        self.maturity_banner = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            hexpand=True,
        )
        self.maturity_banner.add_css_class("maturity-warning")
        self._maturity_icon = get_icon("warning-symbolic")
        self._maturity_icon.add_css_class("warning")
        self._maturity_label = Gtk.Label(wrap=True, xalign=0, hexpand=True)
        self._maturity_label.add_css_class("warning-label")

        self._maturity_link = Gtk.Label(
            label=_("Report an issue"),
            wrap=False,
            xalign=0,
            hexpand=False,
        )
        self._maturity_link.add_css_class("warning-label")
        self._maturity_link.add_css_class("maturity-link")
        link_click = Gtk.GestureClick.new()
        link_click.connect(
            "pressed",
            lambda *_: webbrowser.open(const.ISSUES_URL),
        )
        self._maturity_link.add_controller(link_click)
        link_motion = Gtk.EventControllerMotion()
        link_motion.connect(
            "enter",
            lambda *_: self._maturity_link.set_cursor(
                Gdk.Cursor.new_from_name("pointer")
            ),
        )
        link_motion.connect(
            "leave",
            lambda *_: self._maturity_link.set_cursor(None),
        )
        self._maturity_link.add_controller(link_motion)

        self.maturity_banner.append(self._maturity_icon)
        self.maturity_banner.append(self._maturity_label)
        self.maturity_banner.append(self._maturity_link)
        self.maturity_banner.set_visible(False)
        main_box.append(self.maturity_banner)

        # Navigation Split View for sidebar and content
        split_view = Adw.NavigationSplitView(vexpand=True)
        main_box.append(split_view)

        # Sidebar
        self.sidebar_list = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.SINGLE,
            css_classes=["navigation-sidebar"],
        )
        sidebar_page = Adw.NavigationPage.new(
            self.sidebar_list, _("Categories")
        )
        split_view.set_sidebar(sidebar_page)

        # Content Stack
        self.content_stack = Gtk.Stack()

        # --- Page 1: General ---
        general_page = GeneralPreferencesPage(machine=self.machine)
        self.content_stack.add_titled(general_page, "general", _("General"))

        # --- Page 2: Hardware ---
        hardware_page = HardwarePage(machine=self.machine)
        self.content_stack.add_titled(hardware_page, "hardware", _("Hardware"))

        # --- Page 3: Advanced ---
        advanced_page = AdvancedPreferencesPage(machine=self.machine)
        self.content_stack.add_titled(advanced_page, "advanced", _("Advanced"))

        # --- Page 4: G-code ---
        gcode_page = GcodeSettingsPage(machine=self.machine)
        self.content_stack.add_titled(gcode_page, "gcode", _("G-code"))
        self._gcode_stack_page = self.content_stack.get_page(gcode_page)

        # --- Page 5: Hooks & Macros ---
        hooks_macros_page = HooksMacrosPage(machine=self.machine)
        self.content_stack.add_titled(
            hooks_macros_page, "hooks-macros", _("Hooks & Macros")
        )

        # --- Page 6: Device ---
        device_page = DeviceSettingsPage(machine=self.machine)
        device_page.show_toast.connect(self._on_show_toast)
        self.content_stack.add_titled(device_page, "device", _("Device"))

        # --- Page 7: Laser ---
        laser_page = LaserPreferencesPage(machine=self.machine)
        self.content_stack.add_titled(laser_page, "laser", _("Laser"))

        # --- Page 8: Rotary Module ---
        rotary_module_page = RotaryModulePage(machine=self.machine)
        self.content_stack.add_titled(
            rotary_module_page, "rotary-module", _("Rotary Module")
        )

        # --- Page 9: No-Go Zones ---
        nogo_zones_page = NogoZonesPage(machine=self.machine)
        self.content_stack.add_titled(
            nogo_zones_page, "nogo-zones", _("No-Go Zones")
        )

        # --- Page 10: Camera ---
        self.camera_page = CameraPreferencesPage()
        self.camera_page.camera_add_requested.connect(
            self._on_camera_add_requested
        )
        self.camera_page.camera_remove_requested.connect(
            self._on_camera_remove_requested
        )
        self.content_stack.add_titled(self.camera_page, "camera", _("Camera"))

        # --- Page 9: Maintenance ---
        maintenance_page = MaintenancePage(machine=self.machine)
        self.content_stack.add_titled(
            maintenance_page, "maintenance", _("Maintenance")
        )

        # Create the content's NavigationPage wrapper
        pages = self.content_stack.get_pages()
        first_stack_page = pages.get_item(0)  # type: ignore
        initial_title = first_stack_page.get_title()
        self.content_page = Adw.NavigationPage.new(
            self.content_stack, initial_title
        )
        split_view.set_content(self.content_page)

        # Populate sidebar with rows
        self._add_sidebar_row(
            _("General"), "machine-settings-general-symbolic", "general"
        )
        self._add_sidebar_row(_("Hardware"), "hardware-symbolic", "hardware")
        self._add_sidebar_row(
            _("Advanced"), "machine-settings-advanced-symbolic", "advanced"
        )
        self._add_sidebar_row(_("G-code"), "gcode-symbolic", "gcode")
        self._gcode_row = self.sidebar_list.get_row_at_index(3)
        self._add_sidebar_row(
            _("Hooks & Macros"), "code-symbolic", "hooks-macros"
        )
        self._add_sidebar_row(_("Device"), "settings-symbolic", "device")
        self._add_sidebar_row(_("Laser"), "laser-on-symbolic", "laser")
        self._add_sidebar_row(
            _("Rotary Module"), "rotary-symbolic", "rotary-module"
        )
        self._add_sidebar_row(
            _("No-Go Zones"), "action-unavailable-symbolic", "nogo-zones"
        )
        self._add_sidebar_row(_("Camera"), "camera-on-symbolic", "camera")
        self._add_sidebar_row(
            _("Maintenance"), "timer-symbolic", "maintenance"
        )

        # Connect sidebar selection
        self.sidebar_list.connect("row-selected", self._on_row_selected)

        # Sync UI with CameraManager signals
        camera_mgr = get_context().camera_mgr
        camera_mgr.controller_added.connect(self._sync_camera_page)
        camera_mgr.controller_removed.connect(self._sync_camera_page)
        self.connect("destroy", self._on_destroy)

        # React to driver changes (e.g. show/hide G-code page)
        self.machine.changed.connect(self._on_machine_changed)

        # Initial population of all dependent pages
        self._sync_camera_page()
        self._update_gcode_page_visibility()
        self._update_maturity_banner()

        # Select the specified page or first row by default
        if self._initial_page:
            for row, page_name in self._row_to_page_name.items():
                if page_name == self._initial_page:
                    self.sidebar_list.select_row(row)
                    break
        else:
            self.sidebar_list.select_row(self.sidebar_list.get_row_at_index(0))

    def _on_machine_changed(self, sender=None, **kwargs):
        self._update_gcode_page_visibility()
        self._update_maturity_banner()

    def _update_maturity_banner(self):
        maturity = DriverMaturity.STABLE
        if self.machine.driver_name:
            driver_cls = get_driver_cls(self.machine.driver_name)
            maturity = driver_cls.maturity
        label = DRIVER_MATURITY_LABELS.get(maturity, "")
        if label:
            self._maturity_label.set_text(label)
            self.maturity_banner.set_visible(True)
        else:
            self.maturity_banner.set_visible(False)

    def _update_gcode_page_visibility(self):
        uses_gcode = True
        if self.machine.driver_name:
            driver_cls = get_driver_cls(self.machine.driver_name)
            uses_gcode = driver_cls.uses_gcode

        if self._gcode_stack_page:
            self._gcode_stack_page.set_visible(uses_gcode)
        if self._gcode_row:
            self._gcode_row.set_visible(uses_gcode)

        if not uses_gcode:
            selected = self.sidebar_list.get_selected_row()
            if selected is self._gcode_row:
                self.sidebar_list.select_row(
                    self.sidebar_list.get_row_at_index(0)
                )

    def _on_export_clicked(self, button):
        """Opens a folder chooser to export the machine as a zip."""
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Export Machine Profile"))
        dialog.select_folder(self, None, self._on_export_folder_selected)

    def _on_export_folder_selected(self, dialog, result):
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if not folder:
            return
        dest = Path(folder.get_path())
        context = get_context()
        try:
            zip_path = context.device_profile_mgr.export_machine(
                self.machine, dest, context.model_mgr
            )
            self.toast_overlay.add_toast(
                Adw.Toast(
                    title=_("Exported to {path}").format(path=zip_path.name),
                    timeout=5,
                )
            )
        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.toast_overlay.add_toast(
                Adw.Toast(title=_("Export failed: {error}").format(error=e))
            )

    def _add_sidebar_row(
        self, label_text: str, icon_name: str, page_name: str
    ):
        """Adds a row to the sidebar with an icon and label."""
        row = Gtk.ListBoxRow()
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_start=12,
            margin_end=12,
            margin_top=6,
            margin_bottom=6,
        )
        icon = get_icon(icon_name)
        label = Gtk.Label(label=label_text, xalign=0)
        box.append(icon)
        box.append(label)
        row.set_child(box)
        self._row_to_page_name[row] = page_name
        self.sidebar_list.append(row)

    def _on_row_selected(self, listbox, row):
        """Handler for when a row is selected in the sidebar."""
        if row:
            page_name = self._row_to_page_name[row]
            self.content_stack.set_visible_child_name(page_name)
            child = self.content_stack.get_child_by_name(page_name)
            if child:
                stack_page = self.content_stack.get_page(child)
                if stack_page:
                    title = stack_page.get_title()
                    if title:
                        self.content_page.set_title(title)

    def _on_show_toast(self, sender, message: str):
        """
        Handler to show the toast when requested by the child page.
        """
        self.toast_overlay.add_toast(Adw.Toast(title=message, timeout=5))

    def _on_camera_add_requested(self, sender, *, device_id: str):
        """Handles the request to add a new camera to the machine."""
        if any(c.device_id == device_id for c in self.machine.cameras):
            return  # Safety check

        new_camera = Camera(
            _("Camera {device_id}").format(device_id=device_id),
            device_id,
        )
        new_camera.enabled = True
        self.machine.add_camera(new_camera)
        # The machine.changed signal will handle the UI update

    def _on_camera_remove_requested(self, sender, *, camera: Camera):
        """Handles the request to remove a camera from the machine."""
        camera.enabled = False
        self.machine.remove_camera(camera)
        # The machine.changed signal will handle the UI update

    def _sync_camera_page(self, sender=None, **kwargs):
        """Updates child pages that depend on the list of live controllers."""
        camera_mgr = get_context().camera_mgr
        # Get all live controllers and filter them for this specific
        # machine
        all_controllers = camera_mgr.controllers
        machine_camera_device_ids = {c.device_id for c in self.machine.cameras}
        relevant_controllers = [
            c
            for c in all_controllers
            if c.config.device_id in machine_camera_device_ids
        ]
        self.camera_page.set_controllers(relevant_controllers)

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        camera_mgr = get_context().camera_mgr
        camera_mgr.controller_added.disconnect(self._sync_camera_page)
        camera_mgr.controller_removed.disconnect(self._sync_camera_page)
        self.machine.changed.disconnect(self._on_machine_changed)
