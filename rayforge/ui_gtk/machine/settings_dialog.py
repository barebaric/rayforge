from gi.repository import Gtk, Adw, Gdk
from ...camera.models import Camera
from ..camera.camera_preferences_page import CameraPreferencesPage
from ...context import get_context
from ..icons import get_icon
from ...machine.models.machine import Machine
from .general_preferences_page import GeneralPreferencesPage
from .device_settings_page import DeviceSettingsPage
from .advanced_preferences_page import AdvancedPreferencesPage
from .laser_preferences_page import LaserPreferencesPage
from .machine_hours_page import MachineHoursPage


class MachineSettingsDialog(Adw.Window):
    def __init__(self, *, machine: Machine, transient_for=None, **kwargs):
        super().__init__(**kwargs)
        if transient_for:
            self.set_transient_for(transient_for)
        self.machine = machine
        self._row_to_page_name = {}
        if machine.name:
            self.set_title(_(f"{machine.name} - Machine Settings"))
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
        main_box.append(header_bar)

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

        # --- Page 2: Advanced ---
        advanced_page = AdvancedPreferencesPage(machine=self.machine)
        self.content_stack.add_titled(advanced_page, "advanced", _("Advanced"))

        # --- Page 3: Device ---
        device_page = DeviceSettingsPage(machine=self.machine)
        device_page.show_toast.connect(self._on_show_toast)
        self.content_stack.add_titled(device_page, "device", _("Device"))

        # --- Page 4: Laser ---
        laser_page = LaserPreferencesPage(machine=self.machine)
        self.content_stack.add_titled(laser_page, "laser", _("Laser"))

        # --- Page 5: Camera ---
        self.camera_page = CameraPreferencesPage()
        self.camera_page.camera_add_requested.connect(
            self._on_camera_add_requested
        )
        self.camera_page.camera_remove_requested.connect(
            self._on_camera_remove_requested
        )
        self.content_stack.add_titled(self.camera_page, "camera", _("Camera"))

        # --- Page 6: Machine Hours ---
        hours_page = MachineHoursPage(machine=self.machine)
        self.content_stack.add_titled(hours_page, "hours", _("Hours"))

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
        self._add_sidebar_row(
            _("Advanced"), "machine-settings-advanced-symbolic", "advanced"
        )
        self._add_sidebar_row(_("Device"), "settings-symbolic", "device")
        self._add_sidebar_row(_("Laser"), "laser-on-symbolic", "laser")
        self._add_sidebar_row(_("Camera"), "camera-on-symbolic", "camera")
        self._add_sidebar_row(_("Hours"), "timer-symbolic", "hours")

        # Connect sidebar selection
        self.sidebar_list.connect("row-selected", self._on_row_selected)

        # Sync UI with CameraManager signals
        camera_mgr = get_context().camera_mgr
        camera_mgr.controller_added.connect(self._sync_camera_page)
        camera_mgr.controller_removed.connect(self._sync_camera_page)
        self.connect("destroy", self._on_destroy)

        # Add a key controller to close the dialog on Escape press
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

        # Initial population of all dependent pages
        self._sync_camera_page()

        # Select first row by default
        self.sidebar_list.select_row(self.sidebar_list.get_row_at_index(0))

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
            stack_page = self.content_stack.get_page(page_name)
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

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events, closing the dialog on Escape or Ctrl+W."""
        has_ctrl = state & Gdk.ModifierType.CONTROL_MASK

        if keyval == Gdk.KEY_Escape or (has_ctrl and keyval == Gdk.KEY_w):
            self.close()
            return True
        return False

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        camera_mgr = get_context().camera_mgr
        camera_mgr.controller_added.disconnect(self._sync_camera_page)
        camera_mgr.controller_removed.disconnect(self._sync_camera_page)
