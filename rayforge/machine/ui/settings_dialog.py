from gi.repository import Gtk, Adw
from ...camera.models import Camera
from ...camera.ui.camera_preferences_page import CameraPreferencesPage
from ...context import get_context
from ...icons import get_icon
from ..models.machine import Machine
from .general_preferences_page import GeneralPreferencesPage
from .device_settings_page import DeviceSettingsPage
from .advanced_preferences_page import AdvancedPreferencesPage
from .laser_preferences_page import LaserPreferencesPage


class MachineSettingsDialog(Adw.Window):
    def __init__(self, *, machine: Machine, transient_for=None, **kwargs):
        super().__init__(**kwargs)
        if transient_for:
            self.set_transient_for(transient_for)
        self.machine = machine
        if machine.name:
            self.set_title(_(f"{machine.name} - Machine Settings"))
        else:
            self.set_title(_("Machine Settings"))
        self.set_size_request(700, 700)

        # --- Layout ---
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        toolbar_view = Adw.ToolbarView()
        self.toast_overlay.set_child(toolbar_view)

        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        # View Stack
        self.view_stack = Adw.ViewStack()
        toolbar_view.set_content(self.view_stack)

        # --- Custom Switcher (Icon + Text horizontal) ---
        switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        switcher_box.add_css_class("linked")
        header_bar.set_title_widget(switcher_box)

        # General Tab
        btn_general = Gtk.ToggleButton()
        btn_general.set_child(
            self._create_tab_child(
                _("General"), "machine-settings-general-symbolic"
            )
        )
        btn_general.connect("toggled", self._on_tab_toggled, "general")
        switcher_box.append(btn_general)

        # Advanced Tab
        btn_advanced = Gtk.ToggleButton(group=btn_general)
        btn_advanced.set_child(
            self._create_tab_child(
                _("Advanced"), "machine-settings-advanced-symbolic"
            )
        )
        btn_advanced.connect("toggled", self._on_tab_toggled, "advanced")
        switcher_box.append(btn_advanced)

        # Device Tab
        btn_device = Gtk.ToggleButton(group=btn_general)
        btn_device.set_child(
            self._create_tab_child(_("Device"), "settings-symbolic")
        )
        btn_device.connect("toggled", self._on_tab_toggled, "device")
        switcher_box.append(btn_device)

        # Laser Tab
        btn_laser = Gtk.ToggleButton(group=btn_general)
        btn_laser.set_child(
            self._create_tab_child(_("Laser"), "laser-on-symbolic")
        )
        btn_laser.connect("toggled", self._on_tab_toggled, "laser")
        switcher_box.append(btn_laser)

        # Camera Tab
        btn_camera = Gtk.ToggleButton(group=btn_general)
        btn_camera.set_child(
            self._create_tab_child(_("Camera"), "camera-on-symbolic")
        )
        btn_camera.connect("toggled", self._on_tab_toggled, "camera")
        switcher_box.append(btn_camera)

        # --- Page 1: General ---
        self.view_stack.add_named(
            GeneralPreferencesPage(machine=self.machine), "general"
        )

        # --- Page 2: Advanced ---
        self.view_stack.add_named(
            AdvancedPreferencesPage(machine=self.machine), "advanced"
        )

        # --- Page 3: Device ---
        device_page = DeviceSettingsPage(machine=self.machine)
        device_page.show_toast.connect(self._on_show_toast)
        self.view_stack.add_named(device_page, "device")

        # --- Page 4: Laser ---
        self.view_stack.add_named(
            LaserPreferencesPage(machine=self.machine), "laser"
        )

        # --- Page 5: Camera ---
        self.camera_page = CameraPreferencesPage()
        self.camera_page.camera_add_requested.connect(
            self._on_camera_add_requested
        )
        self.camera_page.camera_remove_requested.connect(
            self._on_camera_remove_requested
        )
        self.view_stack.add_named(self.camera_page, "camera")

        # Sync UI with CameraManager signals
        camera_mgr = get_context().camera_mgr
        camera_mgr.controller_added.connect(self._sync_camera_page)
        camera_mgr.controller_removed.connect(self._sync_camera_page)
        self.connect("destroy", self._on_destroy)

        # Initial population of all dependent pages
        self._sync_camera_page()

        # Set default tab
        btn_general.set_active(True)

    def _create_tab_child(self, text: str, icon_name: str) -> Gtk.Widget:
        """Creates a box with an icon and a label for the toggle button."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        icon = get_icon(icon_name)
        label = Gtk.Label(label=text)
        box.append(icon)
        box.append(label)
        return box

    def _on_tab_toggled(self, button, page_name):
        if button.get_active():
            self.view_stack.set_visible_child_name(page_name)

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
