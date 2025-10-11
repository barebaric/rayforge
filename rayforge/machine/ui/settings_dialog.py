from gi.repository import Adw
from ...camera.models import Camera
from ...config import camera_mgr
from ..models.machine import Machine
from .general_preferences_page import GeneralPreferencesPage
from .device_settings_page import DeviceSettingsPage
from .advanced_preferences_page import AdvancedPreferencesPage
from .laser_preferences_page import LaserPreferencesPage
from ...camera.ui.camera_preferences_page import CameraPreferencesPage


class MachineSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, *, machine: Machine, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine
        if machine.name:
            self.set_title(_(f"{machine.name} - Machine Preferences"))
        else:
            self.set_title(_("Machine Preferences"))

        # Make the dialog resizable
        self.set_size_request(-1, -1)

        # Create and add the preferences pages
        self.add(GeneralPreferencesPage(machine=self.machine))

        # Create the device page
        device_page = DeviceSettingsPage(machine=self.machine)
        device_page.show_toast.connect(self._on_show_toast)
        self.add(device_page)

        self.add(AdvancedPreferencesPage(machine=self.machine))
        self.add(LaserPreferencesPage(machine=self.machine))

        # Create and manage the decoupled camera page
        self.camera_page = CameraPreferencesPage()
        self.camera_page.camera_add_requested.connect(
            self._on_camera_add_requested
        )
        self.camera_page.camera_remove_requested.connect(
            self._on_camera_remove_requested
        )
        self.add(self.camera_page)

        # Sync UI with CameraManager signals
        camera_mgr.controller_added.connect(self._sync_camera_page)
        camera_mgr.controller_removed.connect(self._sync_camera_page)
        self.connect("destroy", self._on_destroy)

        # Initial population of all dependent pages
        self._sync_camera_page()

    def _on_show_toast(self, sender, message: str):
        """
        Handler to show the toast when requested by the child page.
        """
        self.add_toast(Adw.Toast(title=message, timeout=5))

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
        camera_mgr.controller_added.disconnect(self._sync_camera_page)
        camera_mgr.controller_removed.disconnect(self._sync_camera_page)
