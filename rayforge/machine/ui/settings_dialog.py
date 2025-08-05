from gi.repository import Adw  # type: ignore
from ..models.machine import Machine
from .general_preferences_page import GeneralPreferencesPage
from .firmware_settings_page import FirmwareSettingsPage
from .advanced_preferences_page import AdvancedPreferencesPage
from .laser_preferences_page import LaserPreferencesPage
from ...camera.ui.camera_preferences_page import CameraPreferencesPage


class MachineSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, machine: Machine, **kwargs):
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

        # Create the firmware page
        firmware_page = FirmwareSettingsPage(machine=self.machine)
        firmware_page.show_toast.connect(self._on_show_toast)
        self.add(firmware_page)

        self.add(AdvancedPreferencesPage(machine=self.machine))
        self.add(LaserPreferencesPage(machine=self.machine))
        self.add(CameraPreferencesPage(machine=self.machine))

    def _on_show_toast(self, sender, message: str):
        """
        Handler to show the toast when requested by the child page.
        """
        self.add_toast(Adw.Toast(title=message, timeout=5))
