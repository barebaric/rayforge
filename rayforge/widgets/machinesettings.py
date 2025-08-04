from gi.repository import Adw  # type: ignore
from ..models.machine import Machine
from .generalpreferences import GeneralPreferencesPage
from .firmwaresettingspage import FirmwareSettingsPage
from .advancedpreferences import AdvancedPreferencesPage
from .laserheadpreferences import LaserHeadPreferencesPage
from .camerapreferences import CameraPreferencesPage


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
        self.add(FirmwareSettingsPage(machine=self.machine))
        self.add(AdvancedPreferencesPage(machine=self.machine))
        self.add(LaserHeadPreferencesPage(machine=self.machine))
        self.add(CameraPreferencesPage(machine=self.machine))
