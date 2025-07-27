from gi.repository import Adw  # type: ignore
from .generalpreferences import GeneralPreferencesPage
from .gcodepreferences import GCodePreferencesPage
from .laserheadpreferences import LaserHeadPreferencesPage
from .camerapreferences import CameraPreferencesPage


class MachineSettingsDialog(Adw.PreferencesDialog):
    def __init__(self, machine, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine

        # Make the dialog resizable
        self.set_size_request(-1, -1)

        # Create and add the preferences pages
        self.add(GeneralPreferencesPage(machine=self.machine))
        self.add(GCodePreferencesPage(machine=self.machine))
        self.add(LaserHeadPreferencesPage(machine=self.machine))
        self.add(CameraPreferencesPage(machine=self.machine))
