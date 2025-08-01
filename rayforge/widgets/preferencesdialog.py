from gi.repository import Adw  # type: ignore
from .generalprefs import GeneralPreferencesPage


class PreferencesDialog(Adw.PreferencesDialog):
    """
    The main preferences dialog for the application.
    It contains pages for general application settings and machine management.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set dialog properties
        self.set_search_enabled(False)
        self.set_size_request(600, 400)

        # Create and add the preferences pages
        # The MachineManagementPage will be added here later.
        self.add(GeneralPreferencesPage())
