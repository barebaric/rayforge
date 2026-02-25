from gi.repository import Adw
from ...usage import get_usage_tracker


class TrackedPreferencesPage(Adw.PreferencesPage):
    key = ""
    path_prefix = "/settings/"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connect("map", self._on_map)

    def _on_map(self, widget):
        if self.key:
            get_usage_tracker().track_page_view(
                f"{self.path_prefix}{self.key}", self.key
            )
