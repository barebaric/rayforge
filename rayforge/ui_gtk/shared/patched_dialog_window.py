import re
from gi.repository import Adw
from ...usage import get_usage_tracker


def _camel_to_kebab(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


"""
PatchedDialogWindow:
A replacement for Adw.Window that fixes wrong window
being focused when a dialog is closed on windows.
See:
https://bugzilla.gnome.org/show_bug.cgi?id=112404
& https://gitlab.gnome.org/GNOME/gtk/-/issues/7313
"""


class PatchedDialogWindow(Adw.Window):
    def __init__(self, skip_usage_tracking: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._tracked = False
        self._skip_usage_tracking = skip_usage_tracking
        self.connect("map", self._on_map)

    def _on_map(self, widget):
        if not self._tracked:
            self._tracked = True
            if not self._skip_usage_tracking:
                self._track_view()

    def _track_view(self):
        title = self.get_title() or self.__class__.__name__
        url = f"/{_camel_to_kebab(self.__class__.__name__)}"
        get_usage_tracker().track_page_view(url=url, title=title)

    def do_close_request(self, *args) -> bool:
        parent = self.get_transient_for()
        # Focus the original parent
        if parent:
            parent.present()
        # Let GTK close the window
        return False


class PatchedMessageDialog(Adw.MessageDialog):
    def __init__(self, skip_usage_tracking: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._tracked = False
        self._skip_usage_tracking = skip_usage_tracking
        self.connect("map", self._on_map)

    def _on_map(self, widget):
        if not self._tracked:
            self._tracked = True
            if not self._skip_usage_tracking:
                self._track_view()

    def _track_view(self):
        title = self.get_heading() or self.__class__.__name__
        url = f"/{_camel_to_kebab(self.__class__.__name__)}"
        get_usage_tracker().track_page_view(url=url, title=title)
