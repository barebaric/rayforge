import re
import sys

from gi.repository import Adw, Gdk, Gtk

from ...usage import get_usage_tracker
from .keyboard import is_primary_modifier


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


def _detach_transient_for_macos(window: Gtk.Window) -> None:
    """
    When a window with a transient parent hides, GTK's macOS backend
    hands key input back with [parentWindow showAndMakeKey:], which
    tears native full-screen parent windows blank. Without the
    transient link the backend leaves the parent alone and AppKit
    returns focus to the remaining windows of the app on its own.
    """
    if sys.platform == "darwin" and window.get_transient_for() is not None:
        window.set_transient_for(None)


def _can_reactivate() -> bool:
    """
    On macOS, presenting the parent from the dialog-close path tears
    full-screen windows out of their native Space and leaves a blank
    window behind, and native full screen is not reliably reported by
    the backend, so the presentation is skipped unconditionally there.
    The system re-activates the remaining window of the app on its own
    when the dialog closes. Windows still needs the presentation
    (issue #115); elsewhere it is harmless.
    """
    return sys.platform != "darwin"


class PatchedDialogWindow(Adw.Window):
    def __init__(self, skip_usage_tracking: bool = False, **kwargs):
        super().__init__(**kwargs)
        _detach_transient_for_macos(self)
        self._tracked = False
        self._skip_usage_tracking = skip_usage_tracking
        self.connect("map", self._on_map)

        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _on_map(self, widget):
        if not self._tracked:
            self._tracked = True
            if not self._skip_usage_tracking:
                self._track_view()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape or (
            is_primary_modifier(state) and keyval == Gdk.KEY_w
        ):
            self.close()
            return True
        return False

    def _track_view(self):
        title = self.get_title() or self.__class__.__name__
        url = f"/{_camel_to_kebab(self.__class__.__name__)}"
        get_usage_tracker().track_page_view(url=url, title=title)

    def do_close_request(self, *args) -> bool:
        parent = self.get_transient_for()
        # Focus the original parent
        if parent and not parent.is_active() and _can_reactivate():
            parent.present()
        # Let GTK close the window
        return False


class PatchedMessageDialog(Adw.MessageDialog):
    def __init__(self, skip_usage_tracking: bool = False, **kwargs):
        super().__init__(**kwargs)
        _detach_transient_for_macos(self)
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
