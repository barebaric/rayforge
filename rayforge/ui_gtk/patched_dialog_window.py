from gi.repository import Adw

"""
PatchedDialogWindow:
A replacement for Adw.Window that fixes wrong window
being focused when a dialog is closed on windows.
See:
https://bugzilla.gnome.org/show_bug.cgi?id=112404
& https://gitlab.gnome.org/GNOME/gtk/-/issues/7313
"""


class PatchedDialogWindow(Adw.Window):
    def do_close_request(self, *args) -> bool:
        parent = self.get_transient_for()
        # Focus the original parent
        if parent:
            parent.present()
        # Let GTK close the window
        return False
