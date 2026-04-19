import logging
from gettext import gettext as _
from typing import Optional

from blinker import Signal
from gi.repository import Adw, Gtk

from ..icons import get_icon
from ..shared.gtk import apply_css
from ...context import get_context
from ...machine.device.profile import DeviceProfile
from .profile_importer import open_profile_zip

logger = logging.getLogger(__name__)

css = """
.profile-selector-list {
    background: none;
}
"""


class MachineProfileSelectorDialog(Adw.MessageDialog):
    """
    A dialog for selecting a machine profile from a list.

    The dialog is confirmed by activating a row (double-click or Enter).
    An "Import from File" button allows installing a profile from a zip.
    """

    profile_selected = Signal()

    class _ProfileRow(Adw.ActionRow):
        """A custom row to hold a reference to its device profile."""

        def __init__(self, profile: DeviceProfile, **kwargs):
            super().__init__(**kwargs)
            self.profile: DeviceProfile = profile

    def __init__(self, **kwargs):
        """Initializes the Machine Profile Selector dialog."""
        super().__init__(**kwargs)
        self.set_heading(_("Add a New Machine"))
        self.set_body(_("Select a machine profile to use as a template."))
        self.set_transient_for(kwargs.get("transient_for"))

        apply_css(css)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)

        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        scrolled_window.set_min_content_height(300)
        scrolled_window.set_vexpand(True)
        scrolled_window.add_css_class("card")
        content.append(scrolled_window)

        self.profile_list_box = Gtk.ListBox()
        self.profile_list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.profile_list_box.add_css_class("profile-selector-list")
        self.profile_list_box.connect("row-activated", self._on_row_activated)
        scrolled_window.set_child(self.profile_list_box)

        import_button = Gtk.Button()
        import_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        import_box.append(get_icon("open-symbolic"))
        import_box.append(Gtk.Label(label=_("Import from File…")))
        import_button.set_child(import_box)
        import_button.connect("clicked", self._on_import_clicked)
        content.append(import_button)

        self._populate_profile_list()

        self.set_extra_child(content)

        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

    def _populate_profile_list(self):
        """Fills the list box with available device profiles."""
        profiles = get_context().device_profile_mgr.get_all()

        for pkg in profiles:
            row = self._ProfileRow(
                profile=pkg,
                title=pkg.name,
                subtitle=pkg.meta.description,
                activatable=True,
            )
            self.profile_list_box.append(row)

    def _on_row_activated(self, listbox: Gtk.ListBox, row: _ProfileRow):
        self.profile_selected.send(self, profile=row.profile)
        self.close()

    def _on_import_clicked(self, button):
        open_profile_zip(self, self._on_import_result)

    def _on_import_result(
        self, profile: Optional[DeviceProfile], error: Optional[str]
    ):
        if error is not None:
            self._show_import_error(error)
            return
        self.profile_selected.send(self, profile=profile)
        self.close()

    def _show_import_error(self, message: str):
        error_dialog = Adw.MessageDialog(
            transient_for=self,
            modal=True,
            heading=_("Import Failed"),
            body=message,
        )
        error_dialog.add_response("ok", _("OK"))
        error_dialog.set_default_response("ok")
        error_dialog.present()
