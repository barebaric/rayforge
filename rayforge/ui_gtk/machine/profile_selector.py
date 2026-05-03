import logging
from gettext import gettext as _
from typing import Optional

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ..shared.gtk import apply_css
from ...context import get_context
from ...machine.device.profile import DeviceProfile
from .config_wizard import ConfigWizard
from .profile_importer import open_profile_zip

logger = logging.getLogger(__name__)

css = """
.profile-selector-list {
    background: none;
}

.profile-selector-list row:first-child {
    border-radius: 12px 12px 0 0;
}

.profile-selector-list row:last-child {
    border-radius: 0 0 12px 12px;
}

.profile-selector-list row:only-child {
    border-radius: 12px;
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
        self.set_size_request(580, -1)

        apply_css(css)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content.set_margin_top(12)

        self.search_entry = Gtk.SearchEntry()
        self.search_entry.set_placeholder_text(_("Search devices…"))
        self.search_entry.connect(
            "search-changed", lambda *_: self._filter_and_populate_list()
        )
        content.append(self.search_entry)

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

        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=8,
            halign=Gtk.Align.END,
        )

        other_button = Gtk.Button(label=_("Other Device…"))
        other_button.connect("clicked", self._on_other_device_clicked)
        button_box.append(other_button)

        import_button = Gtk.Button(label=_("Import from File…"))
        import_button.add_css_class("suggested-action")
        import_button.connect("clicked", self._on_import_clicked)
        button_box.append(import_button)

        content.append(button_box)

        self.set_extra_child(content)

        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

        self._all_profiles = get_context().device_profile_mgr.get_all()
        self._filter_and_populate_list()

    def _filter_and_populate_list(self):
        """Clears and repopulates the list based on search text."""
        search_text = self.search_entry.get_text().lower()

        while child := self.profile_list_box.get_row_at_index(0):
            self.profile_list_box.remove(child)

        for pkg in self._all_profiles:
            if (
                search_text
                and search_text not in pkg.name.lower()
                and search_text not in pkg.meta.description.lower()
            ):
                continue
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

    def _on_other_device_clicked(self, button):
        wizard = ConfigWizard(transient_for=self)
        wizard.profile_created.connect(self._on_wizard_result)
        wizard.present()

    def _on_wizard_result(self, sender, *, profile: DeviceProfile):
        self.close()
        GLib.idle_add(self._emit_profile_selected, profile)

    def _emit_profile_selected(self, profile: DeviceProfile):
        self.profile_selected.send(self, profile=profile)
        return GLib.SOURCE_REMOVE

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
