from gettext import gettext as _
from blinker import Signal
from gi.repository import Adw, Gtk
from ..shared.gtk import apply_css
from ...context import get_context
from ...machine.device.package import DevicePackage


css = """
.profile-selector-list {
    background: none;
}
"""


class MachineProfileSelectorDialog(Adw.MessageDialog):
    """
    A dialog for selecting a machine profile from a list.

    The dialog is confirmed by activating a row (double-click or Enter).
    """

    profile_selected = Signal()

    class _ProfileRow(Adw.ActionRow):
        """A custom row to hold a reference to its device package."""

        def __init__(self, package: DevicePackage, **kwargs):
            super().__init__(**kwargs)
            self.package: DevicePackage = package

    def __init__(self, **kwargs):
        """Initializes the Machine Profile Selector dialog."""
        super().__init__(**kwargs)
        self.set_heading(_("Add a New Machine"))
        self.set_body(_("Select a machine profile to use as a template."))
        self.set_transient_for(kwargs.get("transient_for"))

        apply_css(css)

        # Build the custom content area
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
        # A single click now selects the row, making it ready for activation.
        self.profile_list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.profile_list_box.add_css_class("profile-selector-list")
        self.profile_list_box.connect("row-activated", self._on_row_activated)
        scrolled_window.set_child(self.profile_list_box)

        self._populate_profile_list()

        self.set_extra_child(content)

        # Add only a "Cancel" response. The dialog closes on any response.
        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

    def _populate_profile_list(self):
        """Fills the list box with available device packages."""
        packages = get_context().device_pkg_mgr.get_all()

        for pkg in packages:
            row = self._ProfileRow(
                package=pkg,
                title=pkg.name,
                subtitle=pkg.meta.description,
                activatable=True,
            )
            self.profile_list_box.append(row)

    def _on_row_activated(self, listbox: Gtk.ListBox, row: _ProfileRow):
        """
        Handles row activation, emits the signal, and closes the dialog.
        """
        self.profile_selected.send(self, package=row.package)
        self.close()
