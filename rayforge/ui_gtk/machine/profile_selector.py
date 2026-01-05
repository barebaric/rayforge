from blinker import Signal
from gi.repository import Adw, Gtk
from ..shared.gtk import apply_css
from ...machine.models.profile import MachineProfile, PROFILES


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
        """A custom row to hold a reference to its machine profile."""

        def __init__(self, profile: MachineProfile, **kwargs):
            super().__init__(**kwargs)
            self.profile: MachineProfile = profile

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
        """Fills the list box with available machine profiles."""
        sorted_profiles = sorted(PROFILES, key=lambda p: p.name.lower())

        for profile in sorted_profiles:
            row = self._ProfileRow(
                profile=profile,
                title=profile.name,
                subtitle=getattr(profile, "description", ""),
                activatable=True,
            )
            self.profile_list_box.append(row)

    def _on_row_activated(self, listbox: Gtk.ListBox, row: _ProfileRow):
        """
        Handles row activation, emits the signal, and closes the dialog.
        """
        self.profile_selected.send(self, profile=row.profile)
        self.close()
