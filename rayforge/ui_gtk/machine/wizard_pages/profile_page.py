"""Step 1 — Pick source.

Offers the user three entry points:

* **Known profile**: pick a built-in / installed device profile from a
  searchable list. The chosen profile pre-fills driver + dims + head;
  Step 3 (Connection) is still required because profiles never carry
  host-specific values like the USB device path or OctoPrint API key.
* **Other / unknown machine**: skip to Step 2 where the user picks
  the controller class manually.
* **Import .lbdev / .zip**: pull a snapshot from disk. Same treatment
  as a known profile but Step 3 is also prefilled with any connection
  args saved in the snapshot.

When a discovered device is pending (no profile matched it
automatically), the list is pre-filtered to profiles of the same
vendor and controller, since the collected identity says which
machine family it belongs to. Searching always covers everything.
"""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Adw, Gtk

from ....context import get_context
from ....machine.device.profile import DeviceProfile
from ....machine.driver import DiscoveredDevice, normalize_tokens
from ..profile_importer import open_profile_file
from . import WizardPage, _makePreferencesGroup


class _ProfileRow(Adw.ActionRow):
    """A custom row to hold a reference to its device profile."""

    def __init__(self, profile: DeviceProfile, **kwargs):
        super().__init__(**kwargs)
        self.profile: DeviceProfile = profile


class ProfilePage(WizardPage):
    step_number = 2
    title = _("Choose a Machine")
    subtitle = _("Pick a starting point for the new machine.")
    # The page advances only through explicit source selection (row
    # activation, Import, or Unlisted Device, so the wizard's
    # generic "Next" button is hidden here — it would be a dead button.
    next_on_footer = False

    # Selected source signal, sent with a source-kind tag and an
    # optional profile payload. The orchestrator uses this to choose
    # the next page.
    def __init__(self, wizard, **kwargs):
        self._all_profiles: list[DeviceProfile] = []
        self._suggested_profiles: list[DeviceProfile] = []
        self.source_selected = Signal()
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.hint_row = Adw.ActionRow(
            title=_("A device was detected"),
            subtitle=_(
                "No profile matched it automatically. Pick the "
                "matching profile below, or continue with "
                "'Device Not Listed'."
            ),
        )
        self.hint_row.set_visible(False)
        self.content.append(self.hint_row)

        group = _makePreferencesGroup(
            title=_("Machine Templates"),
            description=_(
                "Pick a built-in profile to pre-fill common "
                "settings. You will still be asked for "
                "connection-specific values."
            ),
        )
        self.content.append(group)

        self.search_entry = Gtk.SearchEntry()
        self.search_entry.set_placeholder_text(_("Search devices…"))
        self.search_entry.connect(
            "search-changed", lambda *_: self._filter_and_populate_list()
        )
        self.content.append(self.search_entry)

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.add_css_class("frame")
        self.list_box.add_css_class("card")
        self.list_box.connect("row-activated", self._on_row_activated)
        self.content.append(self.list_box)

        # Import / Other affordances live on the wizard footer button
        # bar (see footer_buttons()).
        self.import_button = Gtk.Button(label=_("Import from File…"))
        self.import_button.connect("clicked", self._on_import_clicked)

        self.other_button = Gtk.Button(label=_("Device Not Listed"))
        self.other_button.add_css_class("suggested-action")
        self.other_button.connect("clicked", self._on_other_clicked)

        # The page is always ready because the user can always go to
        # "Other…" or pick a profile — row activation emits
        # source_selected directly and the orchestrator moves on.
        self.set_ready(True)

    def footer_buttons(self) -> list[Gtk.Button]:
        return [self.import_button, self.other_button]

    def enter(self, profile: DeviceProfile) -> None:
        discovered = self.wizard.aux_state.get("discovered")
        self.hint_row.set_visible(bool(discovered))
        self._suggested_profiles = (
            self._compute_suggestions(discovered)
            if isinstance(discovered, DiscoveredDevice)
            else []
        )
        self._all_profiles = list(get_context().device_profile_mgr.get_all())
        self._filter_and_populate_list()

    def _compute_suggestions(
        self, discovered: DiscoveredDevice
    ) -> list[DeviceProfile]:
        """Profiles matching what the collected identity says about
        the machine: same vendor tokens and same controller."""
        name = discovered.probe_name
        tokens = set(discovered.identity.tokens)
        if name:
            tokens |= normalize_tokens(name)
        suggested = []
        for pkg in get_context().device_profile_mgr.get_all():
            vendor_tokens = normalize_tokens(pkg.meta.vendor or "")
            if not vendor_tokens or not vendor_tokens <= tokens:
                continue
            if pkg.machine_config.driver != discovered.driver_name:
                continue
            suggested.append(pkg)
        return suggested

    def _filter_and_populate_list(self) -> None:
        search_text = self.search_entry.get_text().lower()
        # With a pending discovered device and no search active, the
        # list shows only the relevant profiles; typing a search
        # broadens it back to the full catalog.
        pool = self._all_profiles
        if self._suggested_profiles and not search_text:
            pool = self._suggested_profiles

        while child := self.list_box.get_row_at_index(0):
            self.list_box.remove(child)

        for pkg in pool:
            if (
                search_text
                and search_text not in pkg.name.lower()
                and search_text not in (pkg.meta.description or "").lower()
            ):
                continue
            row = _ProfileRow(
                profile=pkg,
                title=pkg.name,
                subtitle=pkg.meta.description or "",
                activatable=True,
            )
            self.list_box.append(row)

    # ----- selection handlers -------------------------------------------

    def _on_row_activated(self, listbox: Gtk.ListBox, row: _ProfileRow):
        self.source_selected.send(self, kind="profile", profile=row.profile)

    def _on_import_clicked(self, button: Gtk.Button):
        open_profile_file(self.wizard, self._on_import_result)

    def _on_import_result(
        self, profile: DeviceProfile | None, error: str | None
    ) -> None:
        if error is not None or profile is None:
            self.wizard.show_error(_("Import Failed"), error or "")
            return
        self.source_selected.send(self, kind="import", profile=profile)

    def _on_other_clicked(self, button: Gtk.Button):
        self.source_selected.send(self, kind="other", profile=None)


__all__ = ["ProfilePage"]
