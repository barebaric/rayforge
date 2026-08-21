"""Dialog reviewing device-profile changes against a configured machine.

Shows one row per differing setting (machine settings and G-code
dialect), lets the user pick which profile values to apply via switch
rows, and records the review so the warning stays quiet until the
profile changes again.
"""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.device.profile import DeviceProfile
from ...machine.device.profile_diff import (
    DIALECT_SECTION,
    HEADS_SECTION,
    MACHINE_SECTION,
    SettingDiff,
    apply_diffs,
    diff_dialect_with_profile,
    diff_heads_with_profile,
    diff_machine_with_profile,
)
from ...machine.models.machine import Machine
from ..shared.patched_dialog_window import PatchedDialogWindow

logger = logging.getLogger(__name__)


def format_value(value) -> str:
    """Renders a diff value for display in a row subtitle."""
    if value is None:
        return _("not set")
    if isinstance(value, bool):
        return _("On") if value else _("Off")
    if isinstance(value, (list, tuple)):
        if not value:
            return _("empty")
        return ", ".join(format_value(v) for v in value)
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


class ProfileReviewDialog(PatchedDialogWindow):
    """
    Lists the differences between *machine* and *profile*.

    Every changed setting is an :class:`Adw.SwitchRow` grouped into
    Machine Settings and G-code Dialect sections; each switch selects
    whether the profile value overwrites the device setting. Closing
    the dialog in any way marks the profile as reviewed so the startup
    check does not re-trigger until the profile changes again.
    """

    def __init__(
        self,
        machine: Machine,
        profile: DeviceProfile,
        transient_for=None,
        on_closed=None,
        **kwargs,
    ):
        super().__init__(skip_usage_tracking=True, **kwargs)
        if transient_for:
            self.set_transient_for(transient_for)
        self.machine = machine
        self.profile = profile
        self._on_closed = on_closed
        self._finished = False

        self.set_title(
            _("{machine} - Profile Updated").format(machine=machine.name)
        )
        self.set_default_size(600, 520)

        # --- Layout ---
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        ignore_button = Gtk.Button(label=_("Ignore"))
        ignore_button.connect("clicked", self._on_ignore_clicked)
        header_bar.pack_start(ignore_button)

        apply_button = Gtk.Button(label=_("Apply Selected"))
        apply_button.add_css_class("suggested-action")
        apply_button.connect("clicked", self._on_apply_clicked)
        header_bar.pack_end(apply_button)

        diffs = (
            diff_machine_with_profile(machine, profile)
            + diff_heads_with_profile(machine, profile)
            + diff_dialect_with_profile(machine, profile)
        )

        # The preferences page provides the scrolling and margins used
        # by the rest of the app's preference dialogs.
        page = Adw.PreferencesPage()
        toolbar_view.set_content(page)

        intro = _(
            "The machine “{machine}” was created from the device "
            "profile “{profile}”, which has changed since its "
            "last review. Turn on the settings you want to take "
            "over from the profile."
        ).format(machine=machine.name, profile=profile.name)

        self._rows: list[tuple[SettingDiff, Adw.SwitchRow]] = []
        first_group: Adw.PreferencesGroup | None = None
        for section in (MACHINE_SECTION, HEADS_SECTION, DIALECT_SECTION):
            section_diffs = [d for d in diffs if d.section == section]
            if not section_diffs:
                continue
            group = self._build_section(section, section_diffs)
            # Fold the intro into the first group's description; an
            # extra empty group would just add dead space above.
            if first_group is None:
                first_group = group
            page.add(group)

        if first_group is not None:
            first_group.set_description(intro)
        else:
            page.add(
                Adw.PreferencesGroup(
                    title=_("No Differences"),
                    description=intro + " " + _("No differences found."),
                )
            )

    # ----- UI construction -------------------------------------------------

    def _build_section(
        self, title: str, diffs: list[SettingDiff]
    ) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup(title=title)
        for diff in diffs:
            row = Adw.SwitchRow(
                title=diff.path,
                subtitle=_("{current}  →  {profile}").format(
                    current=format_value(diff.current_value),
                    profile=format_value(diff.profile_value),
                ),
                active=True,
            )
            group.add(row)
            self._rows.append((diff, row))
        return group

    # ----- actions ---------------------------------------------------------

    def _selected_diffs(self) -> list[SettingDiff]:
        return [diff for diff, row in self._rows if row.get_active()]

    def _on_apply_clicked(self, button):
        apply_diffs(self.machine, self.profile, self._selected_diffs())
        self._finish()

    def _on_ignore_clicked(self, button):
        self._finish()

    def do_close_request(self, *args) -> bool:
        # Treat window close (X, Escape) like Ignore so the review is
        # recorded either way.
        self._finish()
        return False

    def _finish(self):
        """Marks the profile reviewed and closes."""
        if self._finished:
            return
        self._finished = True
        self.machine.reviewed_profile_hash = self.profile.content_hash()
        self.machine.changed.send(self.machine)
        self.close()
        if self._on_closed is not None:
            self._on_closed()
