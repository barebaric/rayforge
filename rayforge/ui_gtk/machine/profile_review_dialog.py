"""Dialogs reviewing setting changes against a configured machine.

Two variants share the same switch-row UI:

* :class:`ProfileReviewDialog` — shown when a machine's source device
  profile changed since its last review. Stamps
  ``reviewed_profile_hash`` on close.

* :class:`SchemaReviewDialog` — shown when a machine was loaded from an
  older app version that predates new head/machine settings (detected
  via ``machine.schema_version``). Stamps ``schema_version`` on close.

Both let the user pick which new values to apply via switch rows and
use the same :class:`SettingDiff` infrastructure.
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
from ...machine.device.schema_migration import (
    CURRENT_SCHEMA_VERSION,
    apply_schema_migrations,
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


class _ReviewDialogBase(PatchedDialogWindow):
    """Shared switch-row layout for both profile and schema reviews.

    Subclasses provide ``_build_diffs()``, ``_apply()``, and
    ``_finish()``.
    """

    def __init__(
        self,
        machine: Machine,
        diffs: list[SettingDiff],
        title: str,
        intro: str,
        transient_for=None,
        on_closed=None,
        **kwargs,
    ):
        super().__init__(skip_usage_tracking=True, **kwargs)
        if transient_for:
            self.set_transient_for(transient_for)
        self.machine = machine
        self._on_closed = on_closed
        self._finished = False

        self.set_title(title)
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

        page = Adw.PreferencesPage()
        toolbar_view.set_content(page)

        self._rows: list[tuple[SettingDiff, Adw.SwitchRow]] = []
        first_group: Adw.PreferencesGroup | None = None
        for section in (MACHINE_SECTION, HEADS_SECTION, DIALECT_SECTION):
            section_diffs = [d for d in diffs if d.section == section]
            if not section_diffs:
                continue
            group = self._build_section(section, section_diffs)
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
        self._apply(self._selected_diffs())
        self._finish()

    def _on_ignore_clicked(self, button):
        self._finish()

    def do_close_request(self, *args) -> bool:
        self._finish()
        return False

    # ----- subclass hooks --------------------------------------------------

    def _apply(self, diffs: list[SettingDiff]) -> None:
        raise NotImplementedError

    def _finish(self) -> None:
        raise NotImplementedError


class ProfileReviewDialog(_ReviewDialogBase):
    """Shows settings where a machine's source device profile changed
    since the last review. Closing marks the profile reviewed."""

    def __init__(
        self,
        machine: Machine,
        profile: DeviceProfile,
        transient_for=None,
        on_closed=None,
        **kwargs,
    ):
        self.profile = profile
        diffs = (
            diff_machine_with_profile(machine, profile)
            + diff_heads_with_profile(machine, profile)
            + diff_dialect_with_profile(machine, profile)
        )
        title = _("{machine} - Profile Updated").format(machine=machine.name)
        intro = _(
            "The machine “{machine}” was created from the device "
            "profile “{profile}”, which has changed since its "
            "last review. Turn on the settings you want to take "
            "over from the profile."
        ).format(machine=machine.name, profile=profile.name)
        super().__init__(
            machine=machine,
            diffs=diffs,
            title=title,
            intro=intro,
            transient_for=transient_for,
            on_closed=on_closed,
            **kwargs,
        )

    def _apply(self, diffs: list[SettingDiff]) -> None:
        apply_diffs(self.machine, self.profile, diffs)

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.machine.reviewed_profile_hash = self.profile.content_hash()
        self.machine.changed.send(self.machine)
        self.close()
        if self._on_closed is not None:
            self._on_closed()


class SchemaReviewDialog(_ReviewDialogBase):
    """Shows new settings a machine is missing because it was saved by
    an older app version. Closing stamps the current schema version."""

    def __init__(
        self,
        machine: Machine,
        diffs: list[SettingDiff],
        transient_for=None,
        on_closed=None,
        **kwargs,
    ):
        title = _("{machine} - New Settings Available").format(
            machine=machine.name
        )
        intro = _(
            "New configuration options are available for the machine "
            "“{machine}”. Turn on the settings you want to apply."
        ).format(machine=machine.name)
        super().__init__(
            machine=machine,
            diffs=diffs,
            title=title,
            intro=intro,
            transient_for=transient_for,
            on_closed=on_closed,
            **kwargs,
        )

    def _apply(self, diffs: list[SettingDiff]) -> None:
        apply_schema_migrations(self.machine, diffs)

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        # Stamp the version even when the user ignores, so the dialog
        # does not reappear on every startup. When applying, this is a
        # no-op (apply_schema_migrations already stamped).
        self.machine.schema_version = CURRENT_SCHEMA_VERSION
        self.machine.changed.send(self.machine)
        self.close()
        if self._on_closed is not None:
            self._on_closed()
