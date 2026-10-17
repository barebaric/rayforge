"""Step 0 — OS permission pre-flight (conditional).

Shown as the wizard's first page only when hardware permission checks
fail: serial ports or cameras are present on the system but the
current user cannot open them. Each detected problem gets its own
section with an explanation and the exact remediation commands.

Every command is rendered as a single-click copy row: activating the
row (clicking anywhere on it) copies the command to the clipboard and
flashes the copy icon into a check mark for feedback.

The page never blocks: "Next" stays enabled so users who do not need
the affected hardware (e.g. G-code export only, no camera) can carry
on regardless.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ....machine.device.profile import DeviceProfile
from ....shared.util.permissions import (
    PermissionIssue,
    check_permissions,
)
from ...icons import get_icon
from ...shared.command_row import CommandRow
from . import WizardPage, _makePreferencesGroup

if TYPE_CHECKING:
    from ..unified_wizard import UnifiedWizard


class PermissionsPage(WizardPage):
    step_number = 0
    title = _("Set Up Permissions")
    subtitle = _(
        "Grant Rayforge access to serial ports and cameras before continuing."
    )
    next_label = _("Continue")

    def __init__(self, wizard: "UnifiedWizard", **kwargs: Any) -> None:
        self._sections: list[Gtk.Widget] = []
        self._command_rows: list[CommandRow] = []
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.intro_label = Gtk.Label(
            wrap=True,
            xalign=0.0,
            hexpand=True,
        )
        self.intro_label.add_css_class("dim-label")
        self.content.append(self.intro_label)

        self._status_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
        )
        self.content.append(self._status_box)

        # Clicking any command row copies it; there is nothing to
        # validate, so Next is always available.
        self.set_ready(True)

    def enter(self, profile: DeviceProfile) -> None:
        """Re-run the checks every time the page is shown."""
        self._rebuild(check_permissions())

    def footer_buttons(self) -> list[Gtk.Button]:
        if not hasattr(self, "_recheck_button"):
            self._recheck_button = Gtk.Button(label=_("Recheck"))
            self._recheck_button.add_css_class("flat")
            self._recheck_button.connect(
                "clicked", lambda *_: self.enter(self.wizard.profile)
            )
        return [self._recheck_button]

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        return True

    # ----- rendering -------------------------------------------------

    def _rebuild(self, issues: list[PermissionIssue]) -> None:
        for section in self._sections:
            self._status_box.remove(section)
        self._sections.clear()
        self._command_rows.clear()

        if not issues:
            self.intro_label.set_text(_("All hardware permissions look good."))
            ok_group = _makePreferencesGroup()
            ok_row = Adw.ActionRow(
                title=_("Permissions are properly configured"),
                subtitle=_("You can continue with device discovery."),
            )
            icon = get_icon("check-circle-symbolic")
            icon.set_valign(Gtk.Align.CENTER)
            ok_row.add_prefix(icon)
            ok_group.add(ok_row)
            self._sections.append(ok_group)
        else:
            self.intro_label.set_text(
                _(
                    "Some hardware permissions are missing. Run the "
                    "commands below in a terminal (click a command to "
                    "copy it), then press Recheck. You can also "
                    "continue without fixing these now."
                )
            )
            for issue in issues:
                self._sections.append(self._build_issue_section(issue))

        for section in self._sections:
            self._status_box.append(section)

    def _build_issue_section(self, issue: PermissionIssue) -> Gtk.Widget:
        """One issue as a titled command list, plus an optional plain
        note below it."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        group = _makePreferencesGroup(
            title=issue.title,
            description=issue.summary,
        )
        for command in issue.commands:
            row = CommandRow(command)
            group.add(row)
            self._command_rows.append(row)
        box.append(group)
        if issue.note:
            note = Gtk.Label(label=issue.note, wrap=True, xalign=0.0)
            note.add_css_class("dim-label")
            note.set_margin_top(6)
            box.append(note)
        return box


__all__ = ["PermissionsPage"]
