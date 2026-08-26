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

from gi.repository import Adw, GLib, Gtk

from ....machine.device.profile import DeviceProfile
from ....shared.util.permissions import (
    PermissionIssue,
    check_permissions,
)
from ...icons import get_icon
from . import WizardPage, _makePreferencesGroup

if TYPE_CHECKING:
    from ..unified_wizard import UnifiedWizard

_COPY_FEEDBACK_MS = 2000


class _CommandRow(Adw.ActionRow):
    """A terminal command that copies itself on a single click.

    The command text is the row title (monospace via Pango markup);
    a copy icon sits in the suffix slot and flips to a check mark
    briefly after copying.
    """

    def __init__(self, command: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._command = command
        self._feedback_source_id: int | None = None

        escaped = GLib.markup_escape_text(command, -1)
        self.set_use_markup(True)
        self.set_title(f'<span font_family="monospace">{escaped}</span>')
        self.set_activatable(True)

        self._icon = get_icon("copy-symbolic")
        self._icon.set_valign(Gtk.Align.CENTER)
        self.add_suffix(self._icon)

        self.connect("activated", self._on_activated)
        self.connect("destroy", self._on_destroy)

    def _on_activated(self, _row: Adw.ActionRow) -> None:
        display = self.get_display()
        if display is None:
            return
        display.get_clipboard().set(self._command)
        self._icon.set_from_icon_name("check-symbolic")
        if self._feedback_source_id is not None:
            GLib.source_remove(self._feedback_source_id)
        self._feedback_source_id = GLib.timeout_add(
            _COPY_FEEDBACK_MS, self._restore_icon
        )

    def _restore_icon(self) -> bool:
        self._feedback_source_id = None
        self._icon.set_from_icon_name("copy-symbolic")
        return GLib.SOURCE_REMOVE

    def _on_destroy(self, _widget: Gtk.Widget) -> None:
        if self._feedback_source_id is not None:
            GLib.source_remove(self._feedback_source_id)
            self._feedback_source_id = None


class PermissionsPage(WizardPage):
    step_number = 0
    title = _("Set Up Permissions")
    subtitle = _(
        "Grant Rayforge access to serial ports and cameras before continuing."
    )
    next_label = _("Continue")

    def __init__(self, wizard: "UnifiedWizard", **kwargs: Any) -> None:
        self._sections: list[Gtk.Widget] = []
        self._command_rows: list[_CommandRow] = []
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
            row = _CommandRow(command)
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
