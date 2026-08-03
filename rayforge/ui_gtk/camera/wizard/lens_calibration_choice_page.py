"""Camera wizard page: choose how to calibrate the lens.

Offers two on-page branches — automatic (Charuco capture) or manual
coefficient entry — plus a "Skip" affordance in the footer bar. The
wizard inserts the relevant follow-on page(s) after this page when a
branch is chosen, so the two methods never appear as sequential
steps.
"""

from gettext import gettext as _
from typing import List, Optional

from blinker import Signal
from gi.repository import Adw, Gtk

from .base_page import CameraWizardPage


class LensCalibrationChoicePage(CameraWizardPage):
    step_name = "lens_choice"
    title = _("Lens Calibration")

    BRANCH_SKIPPED = "skipped"
    BRANCH_AUTOMATIC = "automatic"
    BRANCH_MANUAL = "manual"

    def __init__(self, wizard, controller):
        super().__init__(wizard, controller)
        # Fired with ``branch=...`` when the user picks a branch.
        self.branch_chosen = Signal()
        self.chosen_branch: Optional[str] = None
        self._automatic_btn: Optional[Gtk.Button] = None
        self._manual_btn: Optional[Gtk.Button] = None
        self._skip_btn: Optional[Gtk.Button] = None

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        self.root.append(scrolled)

        group = Adw.PreferencesGroup(
            title=_("Lens Calibration"),
            description=_(
                "Correct lens distortion for straighter lines. "
                "Choose how to calibrate, or skip if your lens has "
                "negligible distortion."
            ),
        )
        scrolled.set_child(group)

        self._automatic_btn = Gtk.Button(
            label=_("Automatic"), valign=Gtk.Align.CENTER
        )
        self._automatic_btn.connect("clicked", self._on_branch_clicked)
        automatic_row = Adw.ActionRow(
            title=_("Automatic Calibration"),
            subtitle=_(
                "Print a calibration card and capture it at several "
                "positions. The wizard solves the distortion "
                "coefficients for you."
            ),
        )
        automatic_row.add_suffix(self._automatic_btn)
        automatic_row.set_activatable_widget(self._automatic_btn)
        group.add(automatic_row)

        self._manual_btn = Gtk.Button(
            label=_("Manual"), valign=Gtk.Align.CENTER
        )
        self._manual_btn.connect("clicked", self._on_branch_clicked)
        manual_row = Adw.ActionRow(
            title=_("Manual Calibration"),
            subtitle=_(
                "Enter the radial and tangential distortion "
                "coefficients by hand."
            ),
        )
        manual_row.add_suffix(self._manual_btn)
        manual_row.set_activatable_widget(self._manual_btn)
        group.add(manual_row)

        # Skip lives in the footer bar, not the page content.
        self._skip_btn = Gtk.Button(label=_("Skip"))
        self._skip_btn.add_css_class("flat")
        self._skip_btn.connect("clicked", self._on_skip_clicked)

        return self.root

    def footer_buttons(self) -> List[Gtk.Button]:
        return [self._skip_btn] if self._skip_btn is not None else []

    def can_proceed(self) -> bool:
        return False

    def _on_branch_clicked(self, button: Gtk.Button) -> None:
        if button is self._automatic_btn:
            self.chosen_branch = self.BRANCH_AUTOMATIC
            self.branch_chosen.send(self, branch=self.BRANCH_AUTOMATIC)
        elif button is self._manual_btn:
            self.chosen_branch = self.BRANCH_MANUAL
            self.branch_chosen.send(self, branch=self.BRANCH_MANUAL)

    def _on_skip_clicked(self, _button: Gtk.Button) -> None:
        self.chosen_branch = self.BRANCH_SKIPPED
        self.branch_chosen.send(self, branch=self.BRANCH_SKIPPED)


__all__ = ["LensCalibrationChoicePage"]
