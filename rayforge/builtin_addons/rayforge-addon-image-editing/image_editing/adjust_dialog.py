"""The Adjust Image dialog: tone and sharpness of the selected images."""

from collections.abc import Sequence
from gettext import gettext as _

from gi.repository import Adw, Gtk

from rayforge.core.undo import HistoryManager
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.shared.patched_dialog_window import (
    PatchedDialogWindow,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpinRow

from .adjustments import ImageAdjustments
from .commands import replace_op_edits, set_op_edits

#: Delay before an edited value updates the canvas preview.
PREVIEW_DELAY_MS = 250


class AdjustImageDialog(PatchedDialogWindow):
    """
    Edits the "sharpen" and "tone" edits of the selected workpieces.
    Changes show on the canvas right away; Apply makes them one undo
    step and closing the dialog without applying restores the images.
    """

    def __init__(
        self,
        parent: Gtk.Window | None,
        history: HistoryManager,
        workpieces: Sequence[WorkPiece],
    ):
        super().__init__(transient_for=parent)
        self._history = history
        self._workpieces = list(workpieces)
        self._originals = {wp.uid: wp.image_edits for wp in self._workpieces}
        self._applied = False

        self.set_title(_("Adjust Image"))
        self.set_default_size(440, -1)
        self.set_modal(True)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.append(self._build_header())
        box.append(self._build_page())
        self.set_content(box)

        first = self._workpieces[0].image_edits if self._workpieces else ()
        self.set_adjustments(ImageAdjustments.from_edits(first))
        self.connect("close-request", self._on_close_request)

    def _build_header(self) -> Adw.HeaderBar:
        header = Adw.HeaderBar()
        reset_btn = Gtk.Button(label=_("_Reset"), use_underline=True)
        reset_btn.connect("clicked", lambda *a: self.reset())
        header.pack_start(reset_btn)
        apply_btn = Gtk.Button(label=_("_Apply"), use_underline=True)
        apply_btn.add_css_class("suggested-action")
        apply_btn.connect("clicked", lambda *a: self.apply())
        header.pack_end(apply_btn)
        return header

    def _spin_row(self, title, subtitle, lower, upper, step, digits):
        row = SpinRow(
            title,
            subtitle,
            lower=lower,
            upper=upper,
            step_increment=step,
            digits=digits,
            numeric=True,
            debounce_ms=PREVIEW_DELAY_MS,
        )
        row.value_changed.connect(lambda *a: self.preview())
        return row

    def _build_page(self) -> Adw.PreferencesPage:
        self.brightness_row = self._spin_row(
            _("Brightness"), _("Lighten or darken the image"), -100, 100, 1, 0
        )
        self.contrast_row = self._spin_row(
            _("Contrast"),
            _("Spread or compress the tones around mid-gray"),
            -100,
            100,
            1,
            0,
        )
        self.gamma_row = self._spin_row(
            _("Gamma"),
            _("Values above 1 lighten the midtones, below 1 darken them"),
            0.2,
            5.0,
            0.05,
            2,
        )
        self.sharpen_row = self._spin_row(
            _("Sharpen"),
            _("Strength of the unsharp mask, in percent"),
            0,
            300,
            5,
            0,
        )
        self.radius_row = LengthSpinRow(
            _("Sharpen Radius"),
            _("Size of the details that get sharpened"),
            lower=0.01,
            upper=5.0,
            step_increment=0.05,
            digits=3,
            debounce_ms=PREVIEW_DELAY_MS,
        )
        self.radius_row.value_changed.connect(lambda *a: self.preview())

        tone = Adw.PreferencesGroup(title=_("Tone"))
        for row in (self.brightness_row, self.contrast_row, self.gamma_row):
            tone.add(row)
        sharpness = Adw.PreferencesGroup(title=_("Sharpness"))
        sharpness.add(self.sharpen_row)
        sharpness.add(self.radius_row)
        page = Adw.PreferencesPage()
        page.add(tone)
        page.add(sharpness)
        return page

    def get_adjustments(self) -> ImageAdjustments:
        return ImageAdjustments(
            brightness=self.brightness_row.get_value(),
            contrast=self.contrast_row.get_value(),
            gamma=self.gamma_row.get_value(),
            sharpen_amount=self.sharpen_row.get_value(),
            sharpen_radius_mm=self.radius_row.get_value_in_base_units(),
        )

    def set_adjustments(self, values: ImageAdjustments) -> None:
        self.brightness_row.set_value(values.brightness)
        self.contrast_row.set_value(values.contrast)
        self.gamma_row.set_value(values.gamma)
        self.sharpen_row.set_value(values.sharpen_amount)
        self.radius_row.set_value_in_base_units(values.sharpen_radius_mm)

    def preview(self) -> None:
        """Shows the current values on the canvas, without undo."""
        edits = self.get_adjustments().to_edits()
        for wp in self._workpieces:
            wp.image_edits = replace_op_edits(self._originals[wp.uid], edits)

    def reset(self) -> None:
        self.set_adjustments(ImageAdjustments())
        self.preview()

    def apply(self) -> None:
        set_op_edits(
            self._history,
            self._workpieces,
            self.get_adjustments().to_edits(),
            _("Adjust Image"),
            self._originals,
        )
        self._applied = True
        self.close()

    def _on_close_request(self, *args) -> bool:
        if not self._applied:
            for wp in self._workpieces:
                wp.image_edits = self._originals[wp.uid]
        return False
