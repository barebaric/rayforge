"""Camera wizard page: generate / print the calibration card."""

import logging
import os
from gettext import gettext as _
from typing import Optional

import cv2

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
from gi.repository import Adw, GdkPixbuf, GLib, Gtk

from ....camera.calibration.charuco import CharucoBoard
from ....context import get_context
from ....shared.units.formatter import format_value
from ...shared.unit_spin_row import UnitSpinRowHelper
from ..capture_surface import numpy_to_pixbuf
from .base_page import CameraWizardPage

logger = logging.getLogger(__name__)


class CardPage(CameraWizardPage):
    step_name = "card"
    title = _("Calibration Card")
    DEFAULT_CARD_RATIO = 0.7

    def __init__(self, wizard, controller):
        super().__init__(wizard, controller)
        self._board: Optional[CharucoBoard] = None
        self._preview_pixbuf: Optional[GdkPixbuf.Pixbuf] = None

        machine = get_context().machine
        if machine:
            unused_x, unused_y, wa_w, wa_h = machine.work_area
            self._card_width = min(100.0, wa_w * self.DEFAULT_CARD_RATIO)
            self._card_height = min(140.0, wa_h * self.DEFAULT_CARD_RATIO)
        else:
            self._card_width = 80.0
            self._card_height = 100.0

    @property
    def board(self) -> Optional[CharucoBoard]:
        return self._board

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)
        self.root.append(left_box)

        preview_frame = Gtk.Frame(
            halign=Gtk.Align.FILL,
            valign=Gtk.Align.FILL,
            hexpand=True,
            vexpand=True,
        )
        preview_frame.add_css_class("card")
        left_box.append(preview_frame)

        self.preview_image = Gtk.Picture(
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        self.preview_image.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.preview_image.set_size_request(400, 400)
        preview_frame.set_child(self.preview_image)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.root.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(4)
        settings_box.set_margin_bottom(12)
        right_scroll.set_child(settings_box)

        intro_group = Adw.PreferencesGroup(
            title=_("Instructions"),
            description=_(
                "Print a calibration card to correct lens distortion. "
                "The card size should fit within your camera view."
            ),
        )
        settings_box.append(intro_group)

        size_group = Adw.PreferencesGroup(
            title=_("Card Size"),
            description=_("Adjust to fit your work surface."),
        )
        settings_box.append(size_group)

        width_spin = Adw.SpinRow(
            title=_("Width"),
            subtitle=_("Card width"),
            adjustment=Gtk.Adjustment(
                lower=20.0,
                upper=300.0,
                step_increment=5.0,
                page_increment=20.0,
            ),
        )
        self._width_helper = UnitSpinRowHelper(
            spin_row=width_spin,
            quantity="length",
            max_value_in_base=300.0,
        )
        self._width_helper.set_value_in_base_units(self._card_width)
        self._width_helper.changed.connect(self._on_size_changed)
        size_group.add(width_spin)

        height_spin = Adw.SpinRow(
            title=_("Height"),
            subtitle=_("Card height"),
            adjustment=Gtk.Adjustment(
                lower=20.0,
                upper=300.0,
                step_increment=5.0,
                page_increment=20.0,
            ),
        )
        self._height_helper = UnitSpinRowHelper(
            spin_row=height_spin,
            quantity="length",
            max_value_in_base=300.0,
        )
        self._height_helper.set_value_in_base_units(self._card_height)
        self._height_helper.changed.connect(self._on_size_changed)
        size_group.add(height_spin)

        info_group = Adw.PreferencesGroup(
            title=_("Generated Pattern"),
            description=_("Details about the calibration pattern."),
            margin_top=12,
        )
        settings_box.append(info_group)

        self.squares_row = Adw.ActionRow(title=_("Grid Size"))
        info_group.add(self.squares_row)

        self.square_size_row = Adw.ActionRow(title=_("Square Size"))
        info_group.add(self.square_size_row)

        self._card_size_row = Adw.ActionRow(title=_("Physical Size"))
        info_group.add(self._card_size_row)

        save_pdf_row = Adw.ActionRow(
            title=_("Save to PDF"),
            subtitle=_("Export the calibration card for printing"),
        )
        save_pdf_btn = Gtk.Button(label=_("Save"), valign=Gtk.Align.CENTER)
        save_pdf_btn.connect("clicked", self._on_save_pdf)
        save_pdf_row.add_suffix(save_pdf_btn)
        save_pdf_row.set_activatable_widget(save_pdf_btn)
        info_group.add(save_pdf_row)

        self._update_card_preview()
        return self.root

    def _on_size_changed(self, helper) -> None:
        self._card_width = self._width_helper.get_value_in_base_units()
        self._card_height = self._height_helper.get_value_in_base_units()
        self._update_card_preview()

    def _update_card_preview(self) -> None:
        config = CharucoBoard.recommend_config(
            card_width_mm=self._card_width,
            card_height_mm=self._card_height,
        )
        self._board = CharucoBoard(config)

        self.squares_row.set_subtitle(
            f"{config.squares_x} x {config.squares_y} squares"
        )
        self.square_size_row.set_subtitle(
            format_value(config.square_length_mm, "length")
        )

        card_w, card_h = self._board.card_size_mm
        self._card_size_row.set_subtitle(
            f"{format_value(card_w, 'length')} x "
            f"{format_value(card_h, 'length')}"
        )

        px_per_mm = 8
        img_w = int(card_w * px_per_mm)
        img_h = int(card_h * px_per_mm)
        image = self._board.generate_image(output_size=(img_w, img_h))

        if image is not None:
            self._preview_pixbuf = numpy_to_pixbuf(image)
            self.preview_image.set_pixbuf(self._preview_pixbuf)

    def _on_save_pdf(self, button) -> None:
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Save Calibration Card"))
        dialog.set_initial_name("calibration_card.pdf")
        dialog.save(self.wizard, None, self._on_save_dialog_response)

    def _on_save_dialog_response(self, dialog, result) -> None:
        try:
            file = dialog.save_finish(result)
            if file:
                self._save_pdf(file.get_path())
        except GLib.Error:
            pass

    def _save_pdf(self, filepath: str) -> None:
        if self._board is None:
            return

        card_w_mm, card_h_mm = self._board.card_size_mm
        dpi = 300
        px_per_mm = dpi / 25.4
        img_w = int(card_w_mm * px_per_mm)
        img_h = int(card_h_mm * px_per_mm)

        image = self._board.generate_image(output_size=(img_w, img_h))
        if image is None:
            return

        page_w = card_w_mm / 25.4 * 72
        page_h = card_h_mm / 25.4 * 72

        doc = pymupdf.open()
        page = doc.new_page(width=page_w, height=page_h)

        temp_path = filepath.replace(".pdf", "_temp.png")
        cv2.imwrite(temp_path, image)

        rect = pymupdf.Rect(0, 0, page_w, page_h)
        page.insert_image(rect, filename=temp_path)

        doc.save(filepath)
        doc.close()

        os.remove(temp_path)
        logger.info(f"Calibration card saved to {filepath}")

        self.wizard.show_toast(_("Calibration card saved"))


__all__ = ["CardPage"]
