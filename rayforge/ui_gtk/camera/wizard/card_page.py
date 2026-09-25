"""Camera wizard page: choose and print the calibration pattern."""

import logging
import os
from gettext import gettext as _

import cv2

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
from gi.repository import Adw, Gdk, GdkPixbuf, GLib, Gtk

from ....camera.calibration import (
    available_target_types,
    create_target,
    recommend_target,
)
from ....camera.calibration.target import (
    CalibrationTarget,
    CalibrationTargetType,
    ConfigField,
    TargetConfig,
)
from ....context import get_context
from ....shared.units.formatter import format_value
from ...shared.pref_rows.base import SpinRow
from ...shared.pref_rows.length_spin_row import LengthSpinRow
from ..capture_surface import numpy_to_pixbuf
from .base_page import CameraWizardPage

logger = logging.getLogger(__name__)

TARGET_TYPE_LABELS: dict[CalibrationTargetType, str] = {
    CalibrationTargetType.CHARUCO: _("ChArUco Board"),
    CalibrationTargetType.ARUCO_GRID: _("ArUco Marker Grid"),
    CalibrationTargetType.DOT_GRID: _("Dot Grid"),
}

# The calibration targets describe themselves with stable keys rather
# than display text, so the translatable strings live here where the
# message catalogue can pick them up.
FIELD_LABELS: dict[str, str] = {
    "squares_x": _("Columns"),
    "squares_y": _("Rows"),
    "markers_x": _("Columns"),
    "markers_y": _("Rows"),
    "dots_x": _("Columns"),
    "dots_y": _("Rows"),
    "square_length_mm": _("Square Size"),
    "marker_length_mm": _("Marker Size"),
    "marker_separation_mm": _("Marker Gap"),
    "spacing_mm": _("Dot Spacing"),
    "row_spacing_mm": _("Row Spacing"),
    "row_offset_mm": _("Row Offset"),
    "dot_diameter_mm": _("Dot Diameter"),
}

SUMMARY_LABELS: dict[str, str] = {
    "grid_size": _("Grid Size"),
    "square_size": _("Square Size"),
    "marker_size": _("Marker Size"),
    "marker_gap": _("Marker Gap"),
    "dot_spacing": _("Dot Spacing"),
    "row_spacing": _("Row Spacing"),
    "dot_diameter": _("Dot Diameter"),
    "physical_size": _("Physical Size"),
}

TARGET_TYPE_DESCRIPTIONS: dict[CalibrationTargetType, str] = {
    CalibrationTargetType.CHARUCO: (
        "Chessboard with markers. Most accurate, needs a good printer."
    ),
    CalibrationTargetType.ARUCO_GRID: (
        "Standalone markers. Tolerates partial views and clutter."
    ),
    CalibrationTargetType.DOT_GRID: (
        "Plain black dots. Cheapest to print, lowest accuracy."
    ),
}


class CardPage(CameraWizardPage):
    step_name = "card"
    title = _("Calibration Card")
    DEFAULT_CARD_RATIO = 0.7
    PREVIEW_PX_PER_MM = 8

    def __init__(self, wizard, controller):
        super().__init__(wizard, controller)
        self._target: CalibrationTarget | None = None
        self._config: TargetConfig | None = None
        self._target_type: CalibrationTargetType = (
            CalibrationTargetType.CHARUCO
        )
        self._field_rows: dict[str, object] = {}
        self._summary_rows: dict[str, Adw.ActionRow] = {}
        self._customized = False
        self._preview_pixbuf: GdkPixbuf.Pixbuf | None = None

        machine = get_context().machine
        if machine:
            _unused_x, _unused_y, wa_w, wa_h = machine.work_area
            self._card_width = min(100.0, wa_w * self.DEFAULT_CARD_RATIO)
            self._card_height = min(140.0, wa_h * self.DEFAULT_CARD_RATIO)
        else:
            self._card_width = 80.0
            self._card_height = 100.0

    @property
    def target(self) -> CalibrationTarget | None:
        return self._target

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

        settings_box.append(
            Adw.PreferencesGroup(
                title=_("Instructions"),
                description=_(
                    "Print a calibration pattern to correct lens "
                    "distortion. The pattern should fit within your camera "
                    "view. You can also describe a pattern you already "
                    "have and calibrate against that instead."
                ),
            )
        )

        self._type_row = Adw.ComboRow(
            title=_("Pattern Type"),
            subtitle=_("ChArUco Board"),
        )
        model = Gtk.StringList()
        for target_type in available_target_types():
            model.append(TARGET_TYPE_LABELS[target_type])
        self._type_row.set_model(model)
        self._type_row.set_selected(0)
        self._type_row.connect("notify::selected", self._on_type_changed)
        type_group = Adw.PreferencesGroup(
            title=_("Pattern"),
            description=_("Choose the printed pattern to calibrate against."),
        )
        type_group.add(self._type_row)
        settings_box.append(type_group)

        self._geometry_group = Adw.PreferencesGroup(
            title=_("Pattern Geometry"),
            description=_(
                "Measured from the printed sheet. Editing these lets you "
                "use a pattern you printed earlier."
            ),
        )
        settings_box.append(self._geometry_group)

        size_group = Adw.PreferencesGroup(
            title=_("Card Size"),
            description=_(
                "Used to suggest a pattern that fits. Ignored once the "
                "geometry above is edited."
            ),
        )
        settings_box.append(size_group)

        self._width_row = LengthSpinRow(
            _("Width"),
            _("Card width"),
            lower=20.0,
            upper=300.0,
            value_in_base=self._card_width,
        )
        self._width_row.value_changed.connect(self._on_size_changed)
        size_group.add(self._width_row)

        self._height_row = LengthSpinRow(
            _("Height"),
            _("Card height"),
            lower=20.0,
            upper=300.0,
            value_in_base=self._card_height,
        )
        self._height_row.value_changed.connect(self._on_size_changed)
        size_group.add(self._height_row)

        self._summary_group = Adw.PreferencesGroup(
            title=_("Pattern Details"),
            description=_("Details about the calibration pattern."),
        )
        settings_box.append(self._summary_group)

        info_group = Adw.PreferencesGroup(
            title=_("Output"),
            margin_top=12,
        )
        settings_box.append(info_group)

        self._card_size_row = Adw.ActionRow(
            title=SUMMARY_LABELS["physical_size"]
        )
        info_group.add(self._card_size_row)

        save_pdf_row = Adw.ActionRow(
            title=_("Save to PDF"),
            subtitle=_("Export the calibration pattern for printing"),
        )
        save_pdf_btn = Gtk.Button(label=_("Save"), valign=Gtk.Align.CENTER)
        save_pdf_btn.connect("clicked", self._on_save_pdf)
        save_pdf_row.add_suffix(save_pdf_btn)
        save_pdf_row.set_activatable_widget(save_pdf_btn)
        info_group.add(save_pdf_row)

        self._rebuild_from_card_size()
        return self.root

    def _on_type_changed(self, row, _pspec) -> None:
        index = row.get_selected()
        types = available_target_types()
        if index < 0 or index >= len(types):
            return
        self._target_type = types[index]
        self._customized = False
        self._rebuild_from_card_size()

    def _on_size_changed(self, _row) -> None:
        self._card_width = self._width_row.get_value_in_base_units()
        self._card_height = self._height_row.get_value_in_base_units()
        if self._customized:
            return
        self._rebuild_from_card_size()

    def _on_field_changed(self, _row) -> None:
        self._customized = True
        self._rebuild_target()

    def _rebuild_from_card_size(self) -> None:
        """Derive a pattern that fits the requested card size."""
        self._config = recommend_target(
            self._target_type,
            card_width_mm=self._card_width,
            card_height_mm=self._card_height,
        ).config
        self._rebuild_geometry_rows()
        self._rebuild_target()

    def _rebuild_geometry_rows(self) -> None:
        if self._config is None:
            return
        self._clear_group(self._geometry_group)
        self._field_rows.clear()

        for field in type(self._config).FIELDS:
            row = self._build_field_row(field)
            if row is not None:
                self._field_rows[field.key] = row
                self._geometry_group.add(row)

    def _build_field_row(self, field: ConfigField):
        assert self._config is not None
        value = getattr(self._config, field.key, None)
        if value is None:
            return None
        title = FIELD_LABELS.get(field.key, field.key)
        if field.integer:
            row = SpinRow(
                title,
                lower=float(field.lower),
                upper=float(field.upper),
                step_increment=1.0,
                digits=0,
                numeric=True,
                value=float(value),
            )
        else:
            row = LengthSpinRow(
                title,
                lower=field.lower,
                upper=field.upper,
                step_increment=field.step,
                digits=2,
                value_in_base=float(value),
            )
        row.value_changed.connect(self._on_field_changed)
        return row

    def _read_field_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for key, row in self._field_rows.items():
            if isinstance(row, LengthSpinRow):
                values[key] = row.get_value_in_base_units()
            else:
                values[key] = float(row.get_value())  # type: ignore[attr-defined]
        return values

    def _rebuild_target(self) -> None:
        """Build the target from the current geometry and refresh the UI."""
        if self._config is None:
            return
        try:
            self._config.apply_field_values(self._read_field_values())
        except (TypeError, ValueError) as error:
            logger.warning("Invalid calibration target geometry: %s", error)
            return

        try:
            self._target = create_target(
                self._target_type, self._config.to_dict()
            )
        except (ValueError, TypeError) as error:
            logger.warning("Could not build calibration target: %s", error)
            return

        self._refresh_summary()
        self._refresh_preview()

    def _refresh_summary(self) -> None:
        if self._target is None:
            return
        rows = self._target.summary()

        for row in list(self._summary_rows.values()):
            self._summary_group.remove(row)
        self._summary_rows.clear()

        for entry in rows:
            label = SUMMARY_LABELS.get(entry.key, entry.key)
            row = Adw.ActionRow(title=label)
            if entry.measurement_mm is not None:
                row.set_subtitle(format_value(entry.measurement_mm, "length"))
            elif entry.value is not None:
                row.set_subtitle(entry.value)
            self._summary_rows[entry.key] = row
            self._summary_group.add(row)

        card_w, card_h = self._target.card_size_mm
        self._card_size_row.set_subtitle(
            f"{format_value(card_w, 'length')} x "
            f"{format_value(card_h, 'length')}"
        )

    def _refresh_preview(self) -> None:
        if self._target is None:
            return
        card_w, card_h = self._target.card_size_mm
        px_per_mm = self.PREVIEW_PX_PER_MM
        image = self._target.generate_image(
            output_size=(
                max(1, int(card_w * px_per_mm)),
                max(1, int(card_h * px_per_mm)),
            )
        )
        if image is None:
            return
        pixbuf = numpy_to_pixbuf(image)
        if pixbuf is None:
            return
        self._preview_pixbuf = pixbuf
        texture = Gdk.Texture.new_for_pixbuf(pixbuf)
        self.preview_image.set_paintable(texture)

    def _clear_group(self, group: Adw.PreferencesGroup) -> None:
        child = group.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            group.remove(child)
            child = nxt

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
        if self._target is None:
            return

        card_w_mm, card_h_mm = self._target.card_size_mm
        dpi = 300
        px_per_mm = dpi / 25.4
        img_w = int(card_w_mm * px_per_mm)
        img_h = int(card_h_mm * px_per_mm)

        image = self._target.generate_image(output_size=(img_w, img_h))
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
        logger.info(f"Calibration pattern saved to {filepath}")

        self.wizard.show_toast(_("Calibration card saved"))


__all__ = ["CardPage"]
