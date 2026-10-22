"""The Barcode group in the properties panel."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from rayforge.core.item import DocItem
from rayforge.ui_gtk.doceditor.property_providers.base import (
    PropertyProvider,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpinRow

from .asset import BarcodeAsset
from .commands import barcode_of, set_barcode_setting
from .encoders import QrErrorCorrection, Symbology

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

SYMBOLOGY_LABELS = {
    Symbology.QR: _("QR Code"),
    Symbology.CODE128: _("Code 128"),
    Symbology.EAN13: _("EAN-13"),
}
ERROR_LABELS = {
    QrErrorCorrection.L: _("Low (7 %)"),
    QrErrorCorrection.M: _("Medium (15 %)"),
    QrErrorCorrection.Q: _("Quartile (25 %)"),
    QrErrorCorrection.H: _("High (30 %)"),
}
SYMBOLOGIES = list(SYMBOLOGY_LABELS)
ERROR_LEVELS = list(ERROR_LABELS)


def _combo(title: str, labels: list[str]) -> Adw.ComboRow:
    return Adw.ComboRow(title=title, model=Gtk.StringList.new(labels))


class BarcodePropertyProvider(PropertyProvider):
    """Edits the content, type and size of a selected barcode."""

    priority = 25
    separate_group = True
    group_title = _("Barcode")

    def can_handle(self, items: list[DocItem]) -> bool:
        return len(items) == 1 and barcode_of(items[0]) is not None

    def create_widgets(self) -> list[Gtk.Widget]:
        self.content_row = Adw.EntryRow(
            title=_("Content"), show_apply_button=True
        )
        self.content_row.connect("apply", self._on_content_applied)
        self.symbology_row = _combo(
            _("Type"), [SYMBOLOGY_LABELS[s] for s in SYMBOLOGIES]
        )
        self.error_row = _combo(
            _("Error Correction"), [ERROR_LABELS[e] for e in ERROR_LEVELS]
        )
        self.module_row = LengthSpinRow(
            _("Module Size"),
            _("Width of the narrowest bar or one QR square"),
            lower=0.05,
            upper=20.0,
            step_increment=0.05,
            digits=2,
        )
        self.bar_height_row = LengthSpinRow(
            _("Bar Height"),
            _("Height of the bars"),
            lower=1.0,
            upper=500.0,
            step_increment=0.5,
            digits=1,
        )
        self.quiet_zone_row = SpinRow(
            _("Quiet Zone"),
            _("Dark border around an inverted code, in modules"),
            lower=0,
            upper=100,
            numeric=True,
        )
        self.text_row = Adw.SwitchRow(
            title=_("Show Text"),
            subtitle=_("Print the content under the bars"),
        )
        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Engrave the light parts, for dark material"),
        )
        self.symbology_row.connect("notify::selected", self._on_symbology)
        self.error_row.connect("notify::selected", self._on_error)
        self.module_row.value_changed.connect(
            lambda row: self._set("module_mm", row.get_value_in_base_units())
        )
        self.bar_height_row.value_changed.connect(
            lambda row: self._set(
                "bar_height_mm", row.get_value_in_base_units()
            )
        )
        self.quiet_zone_row.value_changed.connect(
            lambda row: self._set("quiet_zone", row.get_int_value())
        )
        self.text_row.connect(
            "notify::active",
            lambda row, _p: self._set("show_text", row.get_active()),
        )
        self.invert_row.connect(
            "notify::active",
            lambda row, _p: self._set("invert", row.get_active()),
        )
        self._rows = [
            self.content_row,
            self.symbology_row,
            self.error_row,
            self.module_row,
            self.bar_height_row,
            self.quiet_zone_row,
            self.text_row,
            self.invert_row,
        ]
        return self._rows

    @property
    def asset(self) -> BarcodeAsset | None:
        return barcode_of(self.items[0]) if self.items else None

    def update_widgets(self, editor: "DocEditor", items: list[DocItem]):
        self.editor = editor
        self.items = items
        asset = self.asset
        if asset is None:
            return
        self._in_update = True
        try:
            self.content_row.set_text(asset.content)
            self.symbology_row.set_selected(SYMBOLOGIES.index(asset.symbology))
            self.error_row.set_selected(
                ERROR_LEVELS.index(asset.error_correction)
            )
            self.module_row.set_value_in_base_units(asset.module_mm)
            self.bar_height_row.set_value_in_base_units(asset.bar_height_mm)
            self.quiet_zone_row.set_value(asset.effective_quiet_zone)
            self.text_row.set_active(asset.show_text)
            self.invert_row.set_active(asset.invert)
        finally:
            self._in_update = False
        self._update_visibility(asset)

    def _update_visibility(self, asset: BarcodeAsset) -> None:
        one_dimensional = asset.symbology is not Symbology.QR
        self.error_row.set_visible(not one_dimensional)
        self.bar_height_row.set_visible(one_dimensional)
        self.text_row.set_visible(one_dimensional)
        self.quiet_zone_row.set_sensitive(asset.invert)
        error = asset.error
        self.group_subtitle = SYMBOLOGY_LABELS[asset.symbology]
        if error:
            self.content_row.add_css_class("error")
            self.content_row.set_tooltip_text(error)
        else:
            self.content_row.remove_css_class("error")
            self.content_row.set_tooltip_text(
                _("Templates work here too, e.g. SN-{counter:04d}")
            )

    def _set(self, setting: str, value: Any) -> None:
        asset = self.asset
        if self._in_update or asset is None:
            return
        set_barcode_setting(self.editor, asset, setting, value)
        changed = self.asset
        if changed is not None:
            self._update_visibility(changed)

    def _on_content_applied(self, row: Adw.EntryRow) -> None:
        self._set("content", row.get_text())

    def _on_symbology(self, row: Adw.ComboRow, _pspec) -> None:
        self._set("symbology", SYMBOLOGIES[row.get_selected()])

    def _on_error(self, row: Adw.ComboRow, _pspec) -> None:
        self._set("error_correction", ERROR_LEVELS[row.get_selected()])
