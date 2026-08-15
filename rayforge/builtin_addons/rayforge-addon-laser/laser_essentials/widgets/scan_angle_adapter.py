"""Adapter for the raster scan-angle row with direction preview."""

from typing import Any

from gi.repository import Adw, Gtk

from rayforge.core.varset import Var
from rayforge.ui_gtk.shared.direction_preview import DirectionPreview
from rayforge.ui_gtk.shared.slider import create_slider
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)

from ..scan_angle_var import ScanAngleVar


@register_adapter(ScanAngleVar)
class ScanAngleAdapter(RowAdapter):
    """A slider row whose suffix box places a :class:`DirectionPreview`
    to the left of the slider.

    The preview visualizes the scan direction and the cross-hatch
    pass; the manager feeds it sibling values (``cross_hatch``) via
    :meth:`update_from_values` after each data change.
    """

    def __init__(
        self,
        row: Adw.PreferencesRow,
        scale: Gtk.Scale,
        min_val: float,
        max_val: float,
        preview: DirectionPreview,
    ) -> None:
        super().__init__()
        self._row = row
        self._scale = scale
        self._min_val = min_val
        self._max_val = max_val
        self.preview = preview
        self._cross_hatch = False
        self._scale.connect("value-changed", self._on_value_changed)

    def _on_value_changed(self, scale: Gtk.Scale):
        self.preview.update(scale.get_value(), self._cross_hatch)
        self.changed.send(self)

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "ScanAngleAdapter"]:
        assert isinstance(var, ScanAngleVar)
        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1.0
        val = getattr(var, target_property)
        if val is None:
            val = min_val

        adj = Gtk.Adjustment(
            value=val,
            lower=min_val,
            upper=max_val,
            step_increment=0.1,
            page_increment=10,
        )
        scale = create_slider(
            adjustment=adj,
            digits=1,
            draw_value=True,
        )

        preview = DirectionPreview(val, False)
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.append(preview)
        box.append(scale)

        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        row.add_suffix(box)
        row.set_activatable_widget(scale)
        return row, cls(row, scale, min_val, max_val, preview)

    def get_value(self) -> float:
        return self._scale.get_value()

    def set_value(self, value: Any) -> None:
        self._scale.set_value(float(value))
        self.preview.update(float(self.get_value()), self._cross_hatch)

    def update_from_var(self, var: Var):
        assert isinstance(var, ScanAngleVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_tooltip_text(var.description)

    def update_from_values(self, values: dict[str, Any]) -> None:
        cross_hatch = bool(values.get("cross_hatch", False))
        if cross_hatch != self._cross_hatch:
            self._cross_hatch = cross_hatch
            self.preview.update(float(self.get_value()), cross_hatch)
