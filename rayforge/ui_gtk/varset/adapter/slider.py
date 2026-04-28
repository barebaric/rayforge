from typing import Any, Optional, Tuple

from gi.repository import Adw, Gtk

from ....core.varset import SliderFloatVar, Var
from ...shared.slider import create_slider_row
from .base import RowAdapter, escape_title


class SliderAdapter(RowAdapter):
    def __init__(
        self, scale: Gtk.Scale, min_val: float, max_val: float
    ) -> None:
        self._scale = scale
        self._min_val = min_val
        self._max_val = max_val

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SliderAdapter"]:
        assert isinstance(var, SliderFloatVar)
        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1.0
        val = getattr(var, target_property)
        if val is None:
            val = min_val

        initial_percent = 0.0
        range_size = max_val - min_val
        if range_size > 1e-9:
            initial_percent = ((val - min_val) / range_size) * 100.0

        adj = Gtk.Adjustment(
            value=initial_percent,
            lower=0.0,
            upper=100.0,
            step_increment=0.1,
            page_increment=10,
        )
        row, scale = create_slider_row(
            title=escape_title(var.label),
            subtitle=var.description if var.description else None,
            adjustment=adj,
            digits=1,
            draw_value=var.show_value,
        )
        row.set_activatable_widget(scale)
        return row, cls(scale, min_val, max_val)

    def get_value(self) -> Optional[Any]:
        percent = self._scale.get_value() / 100.0
        return self._min_val + percent * (self._max_val - self._min_val)

    def set_value(self, value: Any) -> None:
        range_size = self._max_val - self._min_val
        percent = 0.0
        if range_size > 1e-9:
            percent = ((float(value) - self._min_val) / range_size) * 100.0
        self._scale.set_value(percent)
