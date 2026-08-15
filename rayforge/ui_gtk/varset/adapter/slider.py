"""Slider row adapters, one per value type.

:class:`SliderIntAdapter` renders :class:`SliderIntVar` rows: the
scale spans the var's real range and maps 1:1 to the stored integer.
:class:`SliderFloatAdapter` renders :class:`SliderFloatVar` rows; it
additionally handles percent vars (``format_suffix="%"``), which store
a fraction but show a 0-100 scale.
"""

from typing import Any

from gi.repository import Adw, Gtk

from ....core.varset import (
    FloatVar,
    IntVar,
    SliderFloatVar,
    SliderIntVar,
    Var,
)
from ...shared.slider import create_slider_row
from .base import RowAdapter, escape_title, register_adapter


class _SliderAdapterBase(RowAdapter):
    """Shared slider row state and model sync."""

    def __init__(self, row: Adw.PreferencesRow, scale: Gtk.Scale) -> None:
        super().__init__()
        self._row = row
        self._scale = scale
        self._scale.connect("value-changed", lambda s: self.changed.send(self))

    def update_from_var(self, var: Var):
        assert isinstance(var, (FloatVar, IntVar))
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_tooltip_text(var.description)


@register_adapter(SliderIntVar)
class SliderIntAdapter(_SliderAdapterBase):
    """Slider for :class:`SliderIntVar` (integers, 0 digits).

    The scale spans the var's real range and maps 1:1 to the stored
    value.
    """

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "SliderIntAdapter"]:
        assert isinstance(var, SliderIntVar)
        min_val = var.min_val if var.min_val is not None else 0
        max_val = var.max_val if var.max_val is not None else 100
        val = getattr(var, target_property)
        if val is None:
            val = min_val

        adj = Gtk.Adjustment(
            value=float(val),
            lower=float(min_val),
            upper=float(max_val),
            step_increment=1,
            page_increment=10,
        )
        row, scale = create_slider_row(
            title=escape_title(var.label),
            subtitle=var.description if var.description else None,
            adjustment=adj,
            digits=0,
            draw_value=var.show_value,
            format_suffix=var.format_suffix,
        )
        row.set_activatable_widget(scale)
        return row, cls(row, scale)

    def get_value(self) -> int:
        return round(self._scale.get_value())

    def set_value(self, value: Any) -> None:
        self._scale.set_value(float(value))


@register_adapter(SliderFloatVar)
class SliderFloatAdapter(_SliderAdapterBase):
    """Slider for :class:`SliderFloatVar` (fractional values).

    Percent vars (``format_suffix="%"``) store a fraction and show a
    0-100 scale; other float vars use their real range.
    """

    def __init__(
        self,
        row: Adw.PreferencesRow,
        scale: Gtk.Scale,
        min_val: float,
        max_val: float,
        is_percent: bool,
    ) -> None:
        super().__init__(row, scale)
        self._min_val = min_val
        self._max_val = max_val
        self._is_percent = is_percent

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "SliderFloatAdapter"]:
        assert isinstance(var, SliderFloatVar)
        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1.0
        val = getattr(var, target_property)
        if val is None:
            val = min_val

        is_percent = var.format_suffix == "%"
        if is_percent:
            initial = 0.0
            range_size = max_val - min_val
            if range_size > 1e-9:
                initial = ((val - min_val) / range_size) * 100.0
            adj = Gtk.Adjustment(
                value=initial,
                lower=0.0,
                upper=100.0,
                step_increment=0.1,
                page_increment=10,
            )
        else:
            adj = Gtk.Adjustment(
                value=float(val),
                lower=float(min_val),
                upper=float(max_val),
                step_increment=0.1,
                page_increment=1.0,
            )
        digits = var.digits if var.digits is not None else 1
        row, scale = create_slider_row(
            title=escape_title(var.label),
            subtitle=var.description if var.description else None,
            adjustment=adj,
            digits=digits,
            draw_value=var.show_value,
            format_suffix=var.format_suffix,
        )
        row.set_activatable_widget(scale)
        return row, cls(row, scale, min_val, max_val, is_percent)

    def get_value(self) -> Any | None:
        if self._is_percent:
            percent = self._scale.get_value() / 100.0
            return self._min_val + percent * (self._max_val - self._min_val)
        return self._scale.get_value()

    def set_value(self, value: Any) -> None:
        if self._is_percent:
            range_size = self._max_val - self._min_val
            percent = 0.0
            if range_size > 1e-9:
                percent = ((float(value) - self._min_val) / range_size) * 100.0
            self._scale.set_value(percent)
        else:
            self._scale.set_value(float(value))
