"""Adapter for composite tuple-valued rows (min/max ranges)."""

from typing import Any, cast

from gi.repository import Adw

from rayforge.core.varset import TupleVar, Var
from rayforge.ui_gtk.shared.pref_rows import SpinRow
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)


@register_adapter(TupleVar)
class TupleAdapter(RowAdapter):
    """Two rows editing a fixed-size tuple (e.g. a min/max range).

    The primary row edits the first component (e.g. "Min Power"), the
    extra row the second (e.g. "Max Power"). The whole tuple is stored
    under the var's single key, so the recipe keeps one entry per
    range.
    """

    def __init__(
        self,
        row: Adw.PreferencesRow,
        extra: Adw.PreferencesRow,
        spins: list[SpinRow],
    ) -> None:
        super().__init__()
        self._row = row
        self._extra = extra
        self._spins = spins
        for spin in spins:
            spin.value_changed.connect(
                lambda s: self.changed.send(self), weak=False
            )

    @classmethod
    def _build_spin(
        cls,
        label: str,
        subtitle: str | None,
        min_val: float,
        max_val: float,
        digits: int,
        value,
    ) -> SpinRow:
        return SpinRow(
            escape_title(label),
            escape_title(subtitle) if subtitle else None,
            lower=min_val,
            upper=max_val,
            digits=digits,
            value=float(value),
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "TupleAdapter"]:
        assert isinstance(var, TupleVar)
        min_val = var.min_val if var.min_val is not None else 0.0
        max_val = var.max_val if var.max_val is not None else 1e9
        digits = var.digits if var.digits is not None else 0
        value = getattr(var, target_property)
        if value is None:
            value = (min_val, max_val)
        subtitles = var.item_subtitles or (None, None)

        row = cls._build_spin(
            var.item_labels[0],
            subtitles[0],
            min_val,
            max_val,
            digits,
            value[0],
        )
        extra = cls._build_spin(
            var.item_labels[1],
            subtitles[1],
            min_val,
            max_val,
            digits,
            value[1],
        )
        if var.description:
            cast(Adw.ActionRow, row).set_subtitle(
                escape_title(var.description)
            )

        return row, cls(row, extra, [row, extra])

    def extra_rows(self) -> list[Adw.PreferencesRow]:
        """The second component's row, appended after the primary."""
        return [self._extra]

    def get_value(self) -> tuple[Any, ...] | None:
        return tuple(spin.get_value() for spin in self._spins)

    def set_value(self, value: Any) -> None:
        for i, spin in enumerate(self._spins):
            if i < len(value):
                spin.set_value(float(value[i]))

    def update_from_var(self, var: Var):
        assert isinstance(var, TupleVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            cast(Adw.ActionRow, self._row).set_subtitle(
                escape_title(var.description)
            )

    def set_bounds(self, lower: float, upper: float) -> None:
        """Apply machine-dependent bounds to every component spin."""
        for spin in self._spins:
            spin.set_range(lower, upper)

    def set_item_labels(self, labels: tuple[str, str]) -> None:
        """Retitle the component rows (e.g. mode-dependent dimension
        labels)."""
        self._row.set_title(escape_title(labels[0]))
        self._extra.set_title(escape_title(labels[1]))

    def set_item_subtitles(self, subtitles: tuple[str, str]) -> None:
        """Retitle the component row subtitles."""
        cast(Adw.ActionRow, self._row).set_subtitle(escape_title(subtitles[0]))
        cast(Adw.ActionRow, self._extra).set_subtitle(
            escape_title(subtitles[1])
        )
