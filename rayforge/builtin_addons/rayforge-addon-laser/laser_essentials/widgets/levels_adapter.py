"""Adapter for the raster brightness-range (levels) histogram row."""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw

from rayforge.core.varset import Var
from rayforge.ui_gtk.shared.histogram_preview import HistogramPreview
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)

from ..levels_range_var import LevelsRangeVar


@register_adapter(LevelsRangeVar)
class LevelsAdapter(RowAdapter):
    """A row for the raster brightness range.

    Renders an :class:`Adw.ActionRow` whose suffix carries a
    :class:`HistogramPreview` for dragging the black/white points. The
    auto-levels switch lives on its own row (a plain ``auto_levels``
    BoolVar); this adapter picks up its state via
    :meth:`update_from_values` to switch the preview into/out of auto
    mode.

    It edits two step attributes at once:

    * ``black_point`` — primary key, the black drag marker
    * ``white_point`` — related key, the white drag marker

    The manager maps the related key to this adapter (no separate
    row) and emits ``data_changed`` for both keys whenever a marker
    moves.
    """

    related_keys = ("white_point",)

    def __init__(
        self,
        row: Adw.ActionRow,
        preview: HistogramPreview,
    ) -> None:
        super().__init__()
        self._row = row
        self.preview = preview
        preview.black_point_changed.connect(
            lambda s, **kw: self.changed.send(self)
        )
        preview.white_point_changed.connect(
            lambda s, **kw: self.changed.send(self)
        )

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "LevelsAdapter"]:
        assert isinstance(var, LevelsRangeVar)
        row = Adw.ActionRow(title=escape_title(var.label))
        if var.description:
            row.set_subtitle(escape_title(var.description))

        preview = HistogramPreview()
        preview.set_points(var.value if var.value is not None else 0, 255)
        row.add_suffix(preview)

        adapter = cls(row, preview)
        adapter._update_subtitle()
        return row, adapter

    def get_value(self) -> Any | None:
        return self.preview.black_point

    def set_value(self, value: Any) -> None:
        self.preview.black_point = int(value)

    def get_value_for_key(self, key: str) -> Any | None:
        if key == "white_point":
            return self.preview.white_point
        return self.get_value()

    def set_value_for_key(self, key: str, value: Any) -> None:
        if key == "white_point":
            self.preview.white_point = int(value)
        else:
            self.set_value(value)

    def update_from_var(self, var: Var):
        assert isinstance(var, LevelsRangeVar)
        if var.label:
            self._row.set_title(escape_title(var.label))
        if var.description:
            self._row.set_subtitle(escape_title(var.description))

    def update_from_values(self, values: dict[str, Any]) -> None:
        """Sync the preview's auto mode from the sibling auto_levels."""
        auto_mode = bool(values.get("auto_levels", True))
        if auto_mode != self.preview.auto_mode:
            self.preview.auto_mode = auto_mode
            self._update_subtitle()

    def _update_subtitle(self):
        if self.preview.auto_mode:
            self._row.set_subtitle(
                escape_title(_("Auto-adjusted based on image content"))
            )
        else:
            self._row.set_subtitle(
                escape_title(_("Drag markers to set black/white points"))
            )
