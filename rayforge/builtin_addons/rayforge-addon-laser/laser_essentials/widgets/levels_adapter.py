"""Adapter for the raster brightness-range (levels) histogram row."""

from gettext import gettext as _
from typing import Any

import numpy as np
from gi.repository import Adw
from raygeo.image.grayscale import compute_auto_levels

from rayforge.core.varset import Var
from rayforge.image.util import get_visible_grayscale_values
from rayforge.ui_gtk.shared.histogram_preview import HistogramPreview
from rayforge.ui_gtk.varset.adapter import (
    RowAdapter,
    escape_title,
    register_adapter,
)

from ..levels_range_var import LevelsRangeVar

_MAX_PREVIEW_PX = 256


@register_adapter(LevelsRangeVar)
class LevelsAdapter(RowAdapter):
    """A row for the raster brightness range.

    Renders an :class:`Adw.ActionRow` whose suffix carries a
    :class:`HistogramPreview` for dragging the black/white points. The
    auto-levels switch lives on its own row (a plain ``auto_levels``
    BoolVar); this adapter picks up its state via
    :meth:`update_from_values` to switch the preview into/out of auto
    mode.

    The adapter also owns the histogram data: the dialog points it at
    the step via :meth:`set_histogram_source` and it (re)computes the
    histogram from the step's layer whenever ``invert`` changes (the
    value arrives through :meth:`update_from_values`).

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
        self._step: Any = None
        self._last_invert: bool | None = None
        self._histogram_ready = False
        preview.black_point_changed.connect(
            lambda s, **kw: self.changed.send(self),
            weak=False,
        )
        preview.white_point_changed.connect(
            lambda s, **kw: self.changed.send(self),
            weak=False,
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

    def set_histogram_source(self, step: Any) -> None:
        """Point the adapter at the step whose layer supplies the
        histogram data. The dialog calls this once at build time.
        """
        self._step = step

    def compute_histogram(self) -> None:
        """(Re)compute the histogram from the step's layer and push
        it into the preview. No-op without a histogram source.
        """
        step = self._step
        if step is None:
            return
        invert = bool(self._last_invert or False)

        layer = step.layer
        if not layer:
            self.preview.update_histogram(None)
            return

        workpieces = layer.all_workpieces
        if not workpieces:
            self.preview.update_histogram(None)
            return

        pixels_per_mm = step.pixels_per_mm
        all_gray_values = []

        for workpiece in workpieces:
            size = workpiece.size
            if not size or size[0] <= 0 or size[1] <= 0:
                continue

            width_px = int(size[0] * pixels_per_mm[0])
            height_px = int(size[1] * pixels_per_mm[1])

            if width_px <= 0 or height_px <= 0:
                continue

            if width_px > _MAX_PREVIEW_PX or height_px > _MAX_PREVIEW_PX:
                scale = min(
                    _MAX_PREVIEW_PX / width_px,
                    _MAX_PREVIEW_PX / height_px,
                )
                width_px = max(int(width_px * scale), 1)
                height_px = max(int(height_px * scale), 1)

            surface = workpiece.render_to_pixels(width_px, height_px)
            if not surface:
                continue

            gray_values = get_visible_grayscale_values(surface, invert)
            if gray_values.size > 0:
                all_gray_values.append(gray_values)

        if not all_gray_values:
            self.preview.update_histogram(None)
            return

        combined_gray = np.concatenate(all_gray_values)

        histogram, _ = np.histogram(combined_gray, bins=64, range=(0, 255))

        self.preview.update_histogram(histogram)

        auto_black, auto_white = compute_auto_levels(combined_gray)
        self.preview.set_auto_points(auto_black, auto_white)
        self._histogram_ready = True

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
        """Sync the preview's auto mode from the sibling auto_levels,
        and recompute the histogram when invert changes."""
        auto_mode = bool(values.get("auto_levels", True))
        if auto_mode != self.preview.auto_mode:
            self.preview.auto_mode = auto_mode
            self._update_subtitle()

        invert = values.get("invert")
        if invert is not None and invert != self._last_invert:
            self._last_invert = bool(invert)
            if self._histogram_ready:
                self.compute_histogram()

    def _update_subtitle(self):
        if self.preview.auto_mode:
            self._row.set_subtitle(
                escape_title(_("Auto-adjusted based on image content"))
            )
        else:
            self._row.set_subtitle(
                escape_title(_("Drag markers to set black/white points"))
            )
