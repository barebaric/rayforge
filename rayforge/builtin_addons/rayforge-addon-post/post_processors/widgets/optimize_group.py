from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject

from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import Optimize

if TYPE_CHECKING:
    from rayforge.core.step import Step


class OptimizeSettingsGroup(TransformerSettingsGroup):
    """UI for configuring the Optimize transformer."""

    def __init__(
        self,
        title: str,
        transformer: Optimize,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self.flip_row = Adw.SwitchRow(
            title=_("Allow Flipping"),
            subtitle=_("Allow reversing path direction for shorter travel"),
        )
        self.flip_row.set_active(transformer.allow_flip)
        self.add(self.flip_row)
        self.flip_row.connect("notify::active", self._on_flip_toggled)

        self.preserve_row = Adw.SwitchRow(
            title=_("Preserve First Workpiece"),
            subtitle=_("Keep the first workpiece at its original position"),
        )
        self.preserve_row.set_active(transformer.preserve_first)
        self.add(self.preserve_row)
        self.preserve_row.connect(
            "notify::active", self._on_preserve_first_toggled
        )

        self.best_start_row = Adw.SwitchRow(
            title=_("Choose Best Start Point"),
            subtitle=_(
                "Enter closed paths at the vertex nearest the current "
                "position, hiding the seam of every loop."
            ),
        )
        self.best_start_row.set_active(transformer.best_start_point)
        self.add(self.best_start_row)
        self.best_start_row.connect(
            "notify::active", self._on_best_start_toggled
        )

        self.prefer_corners_row = Adw.SwitchRow(
            title=_("Prefer Corners"),
            subtitle=_(
                "Only start closed paths at sharp corners, when the "
                "path has any (circles fall back to any vertex)."
            ),
        )
        self.prefer_corners_row.set_active(transformer.prefer_corners)
        self.prefer_corners_row.set_sensitive(transformer.best_start_point)
        self.add(self.prefer_corners_row)
        self.prefer_corners_row.connect(
            "notify::active", self._on_prefer_corners_toggled
        )

        self.merge_row = Adw.SwitchRow(
            title=_("Merge Scanlines"),
            subtitle=_(
                "Merges nearby parallel cut lines into longer scanlines "
                "when this is faster for the machine's acceleration."
            ),
        )
        self.merge_row.set_active(transformer.merge_scanlines)
        self.add(self.merge_row)
        self.merge_row.connect("notify::active", self._on_merge_toggled)

        self.merge_gap_row = LengthSpinRow(
            _("Max gap"),
            _(
                "Maximum gap to bridge at zero power; 0 decides "
                "automatically from the machine's acceleration"
            ),
            lower=0.0,
            upper=1000.0,
            value_in_base=transformer.merge_max_gap_mm,
        )
        self.merge_gap_row.set_sensitive(transformer.merge_scanlines)
        self.add(self.merge_gap_row)
        self.merge_gap_row.value_changed.connect(
            lambda r: self._on_merge_gap_changed(r)
        )

    def _on_flip_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="allow_flip",
            value=row.get_active(),
            name=_("Toggle Flipping"),
        )

    def _on_preserve_first_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="preserve_first",
            value=row.get_active(),
            name=_("Toggle Preserve First Workpiece"),
        )

    def _on_best_start_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.prefer_corners_row.set_sensitive(row.get_active())
        self.param_changed.send(
            self,
            key="best_start_point",
            value=row.get_active(),
            name=_("Toggle Choose Best Start Point"),
        )

    def _on_prefer_corners_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.param_changed.send(
            self,
            key="prefer_corners",
            value=row.get_active(),
            name=_("Toggle Prefer Corners"),
        )

    def _on_merge_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self.merge_gap_row.set_sensitive(row.get_active())
        self.param_changed.send(
            self,
            key="merge_scanlines",
            value=row.get_active(),
            name=_("Merge Scanlines"),
        )

    def _on_merge_gap_changed(self, row: LengthSpinRow) -> None:
        self.param_changed.send(
            self,
            key="merge_max_gap_mm",
            value=row.get_value_in_base_units(),
            name=_("Change maximum bridged gap"),
        )
