from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Gtk

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import AngleSpinRow
from rayforge.ui_gtk.shared.slider import create_slider_row

from ..transformers import Smooth

if TYPE_CHECKING:
    from rayforge.core.step import Step


class SmoothSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the Smooth transformer."""

    def __init__(
        self,
        title: str,
        transformer: Smooth,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        amount_adj = Gtk.Adjustment(
            lower=0, upper=100, step_increment=1, page_increment=10
        )
        amount_adj.set_value(transformer.amount)
        amount_row, _amount_scale = create_slider_row(
            title=_("Smoothness"),
            subtitle=_("Higher values produce smoother curves"),
            adjustment=amount_adj,
            digits=0,
            on_value_changed=lambda s: self._debounce(
                self._on_amount_changed, s
            ),
        )
        self.add(amount_row)

        # Corner Angle Threshold Setting
        corner_row = AngleSpinRow(
            _("Corner Angle Threshold"),
            _("Angles sharper than this are kept as corners (degrees)"),
            lower=0,
            upper=179,
            value=transformer.corner_angle_threshold,
        )
        self.add(corner_row)

        corner_row.value_changed.connect(
            lambda spin_row: self._debounce(
                self._on_corner_angle_changed, spin_row
            )
        )

    def _on_amount_changed(self, scale: Gtk.Scale) -> None:
        new_value = int(scale.get_value())
        self.param_changed.send(
            self, key="amount", value=new_value, name=_("Change smoothness")
        )

    def _on_corner_angle_changed(self, spin_row: AngleSpinRow) -> None:
        new_value = spin_row.get_int_value()
        self.param_changed.send(
            self,
            key="corner_angle_threshold",
            value=new_value,
            name=_("Change corner angle"),
        )
