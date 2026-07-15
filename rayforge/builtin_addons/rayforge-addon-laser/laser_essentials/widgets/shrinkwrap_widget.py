from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from rayforge.core.cut_side import CutSide
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float
from rayforge.ui_gtk.shared.slider import create_slider_row

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ShrinkWrapProducerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the ShrinkWrapStep."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        page: Adw.PreferencesPage,
        step: Any,
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            page=page,
            step=step,
            **kwargs,
        )

        gravity_adj = Gtk.Adjustment(
            lower=0.0, upper=1.0, step_increment=0.01, page_increment=0.1
        )
        gravity_adj.set_value(step.gravity)
        gravity_row, gravity_scale = create_slider_row(
            title=_("Gravity"),
            adjustment=gravity_adj,
            subtitle=_("Pulls the hull inward. 0.0 is a standard convex hull"),
            digits=2,
            on_value_changed=lambda s: self._debounce(
                self._on_param_changed, "gravity", s.get_value()
            ),
        )
        self.add(gravity_row)

        # Cut Side
        cut_side_choices = [cs.label() for cs in CutSide]
        cut_side_row = Adw.ComboRow(
            title=_("Cut Side"), model=Gtk.StringList.new(cut_side_choices)
        )
        cut_side_row.set_selected(list(CutSide).index(CutSide[step.cut_side]))
        self.add(cut_side_row)

        # Path Offset
        offset_adj = Gtk.Adjustment(
            lower=0.0,
            upper=100.0,
            step_increment=0.1,
            page_increment=1.0,
        )
        self.offset_row = Adw.SpinRow(
            title=_("Path Offset"),
            subtitle=_(
                "Absolute distance from original path in machine units, "
                "direction is controlled by Cut Side"
            ),
            adjustment=offset_adj,
            digits=2,
        )
        offset_adj.set_value(step.path_offset_mm)
        self.add(self.offset_row)

        # Connect signals
        self.offset_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed,
                "path_offset_mm",
                get_spinrow_float(r),
            ),
        )
        cut_side_row.connect("notify::selected", self._on_cut_side_changed)

        # Set initial sensitivity
        self._update_offset_sensitivity(CutSide[step.cut_side])
        cut_side_row.connect(
            "notify::selected",
            lambda r, _: self._update_offset_sensitivity(
                list(CutSide)[r.get_selected()]
            ),
        )

    def _update_offset_sensitivity(self, cut_side: CutSide):
        self.offset_row.set_sensitive(cut_side != CutSide.CENTERLINE)

    def _on_param_changed(self, key: str, new_value: Any):
        self.set_step_property(key, new_value)

    def _on_cut_side_changed(self, row, _):
        selected_idx = row.get_selected()
        new_mode = list(CutSide)[selected_idx]
        self._on_param_changed("cut_side", new_mode.name)
