from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from rayforge.pipeline.producer.base import CutSide
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float
from rayforge.ui_gtk.shared.slider import create_slider_row

from ..producers import CutOrder

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ContourProducerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the ContourStep."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        producer,
        page: Adw.PreferencesPage,
        step: Any,
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            component=producer,
            page=page,
            step=step,
            **kwargs,
        )

        switch_row = Adw.SwitchRow(
            title=_("Remove Inner Paths"),
            subtitle=_("If enabled, only trace the outer outline of shapes"),
        )
        switch_row.set_active(step.remove_inner_paths)
        self.add(switch_row)

        # Cut Side
        cut_side_choices = [cs.label() for cs in CutSide]
        cut_side_row = Adw.ComboRow(
            title=_("Cut Side"), model=Gtk.StringList.new(cut_side_choices)
        )
        cut_side_row.set_selected(list(CutSide).index(CutSide[step.cut_side]))
        self.add(cut_side_row)

        # Cut Order
        cut_order_choices = [co.label() for co in CutOrder]
        cut_order_row = Adw.ComboRow(
            title=_("Cut Order"),
            subtitle=_("Processing order for nested paths"),
            model=Gtk.StringList.new(cut_order_choices),
        )
        cut_order_row.set_selected(
            list(CutOrder).index(CutOrder[step.cut_order])
        )
        self.add(cut_order_row)

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

        # Overcut
        overcut_adj = Gtk.Adjustment(
            lower=0.0,
            upper=100.0,
            step_increment=0.1,
            page_increment=1.0,
        )
        self.overcut_row = Adw.SpinRow(
            title=_("Overcut"),
            subtitle=_(
                "Extend closed contours past their start point "
                "so the cut overlaps itself"
            ),
            adjustment=overcut_adj,
            digits=2,
        )
        overcut_adj.set_value(step.overcut)
        self.add(self.overcut_row)

        # Threshold Override Toggle
        self.override_switch_row = Adw.SwitchRow(
            title=_("Rescan Content"),
            subtitle=_(
                "Ignore source geometry and re-trace within the workpiece"
            ),
        )
        self.override_switch_row.set_active(step.override_threshold)
        self.add(self.override_switch_row)

        threshold_adj = Gtk.Adjustment(
            lower=0.0,
            upper=1.0,
            step_increment=0.01,
            page_increment=0.1,
            value=step.threshold,
        )
        self.threshold_row, self.threshold_scale = create_slider_row(
            title=_("Tracing Threshold"),
            adjustment=threshold_adj,
            subtitle=_("Brightness level (0.0-1.0) to define edges"),
            digits=2,
            on_value_changed=lambda s: self._debounce(
                self._on_param_changed, "threshold", s.get_value()
            ),
        )
        self.add(self.threshold_row)

        # Connect signals
        switch_row.connect(
            "notify::active",
            lambda w, _: self._on_param_changed(
                "remove_inner_paths", w.get_active()
            ),
        )
        self.offset_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed, "path_offset_mm", get_spinrow_float(r)
            ),
        )
        self.overcut_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed, "overcut", get_spinrow_float(r)
            ),
        )
        cut_side_row.connect("notify::selected", self._on_cut_side_changed)
        cut_order_row.connect("notify::selected", self._on_cut_order_changed)

        self.override_switch_row.connect(
            "notify::active", self._on_override_changed
        )

        # Set initial sensitivity
        self._update_offset_sensitivity(CutSide[step.cut_side])
        self._update_threshold_sensitivity(step.override_threshold)

        cut_side_row.connect(
            "notify::selected",
            lambda r, _: self._update_offset_sensitivity(
                list(CutSide)[r.get_selected()]
            ),
        )

    def _update_offset_sensitivity(self, cut_side: CutSide):
        self.offset_row.set_sensitive(cut_side != CutSide.CENTERLINE)

    def _update_threshold_sensitivity(self, active: bool):
        self.threshold_row.set_sensitive(active)

    def _on_param_changed(self, key: str, new_value: Any):
        self.set_step_property(key, new_value)

    def _on_cut_side_changed(self, row, _):
        selected_idx = row.get_selected()
        new_mode = list(CutSide)[selected_idx]
        self._on_param_changed("cut_side", new_mode.name)

    def _on_cut_order_changed(self, row, _):
        selected_idx = row.get_selected()
        new_mode = list(CutOrder)[selected_idx]
        self._on_param_changed("cut_order", new_mode.name)

    def _on_override_changed(self, row, _):
        active = row.get_active()
        self._update_threshold_sensitivity(active)
        self._on_param_changed("override_threshold", active)
