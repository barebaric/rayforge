from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer, CutSide
from ....pipeline.producer.contour import ContourProducer, CutOrder
from ....shared.ui.adwfix import get_spinrow_float
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....core.step import Step
    from ...editor import DocEditor


class ContourProducerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the ContourProducer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        producer = cast(ContourProducer, OpsProducer.from_dict(target_dict))

        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        # Remove inner paths toggle
        switch_row = Adw.SwitchRow(
            title=_("Remove Inner Paths"),
            subtitle=_("If enabled, only trace the outer outline of shapes"),
        )
        switch_row.set_active(producer.remove_inner_paths)
        self.add(switch_row)

        # Cut Side
        cut_side_choices = [
            _(cs.name.replace("_", " ").title()) for cs in CutSide
        ]
        cut_side_row = Adw.ComboRow(
            title=_("Cut Side"), model=Gtk.StringList.new(cut_side_choices)
        )
        cut_side_row.set_selected(list(CutSide).index(producer.cut_side))
        self.add(cut_side_row)

        # Cut Order
        cut_order_choices = [
            _(co.name.replace("_", "-").title()) for co in CutOrder
        ]
        cut_order_row = Adw.ComboRow(
            title=_("Cut Order"),
            subtitle=_("Processing order for nested paths"),
            model=Gtk.StringList.new(cut_order_choices),
        )
        cut_order_row.set_selected(list(CutOrder).index(producer.cut_order))
        self.add(cut_order_row)

        # Path Offset
        offset_adj = Gtk.Adjustment(
            lower=0.0,
            upper=100.0,
            step_increment=0.1,
            page_increment=1.0,
        )
        self.offset_row = Adw.SpinRow(
            title=_("Path Offset (mm)"),
            subtitle=_(
                "Absolute distance from original path, direction is "
                "controlled by Cut Side"
            ),
            adjustment=offset_adj,
            digits=2,
        )
        offset_adj.set_value(producer.path_offset_mm)
        self.add(self.offset_row)

        # Threshold Override Toggle
        self.override_switch_row = Adw.SwitchRow(
            title=_("Rescan Content"),
            subtitle=_(
                "Ignore source geometry and re-trace within the workpiece"
            ),
        )
        self.override_switch_row.set_active(producer.override_threshold)
        self.add(self.override_switch_row)

        # Threshold Slider (using ActionRow + Gtk.Scale)
        self.threshold_row = Adw.ActionRow(
            title=_("Tracing Threshold"),
            subtitle=_("Brightness level (0.0-1.0) to define edges"),
        )

        threshold_adj = Gtk.Adjustment(
            lower=0.0,
            upper=1.0,
            step_increment=0.01,
            page_increment=0.1,
        )
        threshold_adj.set_value(producer.threshold)

        self.threshold_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=threshold_adj,
            digits=2,
            draw_value=True,
        )
        self.threshold_scale.set_size_request(150, -1)  # Ensure it has width
        self.threshold_row.add_suffix(self.threshold_scale)
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
        cut_side_row.connect("notify::selected", self._on_cut_side_changed)
        cut_order_row.connect("notify::selected", self._on_cut_order_changed)

        self.override_switch_row.connect(
            "notify::active", self._on_override_changed
        )
        self.threshold_scale.connect(
            "value-changed",
            lambda s: self._debounce(
                self._on_param_changed, "threshold", s.get_value()
            ),
        )

        # Set initial sensitivity
        self._update_offset_sensitivity(producer.cut_side)
        self._update_threshold_sensitivity(producer.override_threshold)

        cut_side_row.connect(
            "notify::selected",
            lambda r, _: self._update_offset_sensitivity(
                list(CutSide)[r.get_selected()]
            ),
        )

    def _update_offset_sensitivity(self, cut_side: CutSide):
        self.offset_row.set_sensitive(cut_side != CutSide.CENTERLINE)

    def _update_threshold_sensitivity(self, active: bool):
        # Disable the entire ActionRow which contains the slider
        self.threshold_row.set_sensitive(active)

    def _on_param_changed(self, key: str, new_value: Any):
        params_dict = self.target_dict.setdefault("params", {})
        self.editor.step.set_step_param(
            target_dict=params_dict,
            key=key,
            new_value=new_value,
            name=_("Change Edge Tracer Setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

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
