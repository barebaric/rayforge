from typing import Dict, Any, TYPE_CHECKING, cast
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.producer.base import OpsProducer, CutSide
from ....pipeline.producer.edge import EdgeTracer
from ....undo import DictItemCommand
from ....shared.util.adwfix import get_spinrow_float
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager


class EdgeTracerSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the EdgeTracer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        producer = cast(EdgeTracer, OpsProducer.from_dict(target_dict))

        super().__init__(
            target_dict=target_dict,
            page=page,
            step=step,
            history_manager=history_manager,
            **kwargs,
        )

        # Remove inner paths toggle
        switch_row = Adw.SwitchRow(
            title=_("Remove Inner Paths"),
            subtitle=_("If enabled, only trace the outer outline of shapes."),
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
                "Absolute distance from original path. Direction is "
                "controlled by Cut Side."
            ),
            adjustment=offset_adj,
            digits=2,
        )
        offset_adj.set_value(producer.path_offset_mm)
        self.add(self.offset_row)

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

        # Set initial sensitivity
        self._update_offset_sensitivity(producer.cut_side)
        cut_side_row.connect(
            "notify::selected",
            lambda r, _: self._update_offset_sensitivity(
                list(CutSide)[r.get_selected()]
            ),
        )

    def _update_offset_sensitivity(self, cut_side: CutSide):
        self.offset_row.set_sensitive(cut_side != CutSide.CENTERLINE)

    def _on_param_changed(self, key: str, new_value: Any):
        params_dict = self.target_dict.setdefault("params", {})
        if isinstance(new_value, float):
            if abs(new_value - params_dict.get(key, 0.0)) < 1e-6:
                return
        elif new_value == params_dict.get(key):
            return

        command = DictItemCommand(
            target_dict=params_dict,
            key=key,
            new_value=new_value,
            name=_("Change Edge Tracer Setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_cut_side_changed(self, row, _):
        selected_idx = row.get_selected()
        new_mode = list(CutSide)[selected_idx]
        self._on_param_changed("cut_side", new_mode.name)
