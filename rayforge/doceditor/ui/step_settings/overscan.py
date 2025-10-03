from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.transformer import OverscanTransformer
from ....shared.util.adwfix import get_spinrow_float
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....core.step import Step
    from ....undo import HistoryManager
    from ....doceditor.editor import DocEditor


class OverscanSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the OverscanTransformer."""

    def __init__(
        self,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        editor: "DocEditor",
        history_manager: "HistoryManager",
        **kwargs,
    ):
        transformer = OverscanTransformer.from_dict(target_dict)

        super().__init__(
            description=transformer.description,
            target_dict=target_dict,
            page=page,
            step=step,
            editor=editor,
            history_manager=history_manager,
            **kwargs,
        )
        self.editor = editor

        # Main toggle switch
        switch_row = Adw.SwitchRow(title=_("Enable Overscan"))
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        # Distance setting
        distance_adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.1, page_increment=1.0
        )
        distance_row = Adw.SpinRow(
            title=_("Overscan Distance (mm)"),
            adjustment=distance_adj,
            digits=2,
        )
        distance_adj.set_value(transformer.distance_mm)
        self.add(distance_row)

        # Connect signals
        switch_row.connect("notify::active", self._on_enable_toggled)
        distance_row.connect(
            "changed",
            lambda r: self._debounce(self._on_distance_changed, r),
        )

        # Set initial sensitivity
        distance_row.set_sensitive(transformer.enabled)
        switch_row.connect(
            "notify::active",
            lambda w, _: distance_row.set_sensitive(w.get_active()),
        )

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Overscan"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

    def _on_distance_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="distance_mm",
            new_value=new_value,
            name=_("Change Overscan Distance"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
