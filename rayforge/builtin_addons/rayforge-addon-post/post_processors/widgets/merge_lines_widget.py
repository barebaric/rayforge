from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk
from gettext import gettext as _

from rayforge.core.undo import DictItemCommand
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float
from ..transformers import MergeLinesTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class MergeLinesSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the MergeLinesTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: MergeLinesTransformer,
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        super().__init__(
            editor,
            title,
            component=transformer,
            page=page,
            step=step,
            description=transformer.description,
            **kwargs,
        )

        switch_row = Adw.SwitchRow(
            title=_("Enable Merge Lines"),
            subtitle=_(
                "Merges overlapping line segments to avoid double passing"
            ),
        )
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        tolerance_adj = Gtk.Adjustment(
            lower=0.01, upper=10.0, step_increment=0.1, page_increment=1.0
        )
        tolerance_adj.set_value(transformer.tolerance)
        tolerance_row = Adw.SpinRow(
            title=_("Tolerance"),
            subtitle=_(
                "Maximum distance for lines to be considered overlapping"
            ),
            adjustment=tolerance_adj,
        )
        tolerance_row.set_digits(2)
        self.add(tolerance_row)

        is_enabled = transformer.enabled
        tolerance_row.set_sensitive(is_enabled)

        switch_row.connect("notify::active", self._on_enable_toggled)
        switch_row.connect(
            "notify::active",
            self._on_sensitivity_toggled,
            tolerance_row,
        )
        tolerance_row.connect(
            "changed",
            lambda spin_row: self._debounce(
                self._on_tolerance_changed, spin_row
            ),
        )

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Merge Lines"),
            on_change_callback=self.step.per_step_transformer_changed.send,
        )
        self.history_manager.execute(command)

    def _on_sensitivity_toggled(self, row, pspec, tolerance_row):
        is_active = row.get_active()
        tolerance_row.set_sensitive(is_active)

    def _on_tolerance_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        command = DictItemCommand(
            target_dict=self.target_dict,
            key="tolerance",
            new_value=new_value,
            name=_("Change merge tolerance"),
            on_change_callback=self.step.per_step_transformer_changed.send,
        )
        self.history_manager.execute(command)
