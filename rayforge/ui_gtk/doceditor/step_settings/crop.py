from typing import Dict, Any, TYPE_CHECKING
from gettext import gettext as _
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.transformer import CropTransformer
from ....shared.util.glib import DebounceMixin
from ....context import get_context
from ...shared.adwfix import get_spinrow_float

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class CropTransformerSettingsWidget(
    DebounceMixin, StepComponentSettingsWidget
):
    """UI for configuring the CropTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        transformer = CropTransformer.from_dict(target_dict)

        machine = get_context().machine
        if machine and "tolerance" not in target_dict:
            target_dict["tolerance"] = machine.arc_tolerance

        super().__init__(
            editor,
            title,
            description=transformer.description,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        switch_row = Adw.SwitchRow(title=_("Enable Crop-to-Stock"))
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        offset_adj = Gtk.Adjustment(
            lower=-100.0, upper=100.0, step_increment=0.1, page_increment=1.0
        )
        offset_row = Adw.SpinRow(
            title=_("Offset"),
            adjustment=offset_adj,
            digits=3,
        )
        offset_row.set_subtitle(
            _("Grow/shrink stock boundary before cropping")
        )
        offset_adj.set_value(transformer.offset)
        self.add(offset_row)
        self.offset_row = offset_row

        is_enabled = transformer.enabled
        offset_row.set_sensitive(is_enabled)

        switch_row.connect("notify::active", self._on_enable_toggled)
        switch_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )
        offset_row.connect(
            "changed",
            lambda r: self._debounce(self._on_offset_changed, r),
        )

    def _set_step_param(self, key, new_value, name):
        """Helper method to set a step parameter with standard callback."""
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

    def _update_sensitivity(self):
        """Update the sensitivity of UI elements based on current state."""
        enabled = self.target_dict.get("enabled", True)
        self.offset_row.set_sensitive(enabled)

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        self._set_step_param("enabled", new_value, _("Toggle Crop-to-Stock"))
        self._update_sensitivity()

    def _on_offset_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
        self._set_step_param("offset", new_value, _("Change Crop Offset"))
