from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.transformer import OverscanTransformer
from ....shared.util.glib import DebounceMixin
from ...shared.unit_spin_row import UnitSpinRowHelper
from ....context import get_context

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class OverscanSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the OverscanTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        transformer = OverscanTransformer.from_dict(target_dict)

        super().__init__(
            editor,
            title,
            description=transformer.description,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        # Main toggle switch
        switch_row = Adw.SwitchRow(title=_("Enable Overscan"))
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        # Auto mode toggle
        self.auto_row = Adw.SwitchRow(
            title=_("Automatic Distance"),
            subtitle=_(
                "Calculate distance based on speed and acceleration with "
                "safety factor"
            ),
        )
        self.auto_row.set_active(transformer.auto)
        self.add(self.auto_row)

        # Distance setting with unit support
        distance_adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.1, page_increment=1.0
        )
        distance_row = Adw.SpinRow(
            title=_("Overscan Distance"),
            adjustment=distance_adj,
            digits=2,
        )
        distance_row.set_subtitle(_("Manual distance setting"))
        self.add(distance_row)
        self.distance_row = distance_row  # Store reference for later access

        # Add unit conversion helper for length
        self.distance_helper = UnitSpinRowHelper(
            spin_row=distance_row, quantity="length", max_value_in_base=50.0
        )
        self.distance_helper.set_value_in_base_units(transformer.distance_mm)

        # Connect signals
        switch_row.connect("notify::active", self._on_enable_toggled)
        self.auto_row.connect("notify::active", self._on_auto_toggled)
        distance_row.connect(
            "changed",
            lambda r: self._debounce(self._on_distance_changed, r),
        )

        # Set initial sensitivity
        is_sensitive = transformer.enabled and not transformer.auto
        distance_row.set_sensitive(is_sensitive)
        switch_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )
        self.auto_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )

    def _update_sensitivity(self):
        """Update the sensitivity of UI elements based on current state."""
        enabled = self.target_dict.get("enabled", True)
        auto = self.target_dict.get("auto", True)

        # Use the stored references to the rows
        self.distance_row.set_sensitive(enabled and not auto)
        self.auto_row.set_sensitive(enabled)

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Overscan"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self._update_sensitivity()

    def _on_auto_toggled(self, row, pspec):
        new_value = row.get_active()
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="auto",
            new_value=new_value,
            name=_("Toggle Auto Overscan"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

        # If auto is enabled, recalculate the distance
        if new_value:
            self._recalculate_distance()

        self._update_sensitivity()

    def _recalculate_distance(self):
        """Recalculate the overscan distance based on current step settings."""
        machine = get_context().machine
        if not machine:
            return

        # Calculate new distance
        new_distance = OverscanTransformer.calculate_auto_distance(
            self.step.cut_speed, machine.acceleration
        )

        # Update the distance
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="distance_mm",
            new_value=new_distance,
            name=_("Auto Calculate Overscan Distance"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

        # Update the UI
        self.distance_helper.set_value_in_base_units(new_distance)

    def _on_distance_changed(self, spin_row):
        # Get the value in base units directly from the helper
        new_value = self.distance_helper.get_value_in_base_units()

        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="distance_mm",
            new_value=new_value,
            name=_("Change Overscan Distance"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
