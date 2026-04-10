from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk
from gettext import gettext as _

from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.context import get_context
from ..transformers import LeadInOutTransformer
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.shared.unit_spin_row import UnitSpinRowHelper

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class LeadInOutSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the LeadInOutTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: LeadInOutTransformer,
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

        self._previous_cut_speed = step.cut_speed
        step.updated.connect(self._on_step_updated)

        machine = get_context().machine
        if machine:
            machine.changed.connect(self._on_machine_changed)

        self.auto_row = Adw.SwitchRow(
            title=_("Automatic Distance"),
            subtitle=_(
                "Calculate distance based on speed and acceleration "
                "with safety factor"
            ),
        )
        self.auto_row.set_active(transformer.auto)
        self.add(self.auto_row)

        lead_in_adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.1, page_increment=1.0
        )
        lead_in_row = Adw.SpinRow(
            title=_("Lead-In Distance"),
            adjustment=lead_in_adj,
            digits=2,
        )
        lead_in_row.set_subtitle(
            _("Distance of zero-power move before cut starts")
        )
        self.add(lead_in_row)
        self.lead_in_row = lead_in_row

        self.lead_in_helper = UnitSpinRowHelper(
            spin_row=lead_in_row, quantity="length", max_value_in_base=50.0
        )
        self.lead_in_helper.set_value_in_base_units(transformer.lead_in_mm)

        lead_out_adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.1, page_increment=1.0
        )
        lead_out_row = Adw.SpinRow(
            title=_("Lead-Out Distance"),
            adjustment=lead_out_adj,
            digits=2,
        )
        lead_out_row.set_subtitle(
            _("Distance of zero-power move after cut ends")
        )
        self.add(lead_out_row)
        self.lead_out_row = lead_out_row

        self.lead_out_helper = UnitSpinRowHelper(
            spin_row=lead_out_row, quantity="length", max_value_in_base=50.0
        )
        self.lead_out_helper.set_value_in_base_units(transformer.lead_out_mm)

        self.auto_row.connect("notify::active", self._on_auto_toggled)
        self.auto_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )
        lead_in_row.connect(
            "changed",
            lambda r: self._debounce(self._on_lead_in_changed, r),
        )
        lead_out_row.connect(
            "changed",
            lambda r: self._debounce(self._on_lead_out_changed, r),
        )

        self._update_sensitivity()

    def _set_step_param(self, key, new_value, name):
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=lambda: self.step.updated.send(self.step),
        )

    def _update_sensitivity(self):
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        auto = self.auto_row.get_active()

        self.auto_row.set_sensitive(enabled)
        self.lead_in_row.set_sensitive(enabled and not auto)
        self.lead_out_row.set_sensitive(enabled and not auto)

    def _on_auto_toggled(self, row, pspec):
        new_value = row.get_active()
        self._set_step_param("auto", new_value, _("Toggle Auto Lead-In/Out"))
        if new_value:
            self._recalculate_distance()
        self._update_sensitivity()

    def _recalculate_distance(self):
        machine = get_context().machine
        if not machine:
            return

        new_distance = LeadInOutTransformer.calculate_auto_distance(
            self.step.cut_speed, machine.acceleration
        )

        self._set_step_param(
            "lead_in_mm",
            new_distance,
            _("Auto Calculate Lead-In/Out Distance"),
        )
        self._set_step_param(
            "lead_out_mm",
            new_distance,
            _("Auto Calculate Lead-In/Out Distance"),
        )

        self.lead_in_helper.set_value_in_base_units(new_distance)
        self.lead_out_helper.set_value_in_base_units(new_distance)

    def _on_step_updated(self, step: "Step"):
        if self.target_dict.get("auto", True):
            if step.cut_speed != self._previous_cut_speed:
                self._previous_cut_speed = step.cut_speed
                self._recalculate_distance()

    def _on_machine_changed(self, machine):
        if self.target_dict.get("auto", True):
            self._recalculate_distance()

    def _on_lead_in_changed(self, spin_row):
        new_value = self.lead_in_helper.get_value_in_base_units()
        if self.target_dict.get("auto", True):
            self._set_step_param("auto", False, _("Disable Auto Lead-In/Out"))
        self._set_step_param(
            "lead_in_mm", new_value, _("Change Lead-In Distance")
        )

    def _on_lead_out_changed(self, spin_row):
        new_value = self.lead_out_helper.get_value_in_base_units()
        if self.target_dict.get("auto", True):
            self._set_step_param("auto", False, _("Disable Auto Lead-In/Out"))
        self._set_step_param(
            "lead_out_mm", new_value, _("Change Lead-Out Distance")
        )
