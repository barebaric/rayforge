from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.context import get_context
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows.length_spin_row import LengthSpinRow

from ..transformers import LeadInOutTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class LeadInOutSettingsGroup(DebounceMixin, TransformerSettingsGroup):
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

        self.lead_in_row = LengthSpinRow(
            _("Lead-In Distance"),
            _("Distance of zero-power move before cut starts"),
            upper=50.0,
            value_in_base=transformer.lead_in_mm,
        )
        self.add(self.lead_in_row)

        self.lead_out_row = LengthSpinRow(
            _("Lead-Out Distance"),
            _("Distance of zero-power move after cut ends"),
            upper=50.0,
            value_in_base=transformer.lead_out_mm,
        )
        self.add(self.lead_out_row)

        self.auto_row.connect("notify::active", self._on_auto_toggled)
        self.auto_row.connect(
            "notify::active",
            lambda w, _: self._update_sensitivity(),
        )
        self.lead_in_row.value_changed.connect(
            lambda r: self._debounce(self._on_lead_in_changed, r),
        )
        self.lead_out_row.value_changed.connect(
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

        self.lead_in_row.set_value_in_base_units(new_distance)
        self.lead_out_row.set_value_in_base_units(new_distance)

    def _on_step_updated(self, step: "Step"):
        if self.target_dict.get("auto", True):
            if step.cut_speed != self._previous_cut_speed:
                self._previous_cut_speed = step.cut_speed
                self._recalculate_distance()

    def _on_machine_changed(self, machine):
        if self.target_dict.get("auto", True):
            self._recalculate_distance()

    def _on_lead_in_changed(self, spin_row):
        new_value = spin_row.get_value_in_base_units()
        if self.target_dict.get("auto", True):
            self._set_step_param("auto", False, _("Disable Auto Lead-In/Out"))
        self._set_step_param(
            "lead_in_mm", new_value, _("Change Lead-In Distance")
        )

    def _on_lead_out_changed(self, spin_row):
        new_value = spin_row.get_value_in_base_units()
        if self.target_dict.get("auto", True):
            self._set_step_param("auto", False, _("Disable Auto Lead-In/Out"))
        self._set_step_param(
            "lead_out_mm", new_value, _("Change Lead-Out Distance")
        )
