from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, GObject

from rayforge.context import get_context
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.post_processor.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow

from ..transformers import LeadInOutTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class LeadInOutSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the LeadInOutTransformer."""

    def __init__(
        self,
        title: str,
        transformer: LeadInOutTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        self._auto = transformer.auto
        self._previous_cut_speed = step.cut_speed if step is not None else None
        if step is not None:
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

    def _update_sensitivity(self) -> None:
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        auto = self.auto_row.get_active()

        self.auto_row.set_sensitive(enabled)
        self.lead_in_row.set_sensitive(enabled and not auto)
        self.lead_out_row.set_sensitive(enabled and not auto)

    def _on_auto_toggled(
        self, row: Adw.SwitchRow, _pspec: GObject.ParamSpec
    ) -> None:
        self._auto = row.get_active()
        self.param_changed.send(
            self,
            key="auto",
            value=self._auto,
            name=_("Toggle Auto Lead-In/Out"),
        )
        if self._auto:
            self._recalculate_distance()
        self._update_sensitivity()

    def _recalculate_distance(self) -> None:
        machine = get_context().machine
        if not machine or self.step is None:
            return

        new_distance = LeadInOutTransformer.calculate_auto_distance(
            self.step.cut_speed, machine.acceleration
        )

        self.param_changed.send(
            self,
            key="lead_in_mm",
            value=new_distance,
            name=_("Auto Calculate Lead-In/Out Distance"),
        )
        self.param_changed.send(
            self,
            key="lead_out_mm",
            value=new_distance,
            name=_("Auto Calculate Lead-In/Out Distance"),
        )

        self.lead_in_row.set_value_in_base_units(new_distance)
        self.lead_out_row.set_value_in_base_units(new_distance)

    def _on_step_updated(self, step: "Step") -> None:
        if self._auto and step.cut_speed != self._previous_cut_speed:
            self._previous_cut_speed = step.cut_speed
            self._recalculate_distance()

    def _on_machine_changed(self, machine) -> None:
        if self._auto:
            self._recalculate_distance()

    def _on_lead_in_changed(self, spin_row: LengthSpinRow) -> None:
        new_value = spin_row.get_value_in_base_units()
        if self._auto:
            self._auto = False
            self.param_changed.send(
                self,
                key="auto",
                value=False,
                name=_("Disable Auto Lead-In/Out"),
            )
        self.param_changed.send(
            self,
            key="lead_in_mm",
            value=new_value,
            name=_("Change Lead-In Distance"),
        )

    def _on_lead_out_changed(self, spin_row: LengthSpinRow) -> None:
        new_value = spin_row.get_value_in_base_units()
        if self._auto:
            self._auto = False
            self.param_changed.send(
                self,
                key="auto",
                value=False,
                name=_("Disable Auto Lead-In/Out"),
            )
        self.param_changed.send(
            self,
            key="lead_out_mm",
            value=new_value,
            name=_("Change Lead-Out Distance"),
        )
