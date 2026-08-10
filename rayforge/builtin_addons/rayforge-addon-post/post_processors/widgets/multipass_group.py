from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    ExpanderHost,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows import LengthSpinRow, SpinRow

from ..transformers import MultiPassTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step


class MultiPassSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the MultiPassTransformer."""

    def __init__(
        self,
        title: str,
        transformer: MultiPassTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        # Passes setting
        self.passes_row = SpinRow(
            _("Number of Passes"),
            _("How often to repeat the entire step"),
            lower=1,
            upper=100,
            value=transformer.passes,
        )
        self.add(self.passes_row)

        # Z Step-down setting
        self.z_step_row = LengthSpinRow(
            _("Z Step-Down per Pass"),
            _("Distance to lower Z-axis for each subsequent pass"),
            upper=50.0,
            value_in_base=transformer.z_step_down,
        )
        self.add(self.z_step_row)
        self.z_step_row.value_changed.connect(
            lambda r: self._debounce(self._on_z_step_down_changed, r)
        )

        # Connect signals with debouncing
        self.passes_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_passes_changed, r, self.z_step_row
            ),
        )

        # Z Step-down is only available with multiple passes
        if transformer.passes <= 1:
            self.z_step_row.set_sensitive(False)

    def _update_sensitivity(self) -> None:
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        self.passes_row.set_sensitive(enabled)
        self.z_step_row.set_sensitive(
            enabled and self.passes_row.get_value() > 1
        )

    def _on_passes_changed(
        self, spin_row: SpinRow, z_step_row: LengthSpinRow
    ) -> None:
        new_value = spin_row.get_int_value()
        z_step_row.set_sensitive(new_value > 1)
        self.param_changed.send(
            self,
            key="passes",
            value=new_value,
            name=_("Change number of passes"),
        )

    def _on_z_step_down_changed(self, row: LengthSpinRow) -> None:
        new_value = row.get_value_in_base_units()
        self.param_changed.send(
            self,
            key="z_step_down",
            value=new_value,
            name=_("Change Z Step-Down"),
        )
