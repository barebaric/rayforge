from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.core.undo import DictItemCommand
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.shared.pref_rows.base import SpinRow
from rayforge.ui_gtk.shared.pref_rows.length_spin_row import LengthSpinRow

from ..transformers import MultiPassTransformer

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class MultiPassSettingsGroup(DebounceMixin, TransformerSettingsGroup):
    """UI for configuring the MultiPassTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: MultiPassTransformer,
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

        # Passes setting
        passes_row = SpinRow(
            _("Number of Passes"),
            _("How often to repeat the entire step"),
            lower=1,
            upper=100,
            value=transformer.passes,
        )
        self.add(passes_row)

        # Z Step-down setting
        z_step_row = LengthSpinRow(
            _("Z Step-Down per Pass"),
            _("Distance to lower Z-axis for each subsequent pass"),
            upper=50.0,
            value_in_base=transformer.z_step_down,
        )
        self.add(z_step_row)
        z_step_row.value_changed.connect(
            lambda r: self._debounce(self._on_z_step_down_changed, r)
        )

        # Connect signals with debouncing
        passes_row.value_changed.connect(
            lambda r: self._debounce(self._on_passes_changed, r, z_step_row),
        )

        # Z Step-down is only available with multiple passes
        if transformer.passes <= 1:
            z_step_row.set_sensitive(False)

    def _update_sensitivity(self):
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        passes_row = self._rows[1]
        z_step_row = self._rows[2]
        passes_row.set_sensitive(enabled)
        z_step_row.set_sensitive(enabled and passes_row.get_value() > 1)

    def _on_passes_changed(self, spin_row, z_step_row: LengthSpinRow):
        new_value = spin_row.get_int_value()
        z_step_row.set_sensitive(new_value > 1)
        if new_value == self.target_dict.get("passes"):
            return

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="passes",
            new_value=new_value,
            name=_("Change number of passes"),
            on_change_callback=self.step.per_step_transformer_changed.send,
        )
        self.history_manager.execute(command)

    def _on_z_step_down_changed(self, row):
        new_value = row.get_value_in_base_units()
        if new_value == self.target_dict.get("z_step_down"):
            return

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="z_step_down",
            new_value=new_value,
            name=_("Change Z Step-Down"),
            on_change_callback=self.step.per_step_transformer_changed.send,
        )
        self.history_manager.execute(command)
