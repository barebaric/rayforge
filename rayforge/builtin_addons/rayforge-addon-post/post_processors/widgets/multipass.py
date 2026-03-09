from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk
from gettext import gettext as _

from rayforge.core.undo import DictItemCommand
from rayforge.pipeline.transformer.multipass_transformer import (
    MultiPassTransformer,
)
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float, get_spinrow_int
from rayforge.shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class MultiPassSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
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
        passes_adj = Gtk.Adjustment(
            lower=1, upper=100, step_increment=1, page_increment=10
        )
        passes_row = Adw.SpinRow(
            title=_("Number of Passes"),
            subtitle=_("How often to repeat the entire step"),
            adjustment=passes_adj,
        )
        passes_adj.set_value(transformer.passes)
        self.add(passes_row)

        # Z Step-down setting
        z_step_adj = Gtk.Adjustment(
            lower=0.0, upper=50.0, step_increment=0.1, page_increment=1.0
        )
        z_step_row = Adw.SpinRow(
            title=_("Z Step-Down per Pass"),
            subtitle=_(
                "Distance to lower Z-axis for each subsequent pass "
                "in machine units"
            ),
            adjustment=z_step_adj,
            digits=2,
        )
        z_step_adj.set_value(transformer.z_step_down)
        self.add(z_step_row)

        # Connect signals with debouncing
        passes_row.connect(
            "changed",
            lambda r: self._debounce(self._on_passes_changed, r, z_step_row),
        )
        z_step_row.connect(
            "changed",
            lambda r: self._debounce(self._on_z_step_down_changed, r),
        )

        # Set initial sensitivity
        z_step_row.set_sensitive(transformer.passes > 1)

    def _on_passes_changed(self, spin_row, z_step_row: Adw.SpinRow):
        new_value = get_spinrow_int(spin_row)
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

    def _on_z_step_down_changed(self, spin_row):
        new_value = get_spinrow_float(spin_row)
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
