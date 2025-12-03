from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Gtk, Adw
from .base import StepComponentSettingsWidget
from ....pipeline.transformer import MultiPassTransformer
from ....undo import DictItemCommand
from ....shared.ui.adwfix import get_spinrow_int, get_spinrow_float
from ....shared.util.glib import DebounceMixin

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class MultiPassSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the MultiPassTransformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        # The transformer is stateless, so we can instantiate it for its
        # properties
        transformer = MultiPassTransformer.from_dict(target_dict)

        super().__init__(
            editor,
            title,
            description=transformer.description,
            target_dict=target_dict,
            page=page,
            step=step,
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
            title=_("Z Step-Down per Pass (mm)"),
            subtitle=_("Distance to lower Z-axis for each subsequent pass"),
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
