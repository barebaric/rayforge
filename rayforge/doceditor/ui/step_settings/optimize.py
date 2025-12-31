from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Adw
from .base import StepComponentSettingsWidget
from ....core.undo import DictItemCommand
from ....pipeline.transformer import Optimize

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class OptimizeSettingsWidget(StepComponentSettingsWidget):
    """UI for configuring the Optimize transformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        transformer = Optimize.from_dict(target_dict)

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
        switch_row = Adw.SwitchRow(title=_("Enable Optimization"))
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        # Connect signals
        switch_row.connect("notify::active", self._on_enable_toggled)

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.target_dict.get("enabled"):
            return

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Path Optimization"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
