from typing import TYPE_CHECKING

from gi.repository import Adw
from gettext import gettext as _

from rayforge.core.undo import DictItemCommand
from rayforge.pipeline.transformer.optimize_transformer import Optimize
from rayforge.ui_gtk.doceditor.step_settings.base import (
    StepComponentSettingsWidget,
)

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.doceditor.editor import DocEditor


class OptimizeSettingsWidget(StepComponentSettingsWidget):
    """UI for configuring the Optimize transformer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        transformer: Optimize,
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
