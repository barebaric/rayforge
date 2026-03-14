from typing import TYPE_CHECKING

from gi.repository import Adw
from gettext import gettext as _

from rayforge.core.undo import DictItemCommand
from ..transformers import Optimize
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
        switch_row = Adw.SwitchRow(
            title=_("Enable Optimization"),
            subtitle=_("Minimizes travel distance by reordering segments"),
        )
        switch_row.set_active(transformer.enabled)
        self.add(switch_row)

        # Connect signals
        switch_row.connect("notify::active", self._on_enable_toggled)

        self.flip_row = Adw.SwitchRow(
            title=_("Allow Flipping"),
            subtitle=_("Allow reversing path direction for shorter travel"),
        )
        self.flip_row.set_active(transformer.allow_flip)
        self.flip_row.set_sensitive(transformer.enabled)
        self.add(self.flip_row)
        self.flip_row.connect("notify::active", self._on_flip_toggled)

        self.preserve_row = Adw.SwitchRow(
            title=_("Preserve First Workpiece"),
            subtitle=_("Keep the first workpiece at its original position"),
        )
        self.preserve_row.set_active(transformer.preserve_first)
        self.preserve_row.set_sensitive(transformer.enabled)
        self.add(self.preserve_row)
        self.preserve_row.connect(
            "notify::active", self._on_preserve_first_toggled
        )

    def _on_enable_toggled(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.target_dict.get("enabled"):
            return

        self.flip_row.set_sensitive(new_value)
        self.preserve_row.set_sensitive(new_value)

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle Path Optimization"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_flip_toggled(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.target_dict.get("allow_flip"):
            return

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="allow_flip",
            new_value=new_value,
            name=_("Toggle Flipping"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)

    def _on_preserve_first_toggled(self, row, pspec):
        new_value = row.get_active()
        if new_value == self.target_dict.get("preserve_first"):
            return

        command = DictItemCommand(
            target_dict=self.target_dict,
            key="preserve_first",
            new_value=new_value,
            name=_("Toggle Preserve First Workpiece"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)
