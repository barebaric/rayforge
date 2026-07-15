from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gi.repository import Adw

from rayforge.core.undo import ChangePropertyCommand
from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor


class StepComponentSettingsWidget(Adw.PreferencesGroup):
    """
    Base class for settings widgets managing a Producer or Transformer.

    Subclasses build UI rows and connect signals to update the component's
    state via the step's dictionary representation.
    """

    # Class property: override to False to hide general settings
    # (power, speed, air assist)
    show_general_settings = True

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        page: Adw.PreferencesPage,
        step: Any,
        component: Optional[OpsTransformer] = None,
        **kwargs,
    ):
        """
        Initializes the base widget.

        Args:
            editor: The DocEditor instance.
            title: The title for the preferences group.
            page: The parent Adw.PreferencesPage to which conditional groups
                  can be added or removed.
            step: The parent Step object, for context and signaling.
            component: Optional OpsTransformer instance. When provided,
                       an enable/disable switch is added automatically.
        """
        super().__init__(title=title, **kwargs)
        self.editor = editor
        self.component = component
        self.page = page
        self.step = step
        self.history_manager = editor.history_manager
        self._rows: List = []
        self.enable_switch: Optional[Adw.SwitchRow] = None

        if isinstance(component, OpsTransformer):
            self._add_enable_switch(component)

    def add(self, child):
        self._rows.append(child)
        if not getattr(self.page, "use_expanders", False):
            super().add(child)
        if self.enable_switch is not None and child is not self.enable_switch:
            child.set_sensitive(self.enable_switch.get_active())

    def _add_enable_switch(self, component):
        switch_row = Adw.SwitchRow(
            title=_("Enable {}").format(component.label),
        )
        switch_row.set_active(component.enabled)
        self.add(switch_row)
        self.enable_switch = switch_row
        switch_row.connect("notify::active", self._on_enable_toggled)

    def _on_enable_toggled(self, row, pspec):
        assert isinstance(self.component, OpsTransformer)
        new_value = row.get_active()
        self.editor.step.set_step_param(
            target_dict=self.target_dict,
            key="enabled",
            new_value=new_value,
            name=_("Toggle {}").format(self.component.label),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
        self._update_sensitivity()

    def _update_sensitivity(self):
        assert self.enable_switch is not None
        enabled = self.enable_switch.get_active()
        for row in self._rows[1:]:
            row.set_sensitive(enabled)

    def is_unsupported(self) -> bool:
        """
        Whether this component is enabled but cannot take effect on the
        active machine (e.g. the driver handles the feature itself).

        Subclasses override this to flag expander-level warnings. Returns
        False by default.
        """
        return False

    def set_step_property(
        self,
        key: str,
        new_value: Any,
        name: Optional[str] = None,
    ):
        """Set a step attribute with an undoable command.

        Args:
            key: The step attribute name.
            new_value: The new value for the attribute.
            name: The command name for the undo stack.
        """
        current = getattr(self.step, key, None)
        if current == new_value:
            return

        def _notify():
            self.step.updated.send(self.step)

        command = ChangePropertyCommand(
            target=self.step,
            property_name=key,
            new_value=new_value,
            name=name or _(
                "Change {key}"
            ).format(key=key.replace("_", " ")),
            on_change_callback=_notify,
        )
        self.history_manager.execute(command)

    @property
    def target_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary backing the transformer component.
        """
        component_name = type(self.component).__name__
        for t_dict in self.step.per_workpiece_transformers_dicts or []:
            if t_dict.get("name") == component_name:
                return t_dict
        for t_dict in self.step.per_step_transformers_dicts or []:
            if t_dict.get("name") == component_name:
                return t_dict

        raise ValueError(
            f"Could not find dict for transformer: {component_name}"
        )
