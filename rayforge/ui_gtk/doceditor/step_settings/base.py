from typing import TYPE_CHECKING, Union, Dict, Any

from gi.repository import Adw

from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from ....core.step import Step
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
        component: Union[OpsProducer, OpsTransformer],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        """
        Initializes the base widget.

        Args:
            editor: The DocEditor instance.
            title: The title for the preferences group.
            component: The OpsProducer or OpsTransformer instance this widget
                will modify.
            page: The parent Adw.PreferencesPage to which conditional groups
                  can be added or removed.
            step: The parent Step object, for context and signaling.
        """
        super().__init__(title=title, **kwargs)
        self.editor = editor
        self.component = component
        self.page = page
        self.step = step
        self.history_manager = editor.history_manager
        self._rows = []

    def add(self, child):
        self._rows.append(child)
        if not getattr(self.page, "use_expanders", False):
            super().add(child)

    @property
    def target_dict(self) -> Dict[str, Any]:
        """
        Get the dictionary backing this component.

        For producers, returns step.opsproducer_dict.
        For transformers, finds matching dict from step's
        transformer lists.
        """
        if isinstance(self.component, OpsProducer):
            result = self.step.opsproducer_dict
            if result is None:
                raise ValueError("Step has no opsproducer_dict")
            return result

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
