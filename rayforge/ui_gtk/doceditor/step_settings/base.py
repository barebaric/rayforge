from typing import Dict, Any, TYPE_CHECKING
from gi.repository import Adw

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class StepComponentSettingsWidget(Adw.PreferencesGroup):
    """
    An abstract base class for a self-contained UI widget that manages the
    settings for a single pipeline component (a Producer or Transformer).

    Subclasses are responsible for building their own UI rows and connecting
    signals to update the provided component model's dictionary representation.
    """

    # Class property: override to False to hide general settings
    # (power, speed, air assist)
    show_general_settings = True

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        """
        Initializes the base widget.

        Args:
            editor: The DocEditor instance.
            title: The title for the preferences group.
            target_dict: The dictionary from the Step model (e.g.,
                step.opsproducer_dict or an item from
                step.per_workpiece_transformers_dicts) that this widget
                will modify.
            page: The parent Adw.PreferencesPage to which conditional groups
                  can be added or removed.
            step: The parent Step object, for context and signaling.
        """
        super().__init__(title=title, **kwargs)
        self.editor = editor
        self.target_dict = target_dict
        self.page = page
        self.step = step
        self.history_manager = editor.history_manager
