from typing import Dict, Any, TYPE_CHECKING
from gettext import gettext as _
from gi.repository import Adw
from .base import StepComponentSettingsWidget

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class PlaceholderSettingsWidget(StepComponentSettingsWidget):
    """
    Error display for missing producer widget.

    This widget is shown when a step's producer type is not available
    (e.g., because the addon that provides it is not installed).
    """

    show_general_settings = False

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        producer_type = target_dict.get("type", "Unknown")

        super().__init__(
            editor=editor,
            title=title,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        error_row = Adw.ActionRow(
            title=_("This feature is not available."),
            subtitle=_(
                "The required component '{}' could not be found. "
                "The document can still be saved."
            ).format(producer_type),
        )
        error_row.add_css_class("error")
        self.add(error_row)
