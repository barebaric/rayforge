from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.pipeline.transformer.base import OpsTransformer

from .base import StepComponentSettingsWidget

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class PlaceholderSettingsWidget(StepComponentSettingsWidget):
    """
    Error display for missing transformer widget.

    This widget is shown when a step's transformer type is not available.
    """

    show_general_settings = False

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        component: OpsTransformer,
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        super().__init__(
            editor=editor,
            title=title,
            component=component,
            page=page,
            step=step,
            **kwargs,
        )

        component_type = type(component).__name__

        error_row = Adw.ActionRow(
            title=_("This feature is not available."),
            subtitle=_(
                "The required component '{}' could not be found. "
                "The document can still be saved."
            ).format(component_type),
        )
        error_row.add_css_class("error")
        self.add(error_row)
