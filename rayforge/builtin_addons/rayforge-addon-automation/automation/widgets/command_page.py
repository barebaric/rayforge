"""Command step settings page: recipe-varset-driven machine-code editor."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.pages.base import (
    StepSettingsPage,
)


class CommandStepSettingsPage(StepSettingsPage):
    """Settings page for the CommandStep.

    Renders the step's recipe varset — a single multi-line "Machine
    Code" editor — through the standard varset machinery, so edits are
    debounced and undo-able, the model resyncs on undo or external
    edits, and the recipe editor offers the same setting.
    """

    def __init__(self, editor: Any, step: Any):
        super().__init__(editor, step)
        self._add_step_sections()

    def _add_step_sections(self):
        self.add_varset_section(
            None,
            self.step.recipe_varset(),
            description=_(
                "One command per line. Path variables are expanded at "
                "encode time."
            ),
        )
